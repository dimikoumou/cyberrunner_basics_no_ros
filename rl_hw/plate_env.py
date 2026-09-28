"""
rl_hw/plate_env.py

Gymnasium environment that trains a goal-conditioned ball-balancing policy directly
against the real CyberRunner hardware (camera + two Dynamixel motors), with the maze
insert swapped for a flat glass plate. No simulator involved -- this talks to the
camera and motors every step. See rl_sim/maze_env.py for the earlier sim-only
prototype whose observation convention this mirrors.

Prerequisite: state_est/markers.csv must hold the corner-marker calibration for the
current physical setup (recalibrate it if the camera/markers moved since the maze was
last used -- see state_est/camera_calibration_realtime.py / board_detection.py).
"""
import os
import sys
import glob
import json
import time
import threading

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

# estimation_pipeline.py and its sibling modules use flat imports (`from measurements
# import Measurements`, etc.) and load markers.csv / calib_razer_data.txt relative to
# the working directory -- so both the path and the cwd need to point at state_est/.
STATE_EST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "state_est"))
sys.path.insert(0, STATE_EST_DIR)
os.chdir(STATE_EST_DIR)

from estimation_pipeline import EstimationPipeline  # noqa: E402
from divers import init_capture  # noqa: E402
import state_est_control  # noqa: E402
from state_est_control import init_dynamixel, set_position, ADDR_TORQUE_ENABLE  # noqa: E402

# state_est_control.DXL_IDS = [1, 2] is wrong for tilt control: confirmed 2026-08-26
# that ID 2 is the unrelated ball-reload elevator motor (it never moved the plate --
# present-position readback was frozen regardless of command -- while the actual
# second tilt axis is ID 3, confirmed via a real alpha/beta response). Override here
# rather than editing state_est_control.py.
DXL_IDS = [1, 3]


def _find_rig_camera(want_wh=(1920, 1080), max_index=4):
    """Index of the rig camera. macOS renumbers cameras when USB devices are
    re-plugged (2026-09-26: the See3CAM moved from index 0 to 1 and index 0 became
    the Mac's built-in camera, so the pipeline silently looked at the wrong
    camera). Pick the first camera that really delivers want_wh frames -- the
    built-in camera tops out at 1280x720. CYBERRUNNER_CAM=<index> overrides."""
    if os.environ.get("CYBERRUNNER_CAM") is not None:
        return int(os.environ["CYBERRUNNER_CAM"])
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" else cv2.VideoCapture(idx)
        try:
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_wh[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_wh[1])
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
            if ok and frame is not None and frame.shape[1] == want_wh[0] and frame.shape[0] == want_wh[1]:
                return idx
        finally:
            cap.release()
    raise RuntimeError(f"no camera delivering {want_wh[0]}x{want_wh[1]} found -- is the rig camera plugged in?")


def _resolve_dynamixel_port():
    """state_est_control.DXL_PORT is a hardcoded device name tied to one specific
    USB-serial adapter's enumerated suffix (e.g. /dev/tty.usbserial-FT79212K) -- this
    breaks any time the OS assigns a different suffix (different cable, different USB
    port, different machine). Auto-detect the live port instead of trusting it."""
    if os.path.exists(state_est_control.DXL_PORT):
        return state_est_control.DXL_PORT
    candidates = sorted(glob.glob("/dev/tty.usbserial-*"))
    if not candidates:
        raise RuntimeError(
            "No /dev/tty.usbserial-* device found -- is the U2D2 plugged in and powered? "
            f"(hardcoded fallback {state_est_control.DXL_PORT!r} also not present)"
        )
    if len(candidates) > 1:
        print(f"[HardwarePlateEnv] warning: multiple usbserial devices found {candidates}, using {candidates[0]}")
    return candidates[0]

CALIBRATION_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "motor_calibration.json")
)
# _level_plate() always started its search from this static file's center every
# single call -- fine when the true level point drifts a little, but observed
# directly over a long session to swing across nearly the ENTIRE +-1000-tick search
# range between separate calls (needing the search to hit first one bound, then the
# opposite bound, on consecutive runs). Restarting from a known-stale reference every
# time means paying that full, slow, unreliable search on every single call. Persist
# whatever position last actually converged and start from there instead -- usually
# already close, so most calls converge in a couple of quick iterations.
LAST_LEVEL_CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "last_level_position.json")
)

# Outer/fixed corner markers, measured once by rl_hw/measure_fixed_markers.py (or
# from saved clean frames) -- see _prime_camera_localization for why.
FIXED_CORNERS_CACHE_PATH = os.path.join(STATE_EST_DIR, "fixed_corners_cache.json")

# Closed-loop tilt control (2026-09-26). Full-range sweeps on the rig
# (rl_hw/characterize_tilt.py) measured motor3->beta -8.4..+10.6deg at ~100
# ticks/deg and motor1->alpha +6.2..-5.6deg at ~140 ticks/deg -- but the tick at
# which a given angle occurs was NOT repeatable on motor3 (m3=2300 read +5.4deg
# once and -1.7deg later), so any fixed action->tick map (the old approach)
# commands the wrong tilt. Instead an action is a TARGET ANGLE and the motors are
# corrected every step against the camera-measured angle.
TILT_MAX_DEG = 5.0                    # |action|=1 -> 5deg, inside both axes' measured range
TICKS_PER_DEG = {1: -140.0, 3: 100.0}  # d(ticks)/d(angle): +m1 lowers alpha, +m3 raises beta
TILT_GAIN = 0.15                       # per-step correction. 0.5 shook the plate (2-3 step lag @22 Hz); 0.25 was fine @22 Hz but jittered 2.5deg once the 30 fps camera fix made the loop ~29 Hz (lag ~4-5 steps)
MAX_TICKS_PER_STEP = 250
# Fixed level position (open loop, no camera): used whenever the ball is not found /
# the elevator runs (2026-09-27, user: "just bring it level when ball not found").
# The camera-closed-loop level once chased a wrong pose to the tick limits.
LEVEL_TICKS = {1: int(os.environ.get("PLATE_LEVEL_M1", "2952")), 3: int(os.environ.get("PLATE_LEVEL_M3", "2812"))}
# The tilt servo trusts the camera only if its reading roughly agrees with the motor
# position (LEVEL_TICKS + TICKS_PER_DEG): a ball right beside a plate marker makes the
# reading wrong AND unresponsive, and the servo then ran motor 3 to its limit (3800) in
# ~1 s. Hard band: never further than TILT_BAND_DEG from the level position.
TILT_TRUST_DEG = 3.0
TILT_BAND_DEG = 6.5
# Direct motor mapping (default, 2026-09-27): ticks = LEVEL_TICKS + (target - level) x
# TICKS_PER_DEG, no camera in the tilt loop at all. Every camera-corrected variant broke
# when the ball sat beside a plate marker (reading wrong by 5-15 deg and unresponsive).
TILT_OPEN_LOOP = os.environ.get("PLATE_TILT_OPEN_LOOP") == "1"     # default: camera closed loop (user)
# Camera closed loop guards (2026-09-27): the runaways to the tick limit happened while the
# plate was NOT following motor 3 (its link slipped): the loop kept adding ticks. If a motor
# has moved STALL_TICKS by correction without the measured angle improving by STALL_GAIN_DEG,
# further correction in that direction is refused (and logged) until the angle responds.
STALL_TICKS = 700       # ~7 deg: motor 3's link has ~300 ticks of play before the plate follows
STALL_GAIN_DEG = 0.3
STALL_RETARGET_DEG = 2.0   # a new command (target moved this much) is a fresh attempt
TICK_BOUNDS = {1: (1800, 3900), 3: (700, 3800)}  # where each axis's angle plateaus (measured)
LEVEL_TOL_DEG = 0.4
# Delay-aligned correction (research workflow 2026-09-26, Smith-predictor idea):
# the camera tilt is 4 (alpha) / 5 (beta) steps old at the ~29 Hz loop (lag from
# cross-correlation of command vs measured tilt), so comparing it with the CURRENT
# target re-corrected every target change on top of the feedforward -- a -0.9deg
# brake command reached the plate as -2.1..-2.4deg and reversed the ball. Compare
# each measurement with the target that was active when that frame was taken.
TILT_MEAS_DELAY_STEPS = {1: 4, 3: 5}
CAMERA_STALL_S = 2.0
# Camera-frame angle at which the ball does NOT accelerate, i.e. true gravity level
# expressed in the camera's (outer-frame) world frame. Estimated 2026-09-26 by
# regressing ball acceleration on measured tilt over a whole PD run
# (pd_20260926_142924.csv, n=1133 off-wall samples, r=0.92 on both axes):
# ax = 0 at beta=+3.00deg, ay = 0 at alpha=-0.82deg. Without it, "level" (0,0)
# rolled the ball into the (-x,-y) corner every time. Refit on the first run WITH
# the offset (pd_20260926_143132.csv, Savitzky-Golay accel, n=2759, r=0.98/0.97,
# run halves agree within 0.06deg): beta0=+2.55, alpha0=-1.13.
LEVEL_OFFSET_DEG = (-1.1, 2.55)  # (alpha, beta)

GOAL_MARGIN = 0.04          # meters, inward margin from the plate's usable extent
GOAL_TOLERANCE = 0.015      # meters, radius of the "balance here" circle
IN_CIRCLE_BONUS = 1.0       # per-step reward bonus while the ball is inside that circle
LOST_BALL_PENALTY = -1.0
BALL_LOST_GRACE_FRAMES = 3  # consecutive not-found frames tolerated before terminating
# A ball wedged against the plate's physical edge/frame reads as ball_found=True the
# whole time (it's visible, just not moving) -- BALL_LOST_GRACE_FRAMES never catches
# this. Observed directly: a ball at the edge stayed frozen to the 4th decimal place
# across 150 steps of a real, meaningful, constant tilt command. Detect it instead by
# watching for "commanding real tilt but the ball isn't responding at all".
STUCK_CHECK_WINDOW = 25          # steps of history to check
STUCK_MOVEMENT_THRESHOLD = 0.004  # meters of total range below which it's "not moving"
STUCK_ACTION_THRESHOLD = 0.15     # mean |action| above which it's "being meaningfully driven"

ADDR_CURRENT_LIMIT = 38  # X-series, 2 bytes
CURRENT_LIMIT = 1193     # verified real max for this motor model -- 1800 is rejected as out-of-range
ADDR_POSITION_I_GAIN = 82  # X-series, 2 bytes, RAM (writable with torque enabled)
POSITION_I_GAIN = 150
# Read directly off the servos: both motor1 and motor3 have P=400 I=0 D=400 -- I=0
# means a Position-Control servo has NO way to fully cancel a steady external load
# (gravity pulling the tilted plate back through the linkage); it settles at a
# droop proportional to that load instead of reaching the exact commanded tick.
# Confirmed directly: holding motor1 at a fixed target under a fixed load left it
# 46-92 ticks short with I=0, using close to zero current (not even trying hard),
# and the gap grew with distance from center (more load -> more droop) -- textbook
# P/D-only steady-state error, not a mechanical stall. The SAME symptom (a servo
# that appears to peg against the search bounds without reaching true alpha/beta=0)
# was misread earlier as motor3's real physical range being exhausted. With I=150,
# the same test closed to a 2-6 tick gap. This is very likely the real cause of
# most of this session's leveling/holding difficulty, not exhausted physical range.


def _load_motor_calibration(path):
    """motor_calibration.json holds empirically-probed safe min/center/max per motor
    (present-position stall detection -- see the "_note" field in the file itself for
    the 2026-08-26 correction: motor2_id is 3, not 2, since ID 2 turned out to be the
    unrelated ball-reload elevator motor). Still clamps to a SYMMETRIC usable range
    around center (using the smaller of the two half-ranges) as a defensive backstop
    in case the file's min/max aren't already symmetric -- equal action magnitude
    should produce comparable physical response in both directions, or the
    action->effect mapping becomes much harder to learn from."""
    with open(path) as f:
        cal = json.load(f)
    result = {}
    for name in ("motor1", "motor2"):
        motor_id = cal[f"{name}_id"]
        min_pos, center_pos, max_pos = cal[f"{name}_min_position"], cal[f"{name}_center_position"], cal[f"{name}_max_position"]
        half_range = min(center_pos - min_pos, max_pos - center_pos)
        if half_range < (center_pos - min_pos) or half_range < (max_pos - center_pos):
            print(f"[HardwarePlateEnv] {name}: clamping to a symmetric +-{half_range} ticks around "
                  f"center={center_pos} (raw calibration was -{center_pos - min_pos}/+{max_pos - center_pos})")
        result[motor_id] = (center_pos - half_range, center_pos, center_pos + half_range)
    return result


def _symmetric_range(min_pos, center_pos, max_pos):
    """(center-h, center, center+h) with h = the smaller side. 2026-09-26: with the
    leveled center sitting far from the middle of the safe bounds (e.g. m3 level
    2515 in 1926..4090), the piecewise map gave the same |action| a 2.7-8.6x
    different tilt depending on direction -- the controller's effective gain
    flipped with the sign of its own output. Equal ticks per unit both ways."""
    h = min(center_pos - min_pos, max_pos - center_pos)
    return (center_pos - h, center_pos, center_pos + h)


def map_action_to_position(a, min_pos, center_pos, max_pos):
    """Piecewise-linear map: a in [-1,0] -> [min_pos, center_pos], a in [0,1] -> [center_pos, max_pos]."""
    a = float(np.clip(a, -1.0, 1.0))
    if a < 0:
        pos = center_pos + a * (center_pos - min_pos)
    else:
        pos = center_pos + a * (max_pos - center_pos)
    return int(round(pos))


class HardwarePlateEnv(gym.Env):
    """Goal-conditioned ball-on-plate env, driven directly by the real camera + motors.

    Observation (8,): [x_b, y_b, vx_b, vy_b, alpha, beta, goal_x - x_b, goal_y - y_b]
    Action (2,):       [a1, a2] in [-1, 1], mapped through each motor's calibrated
                        (min, center, max) position from motor_calibration.json.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        control_hz=55,
        goal_tolerance=GOAL_TOLERANCE,
        goal_margin=GOAL_MARGIN,
        max_episode_steps=600,
        camera_index=None,
        show_image=False,
        fixed_goal=None,
        max_action_delta=0.35,
        allow_unstick=False,
        tilt_control=True,
    ):
        """
        fixed_goal: None keeps the original behavior (a new random goal every
            episode -- the policy learns to go anywhere it's told). Pass "center" or
            an explicit (x, y) in meters to instead train/hold a single fixed target
            every episode -- simpler task, useful as a first "does this even work"
            validation before goal-conditioned training.
        max_action_delta: caps how much the ACTUAL commanded action can change from
            the previous step, regardless of what's requested. SAC's pre-learning_starts
            random phase samples a fresh independent action every step, which without
            this would slam the plate between opposite extremes every ~18ms -- hard on
            the hardware and low-value training data. The agent still sees/learns over
            the full [-1,1] action space; this only rate-limits actuation.
        allow_unstick: default False -- disables _attempt_unstick() entirely (no sweep, no
            violent shake). Those escapes only existed for the glass bezel ledge;
            with the glass removed (flat paper, 2026-09-26) they just add vibration
            that can drop the U2D2. A wedged ball then simply ends the episode.
        """
        super().__init__()
        self._closed = False  # set first, so close()/__del__ work even if startup fails
        self._motors_torqued = False
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz
        self.goal_tolerance = goal_tolerance
        self.goal_margin = goal_margin
        self.max_episode_steps = max_episode_steps
        self.fixed_goal = fixed_goal
        self.max_action_delta = max_action_delta
        self.allow_unstick = allow_unstick
        self.tilt_control = tilt_control
        self._cmd_ticks = {}           # last commanded goal position per motor (tilt_control)
        self._tilt_target = LEVEL_OFFSET_DEG  # last target (alpha, beta) in degrees
        self._meas_tilt = None          # last plausible measured (alpha, beta) in degrees
        self._stall = {}                # dxl_id -> correction-without-response tracker (see STALL_TICKS)
        self._pose_ok = True
        self._target_hist = []          # recent (alpha, beta) targets, newest last
        self._last_commanded_action = np.zeros(2, dtype=np.float32)
        self._pos_history = []
        self._action_history = []

        self.calibration = _load_motor_calibration(CALIBRATION_PATH)
        # Overwritten every _level_plate() call with the position that leveling
        # actually achieved -- see the comment in _write_action() for why this
        # matters: mapping action=0 through the STATIC calibration center instead
        # of the just-achieved level position was a real, severe bug.
        self._effective_calibration = dict(self.calibration)

        self.pipeline = EstimationPipeline(
            fps=control_hz,
            estimator="FiniteDiff",
            FiniteDiff_mean_steps=4,  # 0 produces NaN velocities out of the estimator
            print_measurements=False,
            show_image=show_image,
        )
        self._exp_h = int(self.pipeline.measurements.plate_pose.o.height)
        self._exp_w = int(self.pipeline.measurements.plate_pose.o.width)
        # The ball's own (xb, yb) is measured in the plate's MODEL_POINTS_CORNERS
        # frame (plate_pose.py), which is CENTERED AT (0,0) -- half-extents are
        # C2C_X/2, C2C_Y/2, not [0, L_EXT_INT_X/Y] (that pair is used for a
        # different purpose: create_mask's outer valid-pixel-region check). Using
        # L_EXT_INT_X/Y here previously placed "center" and every randomly sampled
        # goal outside the ball's actual reachable area entirely.
        self._x_half = float(self.pipeline.measurements.plate_pose.C2C_X) / 2
        self._y_half = float(self.pipeline.measurements.plate_pose.C2C_Y) / 2

        if camera_index is None:
            camera_index = _find_rig_camera()
        print(f"[HardwarePlateEnv] using camera index {camera_index}")
        self.cap, _, _ = init_capture("CAM", camera_index, None, None)
        # init_capture() requests 55 fps. Measured 2026-09-26 (See3CAM_24CUG,
        # 1920x1080, USB 3): at the 55 fps request the camera delivered ~43 fps but
        # with ~1.03 s of image delay (motor step -> pixels changing), which made the
        # tilt servo swing the plate side to side; at 30 fps the delay is ~0.15 s.
        # The control loop only consumes ~22 frames/s, so 30 fps loses nothing.
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("HardwarePlateEnv: could not open the camera")
        self._lock_camera_exposure()
        self._start_frame_reader()

        state_est_control.DXL_PORT = _resolve_dynamixel_port()
        self.port_handler, self.packet_handler = init_dynamixel()
        if self.port_handler is None:
            raise RuntimeError("HardwarePlateEnv: could not open the Dynamixel port")
        # Current limit was never explicitly set anywhere in this codebase's history --
        # motors were running on whatever default happened to be in EEPROM (600/800),
        # far below what this mechanism's friction needs. 1193 is the real verified max
        # for this motor model (1800, used in an old calibration script, is invalid and
        # was silently rejected there too). Must be written while torque is disabled.
        for dxl_id in DXL_IDS:
            self.packet_handler.write2ByteTxRx(self.port_handler, dxl_id, ADDR_CURRENT_LIMIT, CURRENT_LIMIT)
        for dxl_id in DXL_IDS:
            self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, 1)
        self._motors_torqued = True
        # Position I Gain is RAM (writable with torque already enabled) -- must be
        # written AFTER torque-enable above, or it has no chance to take effect
        # before the very first set_position() call a few lines down.
        for dxl_id in DXL_IDS:
            self.packet_handler.write2ByteTxRx(self.port_handler, dxl_id, ADDR_POSITION_I_GAIN, POSITION_I_GAIN)

        # Drive to the calibrated (level) center BEFORE priming the camera's world-frame
        # reference. Previously this ran before torque was even enabled, so the plate was
        # unpowered and sagging to gravity's whim (wherever the last process left it right
        # before disabling torque on close) -- an uncontrolled tilt during the one-shot
        # reference-corner calibration. That's not just cosmetic: it plausibly explains the
        # multi-degree, run-to-run bias in reported beta seen throughout calibration testing
        # (each run baked in a different incidental resting tilt), and on at least one run
        # left the plate tilted enough to fully occlude a fixed reference corner from the
        # camera, crashing this method outright. Calibrating from a known, controlled,
        # already-torqued position removes that variable.
        # 2026-09-26: start from the cached level position when there is one. The
        # static calibration center for motor3 (2476) is ~1400 ticks off level
        # since the homing-offset change, i.e. a hard tilt that rolls the ball
        # straight into a corner before the first episode even starts.
        startup = {dxl_id: c for dxl_id, (_, c, _) in self.calibration.items()}
        try:
            with open(LAST_LEVEL_CACHE_PATH) as f:
                cached = json.load(f)
            m1_id, m3_id = sorted(self.calibration.keys())
            startup[m1_id], startup[m3_id] = int(cached["m1"]), int(cached["m3"])
        except (OSError, KeyError, ValueError):
            pass
        if os.environ.get("PLATE_OPEN_LOOP_LEVEL") == "1":
            startup.update(LEVEL_TICKS)          # the fixed, measured level position
        for dxl_id, target in startup.items():
            set_position(self.port_handler, self.packet_handler, dxl_id, target)
            self._cmd_ticks[dxl_id] = int(target)
        time.sleep(1.5)

        self._prime_camera_localization()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.goal = np.zeros(2, dtype=np.float32)
        self._prev_ball = np.zeros(2, dtype=np.float32)
        self._prev_t = None
        self._last_plausible_tilt = (0.0, 0.0)
        self._bad_tilt_streak = 0
        self._step_count = 0
        self._lost_count = 0
        self._closed = False

    def _lock_camera_exposure(self, settle_s=2.0):
        """
        divers.py's init_capture() never touches exposure/white-balance -- the
        camera runs on whatever auto-exposure/auto-WB the driver picks, continuously
        re-adjusting. That's a real, likely root cause behind corner/marker
        detection intermittently failing: direct inspection showed the tiny corner
        markers sitting right at the edge of their HSV threshold (max contour area
        ~21px, several times smaller than a comfortable margin) -- exactly what
        you'd expect if brightness/gain keeps drifting frame to frame instead of
        holding still. Let auto-exposure converge for a couple seconds (as before),
        then switch to manual mode AT WHATEVER VALUES it converged to -- freezing
        a good, lighting-appropriate exposure instead of guessing a fixed one, and
        stopping it from continuing to hunt during actual detection.
        """
        t0 = time.time()
        while time.time() - t0 < settle_s:
            self._grab_frame()
        exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        gain = self.cap.get(cv2.CAP_PROP_GAIN)
        wb = self.cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
        # CAP_PROP_AUTO_EXPOSURE's manual-mode value is backend-dependent (0.25 on
        # some V4L2 builds, 1 on others/AVFoundation) -- try the common ones and
        # verify by readback rather than assuming one is correct for this backend.
        locked = False
        for manual_value in (0.25, 1, 0):
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual_value)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            if abs(self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) - manual_value) < 1e-3:
                locked = True
                break
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if wb > 0:
            self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        if not locked:
            print("[HardwarePlateEnv] warning: could not confirm manual-exposure lock on this "
                  "camera backend -- auto-exposure may still be adjusting during detection")
        else:
            print(f"[HardwarePlateEnv] locked exposure={exposure} gain={gain} wb={wb}")

    def _prime_camera_localization(self, n_frames=40):
        """
        Measurements.camera_localization() (state_est/measurements.py) runs exactly
        once per pipeline instance, off a single captured frame, with no averaging --
        it uses that one frame's detection of the physically-static reference corners
        to fix the camera's world-frame pose for the rest of the process's lifetime.
        A single noisy corner detection there bakes in a persistent angular bias for
        the whole run. Confirmed by direct test: two separate HardwarePlateEnv
        instances, at the identical commanded motor positions, reported beta values
        differing by several degrees (-0.10 vs -4.24) purely because each one's
        one-shot calibration frame was independently noisy -- not a hardware issue,
        not a sign bug, not a motor problem.

        A plain mean over 15 frames (the original fix) was NOT enough: separate
        process launches this session, at nominally-identical plate positions,
        still swung by ~20 degrees in reported beta (e.g. +10.78 vs -5 to -9), and
        twice showed a hard corner-detection failure ("Unable to find corner 1/4")
        that still passed the >=n_frames//2 threshold and got silently averaged in
        alongside good frames. Outliers, not just noise, were getting into the mean.
        Fix: take more frames, then use the per-frame MEDIAN corner position (robust
        to occasional bad detections in a way a mean isn't), and separately verify
        no individual frame's corner positions strayed far from that median before
        trusting the result -- surfacing a real problem instead of quietly averaging
        over it.
        """
        # Let auto-exposure/auto-white-balance settle after a fresh camera open --
        # a cold-started camera's first frames can differ enough in brightness/gain
        # to shift the reference corners' detected sub-pixel position between runs,
        # which this axis's geometry is disproportionately sensitive to.
        t0 = time.time()
        while time.time() - t0 < 2.0:
            self._grab_frame()

        measurements = self.pipeline.measurements
        # 2026-09-26: the fixed markers never move, but since the glass came out the
        # two left-side ones are intermittently half-hidden behind a strip of the
        # tilting frame (history-dependent, not a function of the commanded
        # posture). A half-hidden marker leaves ~5px of blue, detection falls back
        # onto a wrong blob, and the pose comes out ~30deg off every attempt. Use the
        # positions measured once from frames where all 4 were cleanly visible.
        if os.path.exists(FIXED_CORNERS_CACHE_PATH):
            with open(FIXED_CORNERS_CACHE_PATH) as f:
                cache = json.load(f)
            if tuple(cache["resolution_wh"]) != (self._exp_w, self._exp_h):
                raise RuntimeError(f"{FIXED_CORNERS_CACHE_PATH} was measured at {cache['resolution_wh']}, "
                                   f"pipeline expects {(self._exp_w, self._exp_h)}")
            pts = np.array(cache["corners_row_col"], dtype=np.float32)
            measurements.detector.fixed_corners = pts
            measurements.plate_pose.camera_localization(pts)
            R = measurements.plate_pose.T__W_C[:3, :3]
            if not (R[0, 0] > 0.9 and R[1, 1] < -0.9 and R[2, 2] < -0.9):
                raise RuntimeError(f"cached fixed corners give an implausible camera pose {np.diag(R)}")
            frame, t_wait = None, time.time()
            while frame is None:
                if time.time() - t_wait > 5.0:
                    raise RuntimeError("HardwarePlateEnv: no camera frame within 5s while priming")
                frame = self._grab_frame()
            measurements.create_mask(frame)
            print(f"[HardwarePlateEnv] camera localized from cached fixed corners "
                  f"({cache.get('measured', '?')}), rotation diagonal {np.round(np.diag(R), 3)}")
            return
        # Outlier-rejection on the 4 corner POINTS doesn't catch every failure mode:
        # confirmed directly that a bad detection can be wrong in a way that's
        # internally self-consistent across many frames (all agreeing on the same
        # wrong spot), which looks exactly like a confident inlier cluster to a
        # per-frame deviation check. The one thing that's independently checkable
        # is the RESULT: this camera is bolted down looking at a roughly-overhead,
        # roughly-fixed view of the plate, so a correct calibration should always
        # produce close to the SAME rotation matrix run to run -- confirmed
        # empirically across every known-good run this session (diagonal close to
        # (+1,-1,-1), small off-diagonal terms). A calibration whose result doesn't
        # look like that is almost certainly wrong regardless of how "confident"
        # the point detections were, so re-run the whole capture instead of
        # trusting it.
        MAX_LOCALIZATION_ATTEMPTS = 4
        frame = None
        for attempt in range(1, MAX_LOCALIZATION_ATTEMPTS + 1):
            detected = []
            for _ in range(n_frames):
                frame = self._grab_frame()
                if frame is None:
                    continue
                pts = measurements.detector_fixed_points.detect_corners(frame)
                if measurements.detector_fixed_points.corners_missing:
                    continue
                detected.append(pts)
            if len(detected) < max(3, n_frames // 2):
                print(f"[HardwarePlateEnv] priming attempt {attempt}: only got "
                      f"{len(detected)}/{n_frames} good reference-corner detections -- retrying")
                continue
            stacked = np.stack(detected, axis=0)  # (n_detected, 4 corners, 2)
            median_pts = np.median(stacked, axis=0)
            per_frame_max_dev = np.max(np.linalg.norm(stacked - median_pts[None], axis=2), axis=1)
            OUTLIER_PX = 3.0
            inliers = stacked[per_frame_max_dev <= OUTLIER_PX]
            n_outliers = len(stacked) - len(inliers)
            if n_outliers:
                print(f"[HardwarePlateEnv] priming attempt {attempt}: dropped {n_outliers}/"
                      f"{len(stacked)} outlier reference-corner detections (>{OUTLIER_PX}px "
                      f"from the cross-frame median)")
            if len(inliers) < max(3, n_frames // 4):
                print(f"[HardwarePlateEnv] priming attempt {attempt}: only {len(inliers)}/"
                      f"{len(stacked)} detections agreed with each other -- retrying")
                continue
            avg_pts = np.mean(inliers, axis=0)
            measurements.detector.fixed_corners = avg_pts
            measurements.plate_pose.camera_localization(avg_pts)
            R = measurements.plate_pose.T__W_C[:3, :3]
            plausible = R[0, 0] > 0.9 and R[1, 1] < -0.9 and R[2, 2] < -0.9
            if plausible:
                measurements.create_mask(frame)
                return
            print(f"[HardwarePlateEnv] priming attempt {attempt}: resulting camera pose looks "
                  f"physically implausible (rotation diagonal {np.diag(R)}, expected roughly "
                  f"(+1,-1,-1)) -- retrying")
        raise RuntimeError(
            f"HardwarePlateEnv: could not get a physically plausible camera calibration in "
            f"{MAX_LOCALIZATION_ATTEMPTS} attempts -- check the camera/lighting/markers"
        )

    # ---- hardware I/O ----------------------------------------------------

    def _start_frame_reader(self):
        """Read the camera continuously in a background thread and keep only the
        newest frame. The capture backend queues frames the camera delivers faster
        than the ~22 Hz control loop consumes them; measured 2026-09-26 after a USB
        re-plug (camera at ~43 fps): command->camera delay grew from ~0.1 s to
        ~1.0 s (motor itself arrives in 0.11 s), and the tilt servo swung the plate
        side to side chasing the past. Consuming at the camera's own rate keeps the
        queue empty regardless of fps or USB path."""
        self._frame_lock = threading.Condition()
        self._latest_frame = None
        self._latest_seq = 0
        self._returned_seq = 0
        self._reader_stop = False

        def _reader():
            # Camera watchdog (2026-09-26: the camera once stopped delivering frames
            # while still listed by macOS): after CAMERA_STALL_S without a frame,
            # release it and reopen the rig camera (index may have changed).
            last_ok, last_reopen = time.time(), 0.0
            while not self._reader_stop:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    now = time.time()
                    if now - last_ok > CAMERA_STALL_S and now - last_reopen > CAMERA_STALL_S:
                        last_reopen = now
                        print(f"[HardwarePlateEnv] no camera frame for {now - last_ok:.1f}s -- reopening the camera")
                        try:
                            self.cap.release()
                            idx = _find_rig_camera()
                            self.cap, _, _ = init_capture("CAM", idx, None, None)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                            print(f"[HardwarePlateEnv] camera reopened (index {idx})")
                        except Exception as e:
                            print(f"[HardwarePlateEnv] camera reopen failed: {e}")
                    time.sleep(0.005)
                    continue
                last_ok = time.time()
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_seq += 1
                    self._frame_lock.notify_all()

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

    def _grab_frame(self):
        if getattr(self, "_reader_thread", None) is not None:
            # newest frame, waiting (briefly) for one we haven't returned yet
            with self._frame_lock:
                self._frame_lock.wait_for(lambda: self._latest_seq > self._returned_seq, timeout=0.5)
                frame = self._latest_frame
                self._returned_seq = self._latest_seq
            ok = frame is not None
        else:
            ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if frame.shape[:2] != (self._exp_h, self._exp_w):
            frame = cv2.resize(frame, (self._exp_w, self._exp_h))
        return frame

    def _read_state(self):
        """Returns (xb, yb, alpha, beta, ball_found)."""
        frame = self._grab_frame()
        if frame is None:
            return np.nan, np.nan, 0.0, 0.0, False
        try:
            _, _, inputs, xb, yb = self.pipeline.estimate(frame)
            self._last_frame = frame  # the frame the current plate pose belongs to
            cb = getattr(self, "frame_callback", None)  # e.g. the web UI's live view
            if cb is not None:
                try:
                    cb(frame)
                except Exception as e:  # the UI must never break the control loop
                    print(f"[HardwarePlateEnv] frame callback error: {e}")
        except Exception as e:
            print(f"[HardwarePlateEnv] estimate() failed on this frame, skipping: {e}")
            return np.nan, np.nan, 0.0, 0.0, False
        alpha, beta = inputs
        ball_found = not (np.isnan(xb) or np.isnan(yb))
        # a plate marker not found this frame -> the detector used the middle of its search
        # window: the tilt is wrong, so the servo must not correct on it (see _servo_tilt)
        self._pose_ok = not bool(getattr(self.pipeline.measurements.detector, "corners_missing", False))
        # Defense in depth: a mis-detected blob (e.g. a large false-positive region
        # under bad lighting getting through the ball detector) can still produce a
        # numeric, non-NaN position -- just a physically impossible one. Observed
        # directly: xb/yb of +-0.5+ meters when the plate's actual half-extents are
        # ~0.14/0.12m. Treat anything meaningfully outside the plate as not-found
        # rather than trusting it into the reward/goal-distance calculation.
        if ball_found and (abs(xb) > self._x_half * 1.5 or abs(yb) > self._y_half * 1.5):
            print(f"[HardwarePlateEnv] rejecting physically implausible ball position "
                  f"({xb:.3f}, {yb:.3f}) -- treating as not found")
            return np.nan, np.nan, alpha, beta, False
        # Same defense-in-depth idea, for plate tilt: a violent shake can throw the
        # continuous reference-corner TRACKING (not the one-shot priming -- this can
        # happen mid-run, well after that) onto the wrong nearby blob, which then
        # self-confirms frame to frame since tracking searches near its own last
        # answer. Observed directly: alpha/beta suddenly reading +53/-46 degrees --
        # far outside every real, direct-probed achievable range this session
        # (worst case ~+-20deg) -- which then fed a leveling hill-climb chasing a
        # target that was never real. Clamp to the last known-plausible reading
        # instead of trusting an outlier this far outside physical possibility.
        MAX_PLAUSIBLE_TILT_DEG = 30.0
        alpha_deg, beta_deg = np.degrees(alpha), np.degrees(beta)
        # Logged by pd_balance for diagnosing tracking glitches. (A per-frame tilt-jump
        # rejection filter was tried 2026-09-26 and reverted: it rejected real readings,
        # the servo then corrected against a stale angle and swung the plate side to
        # side -- 0% in circle.)
        self.last_inner_corners = None if self.pipeline.measurements.detector.corners is None \
            else np.array(self.pipeline.measurements.detector.corners, dtype=float).copy()
        if abs(alpha_deg) > MAX_PLAUSIBLE_TILT_DEG or abs(beta_deg) > MAX_PLAUSIBLE_TILT_DEG:
            self._bad_tilt_streak += 1
            print(f"[HardwarePlateEnv] rejecting physically implausible tilt reading "
                  f"(alpha={alpha_deg:+.1f} beta={beta_deg:+.1f}) -- likely lost reference-corner "
                  f"tracking, using last known-plausible tilt instead")
            # Clamping the READING doesn't fix the cause: the general Detector's
            # corner tracking (self.pipeline.measurements.detector, used every
            # frame for alpha/beta -- separate from the one-shot priming detector)
            # searches near ITS OWN last remembered corner positions, so once a
            # shake throws it onto the wrong blob it keeps re-finding that same
            # wrong blob forever and never recovers on its own. After a couple of
            # consecutive bad readings, force it to forget and re-search from the
            # default positions instead of the bad memory.
            if self._bad_tilt_streak >= 2:
                print("[HardwarePlateEnv] repeated bad tilt readings -- resetting corner "
                      "tracker to force re-acquisition from default search positions")
                self.pipeline.measurements.detector.corners = None
                self._bad_tilt_streak = 0
            alpha, beta = self._last_plausible_tilt
        else:
            self._bad_tilt_streak = 0
            self._last_plausible_tilt = (alpha, beta)
            self._meas_tilt = (float(np.degrees(alpha)), float(np.degrees(beta)))
        return xb, yb, alpha, beta, ball_found

    def _servo_tilt(self, action):
        """One closed-loop step toward the target tilt for `action`; positive
        action[0]/action[1] rolls the ball toward +x/+y (the convention every
        caller, incl. pd_balance.py, assumes). Axis mapping MEASURED on the rig
        2026-09-26 with this servo holding exact tilts (rl_hw/tilt_probe.py,
        phase3_logs/tilt_probe_2.log, tilt_probe_3x.log): a pure alpha tilt moves
        the ball in y (alpha -2.6deg -> dy +0.04..+0.05, dx ~0) and beta moves it in
        x (beta +2.9 -> +x) -- i.e. ball x <- +beta (motor3), ball y <- -alpha
        (motor1), NOT action[0]->motor1 as the old tick mapping assumed.
        Feedforward on the target change plus a proportional correction on the
        measured error (the camera frame read after the previous command)."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = (-action[1] * TILT_MAX_DEG + LEVEL_OFFSET_DEG[0],
                  action[0] * TILT_MAX_DEG + LEVEL_OFFSET_DEG[1])  # (alpha, beta)
        meas = self._meas_tilt if self._meas_tilt is not None else self._tilt_target
        self._target_hist.append(target)
        self._target_hist = self._target_hist[-12:]
        for i, dxl_id in enumerate(DXL_IDS):
            tpd = TICKS_PER_DEG[dxl_id]
            if TILT_OPEN_LOOP:
                band = abs(TILT_BAND_DEG * tpd)
                want = LEVEL_TICKS[dxl_id] + (target[i] - LEVEL_OFFSET_DEG[i]) * tpd
                pos = int(np.clip(want, LEVEL_TICKS[dxl_id] - band, LEVEL_TICKS[dxl_id] + band))
                self._cmd_ticks[dxl_id] = pos
                set_position(self.port_handler, self.packet_handler, dxl_id, pos)
                continue
            d = min(TILT_MEAS_DELAY_STEPS[dxl_id], len(self._target_hist) - 1)
            target_then = self._target_hist[-1 - d][i]
            ff = (target[i] - self._tilt_target[i]) * tpd          # feedforward on the target change
            corr = 0.0
            if self._pose_ok:                                      # all 4 plate markers seen this frame
                err = target_then - meas[i]
                corr = TILT_GAIN * err * tpd
                cur = self._cmd_ticks.get(dxl_id)
                st = self._stall.get(dxl_id)
                if (st is None or cur is None or abs(err) < st["err"] - STALL_GAIN_DEG
                        or abs(target_then - st.get("target", target_then)) > STALL_RETARGET_DEG):
                    st = self._stall[dxl_id] = {"ticks": cur, "err": abs(err), "warned": False,
                                                "target": target_then}
                st["ticks"] = (st["ticks"] or 0) + ff                 # feedforward moves are legitimate
                moved = (cur or 0) - st["ticks"]
                if abs(moved) > STALL_TICKS and np.sign(corr) == np.sign(moved):
                    if not st["warned"]:
                        print(f"[HardwarePlateEnv] motor {dxl_id}: {abs(moved):.0f} ticks of correction but the "
                              f"measured angle did not follow -- not pushing further (plate not following?)")
                        st["warned"] = True
                    corr = 0.0
            delta = float(np.clip(ff + corr, -MAX_TICKS_PER_STEP, MAX_TICKS_PER_STEP))
            lo, hi = TICK_BOUNDS[dxl_id]
            pos = int(np.clip(self._cmd_ticks.get(dxl_id, (lo + hi) // 2) + delta, lo, hi))
            self._cmd_ticks[dxl_id] = pos
            set_position(self.port_handler, self.packet_handler, dxl_id, pos)
        self._tilt_target = target

    def _hold_tilt(self, action, seconds):
        """Hold a target tilt for `seconds` with the closed loop running (a single
        _write_action only applies one correction step)."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            self._write_action(action)
            time.sleep(self.dt)
            self._read_state()

    def _servo_level(self, max_s=12.0):
        """Closed-loop leveling: servo to (0, 0) until both measured angles stay
        within LEVEL_TOL_DEG for several consecutive frames."""
        zero = np.zeros(2, dtype=np.float32)
        self._read_state()
        t0, ok = time.time(), 0
        while time.time() - t0 < max_s:
            self._write_action(zero)
            time.sleep(self.dt)
            self._read_state()
            a, b = self._meas_tilt if self._meas_tilt is not None else (99.0, 99.0)
            ok = ok + 1 if (abs(a - LEVEL_OFFSET_DEG[0]) < LEVEL_TOL_DEG
                            and abs(b - LEVEL_OFFSET_DEG[1]) < LEVEL_TOL_DEG) else 0
            if ok >= 8:
                break
        a, b = self._meas_tilt if self._meas_tilt is not None else (np.nan, np.nan)
        m1, m3 = self._cmd_ticks.get(1), self._cmd_ticks.get(3)
        if ok >= 8:
            try:
                with open(LAST_LEVEL_CACHE_PATH, "w") as f:
                    json.dump({"m1": m1, "m3": m3}, f)
            except OSError:
                pass
        else:
            print(f"[HardwarePlateEnv] servo leveling did not settle in {max_s:.0f}s "
                  f"(alpha={a:+.2f} beta={b:+.2f}, m1={m1} m3={m3}) -- proceeding anyway")

    def _level_open_loop(self):
        """plate straight to the fixed level position (LEVEL_TICKS), no camera involved"""
        for dxl_id, pos in LEVEL_TICKS.items():
            self._cmd_ticks[dxl_id] = pos
            set_position(self.port_handler, self.packet_handler, dxl_id, pos)
        self._tilt_target = LEVEL_OFFSET_DEG
        self._target_hist = [LEVEL_OFFSET_DEG] * 12

    def _write_action(self, action):
        if self.tilt_control:
            self._servo_tilt(action)
            return
        # action[1] (motor3/beta) is inverted relative to yb -- confirmed directly:
        # commanding a sustained action=(0,+1.0) moved yb from -0.111 to -0.123, the
        # OPPOSITE of what +1.0 should mean (push toward +yb). Every "correct toward
        # center" push on this axis in every script this session was silently
        # reinforcing whatever corner the ball was already heading into instead of
        # opposing it. Flipping the sign here, once, means every caller's action[1]
        # now means what it says (positive -> pushes yb positive) without having to
        # remember this quirk in each script.
        action = np.array([action[0], -action[1]], dtype=np.float32)
        # Map through _effective_calibration (refreshed every _level_plate() call to
        # center on whatever position leveling actually achieved), NOT the static
        # self.calibration -- confirmed by direct measurement that mapping through
        # the static center=3500 for motor3 was a severe, previously-undiagnosed bug:
        # _level_plate() converges motor3 to ~4046-4050 (as close to beta=0 as this
        # axis's real travel allows -- see its own docstring on the hysteresis), but
        # action=0.0 through the OLD static mapping commanded position 3500 -- 546
        # ticks below the just-achieved level point, i.e. a big NEGATIVE-beta shove.
        # Only action=+1.0 exactly happened to land near level (3500+1*(4050-3500)=
        # 4050). Every non-extreme action was silently fighting the controller's own
        # goal on this axis, which explains why the ball could never be recovered
        # from a -y corner even under a nominally "correct toward +y" action.
        for motor_id, a in zip(self._effective_calibration.keys(), action):
            min_pos, center_pos, max_pos = self._effective_calibration[motor_id]
            pos = map_action_to_position(a, min_pos, center_pos, max_pos)
            set_position(self.port_handler, self.packet_handler, motor_id, pos)

    def _recenter_motors(self):
        if getattr(self, "open_loop_level", os.environ.get("PLATE_OPEN_LOOP_LEVEL") == "1"):
            self._level_open_loop()
            return
        if self.tilt_control:
            self._write_action(np.zeros(2, dtype=np.float32))
            return
        # 2026-09-26: was self.calibration (static centre, motor3 ~1400 ticks off
        # level), which tilted the plate hard into a corner on every ball_lost.
        for motor_id, (_, center_pos, _) in self._effective_calibration.items():
            set_position(self.port_handler, self.packet_handler, motor_id, center_pos)

    def _level_plate(self, tolerance_deg=0.5, max_iters=20, max_step=150, settle_s=0.8, quick=False):
        if getattr(self, "open_loop_level", os.environ.get("PLATE_OPEN_LOOP_LEVEL") == "1"):
            self._level_open_loop()
            time.sleep(0.5)
            return
        """Closed-loop leveling against live camera feedback, run at the start of
        every episode. A single calibrated 'center' tick value turned out NOT to be
        trustworthy on its own: identical commanded ticks for motor3's axis were
        directly observed giving beta readings anywhere from -0.5deg to -4.7deg
        across separate tests, with no code or calibration change in between --
        real path/history-dependent hysteresis on that axis, not measurement noise
        or a calibration bug. Starting from the calibrated center as a reasonable
        initial guess and then correcting against a live reading makes each episode
        self-correcting instead of trusting that guess to still be right.

        quick=True skips the iterative search and just drives straight to the
        cached last-good position -- used right after _attempt_unstick() frees the
        ball. The full search actively explores several DIFFERENT tilt postures
        before settling (each held ~settle_s), and confirmed directly: that
        exploration trends toward the same extreme that traps the ball at a given
        corner, so running the full multi-second search right after freeing it
        can tilt it straight back before real step()-level PD control ever
        resumes. A quick, non-exploratory settle avoids re-baiting the trap it
        was just pulled out of; the next real reset() still does a full search."""
        if self.tilt_control:
            self._servo_level(max_s=3.0 if quick else 12.0)
            return
        m1_id, m3_id = sorted(self.calibration.keys())
        _, m1p, _ = self.calibration[m1_id]
        _, m3p, _ = self.calibration[m3_id]
        try:
            with open(LAST_LEVEL_CACHE_PATH) as f:
                cached = json.load(f)
            m1p, m3p = int(cached["m1"]), int(cached["m3"])
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            pass
        # These bounds were stale artifacts of a real bug, not the true safe range:
        # both servos had Position I Gain=0 (see POSITION_I_GAIN above), so under the
        # plate's gravity load they settled short of whatever was commanded, and the
        # hill-climb below misread that droop as "hit a real wall" and pegged here
        # permanently, every single call, without ever reaching alpha/beta=0. With
        # I gain now fixed, direct probing (present-position readback under real load,
        # not just a camera reading) confirmed clean, non-stalling tracking well past
        # these old numbers: motor1 up to at least 3084 (alpha still dropping linearly,
        # no gap growth), motor3 up to at least 4090 (beta still responding, gap <=4
        # ticks). Widened accordingly, with a margin kept back from the tested ceiling.
        # motor3/id3 got a Homing Offset of -1024 written to its EEPROM (see the
        # "_note" in motor_calibration.json) specifically because its real level
        # point kept converging right at the single-turn firmware ceiling (4095),
        # leaving ~5 ticks of headroom on one side regardless of tuning. Every
        # absolute tick number for motor3 below is shifted by that same -1024 to
        # stay physically consistent with the new numbering.
        #
        # m3_max initially got the same naive shift (4090-1024=3066), which was a
        # real bug: that just relabels the OLD ceiling, throwing away the entire
        # point of the offset. Direct probing post-offset (present-position
        # readback under load, not just a camera reading) confirmed clean,
        # non-stalling tracking with ~0 gap all the way from 3066 up to 4090 --
        # over 1000 ticks of real, previously-inaccessible range now open. 4090
        # keeps a small margin below the true firmware ceiling (4095, confirmed:
        # commanding past it clamps to 4095 and gap grows for real).
        # m1_max=3200 was itself a premature ceiling, same class of issue as m3's:
        # direct probing (present-position readback under load, generous settle
        # time) showed clean, non-stalling tracking all the way to the firmware
        # ceiling (4095). Unlike m3, alpha's RESPONSE genuinely plateaus around
        # -11deg starting near m1~3350-3600 (a real kinematic saturation of this
        # axis's tilt authority, confirmed by it holding flat across 800+ further
        # ticks of clean travel, not a numbering artifact) -- so there's no benefit
        # pushing all the way to 4095. But 3200 was stopping WELL BEFORE that
        # plateau even starts (alpha was still only -4.45deg there, not the -11deg
        # this axis can actually reach), so real, previously-inaccessible tilt
        # authority was being left unused. 3700 comfortably covers the plateau.
        m1_min, m1_max = 2374, 3700
        m3_min, m3_max = 1926, 4090
        m1p = int(np.clip(m1p, m1_min, m1_max))
        m3p = int(np.clip(m3p, m3_min, m3_max))

        set_position(self.port_handler, self.packet_handler, m1_id, m1p)
        set_position(self.port_handler, self.packet_handler, m3_id, m3p)
        time.sleep(settle_s * 2)

        if quick:
            self._effective_calibration[m1_id] = _symmetric_range(m1_min, m1p, m1_max)
            self._effective_calibration[m3_id] = _symmetric_range(m3_min, m3p, m3_max)
            return

        _, _, alpha, beta, _ = self._read_state()
        a, b = np.degrees(alpha), np.degrees(beta)
        # Fixed step size in the KNOWN direction, halved on overshoot -- not a secant/
        # slope estimate. The slope magnitude is unreliable under this axis's hysteresis
        # (a bad estimate on one iteration can send a proportional controller badly off
        # course, confirmed: alpha/beta ended up worse, not better, across repeated
        # trials). The DIRECTION has been 100% consistent all session regardless of
        # hysteresis: increasing motor1 always decreases alpha, increasing motor3
        # always increases beta. Hill-climb with that fixed direction and a shrinking
        # step is slower per-step but can't diverge the way a bad slope estimate can.
        step1 = step3 = max_step

        for _ in range(max_iters):
            if abs(a) < tolerance_deg and abs(b) < tolerance_deg:
                break
            d1 = 0 if abs(a) < tolerance_deg else (step1 if a > 0 else -step1)
            d3 = 0 if abs(b) < tolerance_deg else (-step3 if b > 0 else step3)
            new_m1p = int(np.clip(m1p + d1, m1_min, m1_max))
            new_m3p = int(np.clip(m3p + d3, m3_min, m3_max))
            set_position(self.port_handler, self.packet_handler, m1_id, new_m1p)
            set_position(self.port_handler, self.packet_handler, m3_id, new_m3p)
            time.sleep(settle_s)
            _, _, alpha_new, beta_new, _ = self._read_state()
            a_new, b_new = np.degrees(alpha_new), np.degrees(beta_new)
            # overshot (sign flipped) or made it worse -> the step was too big, halve it.
            # Otherwise grow it back (capped at max_step) -- without this, a step that
            # shrank early from one bad overshoot stays tiny for the rest of the run
            # and can't cover a large remaining error within the iteration budget
            # (observed directly: beta stuck creeping ~5-9 ticks/iter, needing ~300
            # more, after an early halving cascade).
            if d1 != 0:
                if np.sign(a_new) != np.sign(a) or abs(a_new) > abs(a):
                    step1 = max(step1 // 2, 5)
                else:
                    step1 = min(step1 + step1 // 2, max_step)
            if d3 != 0:
                if np.sign(b_new) != np.sign(b) or abs(b_new) > abs(b):
                    step3 = max(step3 // 2, 5)
                else:
                    step3 = min(step3 + step3 // 2, max_step)
            m1p, m3p, a, b = new_m1p, new_m3p, a_new, b_new

        if abs(a) >= tolerance_deg or abs(b) >= tolerance_deg:
            print(f"[HardwarePlateEnv] leveling did not fully converge "
                  f"(alpha={a:+.2f} beta={b:+.2f}) m1p={m1p}(bounds {m1_min}-{m1_max}) "
                  f"m3p={m3p}(bounds {m3_min}-{m3_max}) step1={step1} step3={step3} "
                  f"-- proceeding anyway")
        else:
            try:
                with open(LAST_LEVEL_CACHE_PATH, "w") as f:
                    json.dump({"m1": m1p, "m3": m3p}, f)
            except OSError as e:
                print(f"[HardwarePlateEnv] couldn't cache the converged level position: {e}")

        # Refresh the action-mapping reference to center on whatever position
        # leveling actually landed at (m1p, m3p), whether or not it fully converged
        # -- landing near a hard bound (as motor3 often does) still beats mapping
        # action=0 through the old static, badly-off-level calibration center. The
        # resulting range is intentionally ASYMMETRIC (not run through
        # _load_motor_calibration's symmetric-clamp helper): forcing symmetry here
        # would mean either crippling the large, real, safe range on one side to
        # match a tiny one on the other, or overshooting the hard safe bound on the
        # short side -- neither is what map_action_to_position's independent
        # negative-half/positive-half interpolation needs.
        self._effective_calibration[m1_id] = _symmetric_range(m1_min, m1p, m1_max)
        self._effective_calibration[m3_id] = _symmetric_range(m3_min, m3p, m3_max)

    def _gentle_recover(self, last_xy=None):
        """Bring a lost ball back into view without shaking (2026-09-26, flat paper).

        The ball is usually 'lost' because it rolled into a corner, where the
        detector can't separate it from the corner marker / frame shadow. From the
        leveled plate, tilt slowly AWAY from where it was last seen (or, if that's
        unknown, away from each corner in turn), one sustained tilt at a time, with
        small escalating magnitudes, returning to level and checking detection
        after each. Never a rapid reversal. Returns True once the ball is visible."""
        def _visible():
            for _ in range(3):
                _, _, _, _, found = self._read_state()
                if found:
                    return True
            return False

        self._hold_tilt(np.zeros(2, dtype=np.float32), 0.5)
        if _visible():
            return True
        if last_xy is not None and np.all(np.isfinite(last_xy)) and np.any(np.abs(last_xy) > 0.03):
            directions = [-np.sign(np.asarray(last_xy, dtype=np.float32))]
        else:
            directions = [np.array(d, dtype=np.float32) for d in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        for mag in (0.25, 0.35, 0.5):
            for d in directions:
                print(f"[HardwarePlateEnv] gentle recovery: tilt {tuple(np.round(d * mag, 2))} for 1.0s")
                self._hold_tilt((d * mag).astype(np.float32), 1.0)
                found_mid = _visible()
                self._hold_tilt(np.zeros(2, dtype=np.float32), 0.4)
                if found_mid or _visible():
                    print("[HardwarePlateEnv] gentle recovery: ball visible again")
                    return True
        return False

    def _attempt_unstick(self, stuck_pos=None):
        """Sweeps the plate hard through both axes -- used when the ball can't be
        found for a while during reset(), or (via stuck_pos) when step()'s wedge
        detector catches it sitting still despite a real commanded tilt. Most likely
        cause is it settled in one of the known detection blind spots (on a corner
        marker, or right at an edge); a full tilt sweep is a decent chance of rolling
        it away from there, since there's no person around to nudge it by hand.
        Finishes with a real camera-verified _level_plate() rather than just
        commanding action=(0,0) -- action 0 maps to the calibrated center ticks,
        which motor3's hysteresis has shown repeatedly is NOT reliably flat, so
        ending the sweep there could easily leave the plate tilted enough to re-trap
        the ball right after the sweep."""
        if not self.allow_unstick:
            print("[HardwarePlateEnv] unstick disabled (allow_unstick=False) -- not shaking")
            return False

        def _freed(prev_xy):
            xb, yb, _, _, found = self._read_state()
            if not found:
                return False
            if prev_xy is None:
                # called from reset()'s ball-not-found path -- no reference position
                # to measure displacement against, so simply becoming visible counts
                return True
            return np.hypot(xb - prev_xy[0], yb - prev_xy[1]) > STUCK_MOVEMENT_THRESHOLD * 2

        def _consolidate_escape(gentle):
            # "Freed" so far has only meant "moved enough to prove it's not still
            # wedged" -- often just a few mm, nowhere near clear of the trap. Spend
            # a bit more time here driving it decisively toward the goal before
            # handing back to normal control, so the ball actually arrives
            # somewhere PD only needs to fine-tune from.
            #
            # gentle=True (called right after a plain targeted push already
            # worked) skips the oscillating shake entirely and does a single
            # smooth, sustained push -- no reason to also rattle the plate
            # corner-to-corner when the gentle method alone was enough to move
            # it. Watching this live, exactly that redundant shake (violent
            # back-and-forth immediately after an already-successful gentle
            # push, then dropping the ball back near where it started) is what
            # looked wrong and unnecessary, not the underlying escape logic.
            xb, yb, _, _, found = self._read_state()
            if found:
                start_xy = (xb, yb)
                target = self.goal if np.any(self.goal) else np.zeros(2)
                dx, dy = float(target[0] - xb), float(target[1] - yb)
                norm = max(np.hypot(dx, dy), 1e-6)
                toward = np.array([dx, dy]) / norm
                if gentle:
                    print(f"[HardwarePlateEnv] carrying the ball toward the goal after escape "
                          f"(direction={tuple(toward.astype(np.float32))})...")
                    self._write_action(toward.astype(np.float32))
                    time.sleep(2.0)
                else:
                    # STEADY push toward goal was confirmed directly to NOT work
                    # for a genuinely wedged ball, even sustained for a full 6
                    # seconds at near-maximum tilt: it ended up within a few mm
                    # of where it started, every time. A V-notch/ledge the ball
                    # can nest into is exactly the situation where static
                    # friction/geometry resists a constant force but not an
                    # oscillating one (the same reason the shake escalation
                    # below frees it at all when a steady tilt sweep doesn't).
                    # So carry the same way here -- but ONLY when the escape
                    # itself needed that level of force (tilt sweep or violent
                    # shake succeeded, not a plain gentle push).
                    #
                    # A continuously-rotating "orbital" shake was also tried and
                    # made things WORSE, not just ineffective: confirmed
                    # directly, a fast-rotating tilt disrupts the camera's
                    # continuous reference-corner tracking badly enough to
                    # trigger repeated "lost tracking" resets in a near-endless
                    # loop (a single escape attempt ran for MINUTES instead of
                    # the intended ~6s). Stick with the plain, already-proven
                    # 4-corner pattern -- same dwell (0.25s) as the escalation
                    # tier's own violent shake, which has never caused a
                    # tracking problem -- and only check displacement once per
                    # full 4-corner cycle (~1s), not every tick.
                    corners = [(-1.0, -1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)]
                    print(f"[HardwarePlateEnv] shaking the ball toward the goal after escape "
                          f"(direction={tuple(toward.astype(np.float32))})...")
                    CARRY_TARGET_DIST = 0.03
                    CARRY_MAX_S = 6.0
                    t0 = time.time()
                    while time.time() - t0 < CARRY_MAX_S:
                        for a1, a2 in corners:
                            self._write_action(np.array([a1, a2], dtype=np.float32))
                            time.sleep(0.25)
                        xb2, yb2, _, _, found2 = self._read_state()
                        if found2 and np.hypot(xb2 - start_xy[0], yb2 - start_xy[1]) >= CARRY_TARGET_DIST:
                            break
                    # Whichever direction the shake happened to end up moving it,
                    # follow up with a real directional push toward the goal now
                    # that it's (hopefully) clear of the notch's immediate grip.
                    self._write_action(toward.astype(np.float32))
                    time.sleep(1.5)
                self._write_action(np.array([0.0, 0.0], dtype=np.float32))
                time.sleep(0.3)
            self._level_plate(quick=True)

        # Escalating attempts, cheapest/gentlest first -- stop as soon as one frees
        # the ball rather than always running the full aggressive sequence. That
        # matters beyond just wasted time: the rapid-reversal shake (last resort,
        # below) vibrates the whole rig hard enough that it's a plausible cause of
        # the U2D2 USB adapter repeatedly dropping mid-session, so it shouldn't run
        # when a gentler attempt already worked.
        if stuck_pos is not None:
            # A physical pocket (paper edge + corner marker, confirmed by direct
            # photo) didn't budge under a brief pulse at all -- try a SUSTAINED push
            # specifically away from the known stuck position toward center first.
            # Holding several seconds gives gravity a real chance to work the ball
            # out of a real lip, which a quick tap may not.
            dx, dy = -float(stuck_pos[0]), -float(stuck_pos[1])
            norm = max(np.hypot(dx, dy), 1e-6)
            action = np.clip(np.array([dx, dy]) / norm, -1.0, 1.0).astype(np.float32)
            print(f"[HardwarePlateEnv] targeted sustained push toward center from "
                  f"stuck position {tuple(stuck_pos)}, action={tuple(action)}...")
            self._write_action(action)
            time.sleep(3.0)
            self._write_action(np.array([0.0, 0.0], dtype=np.float32))
            time.sleep(0.3)
            if _freed(stuck_pos):
                print("[HardwarePlateEnv] targeted push freed the ball")
                _consolidate_escape(gentle=True)
                return True

        print("[HardwarePlateEnv] attempting to unstick the ball with a tilt sweep...")
        sweep = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        for _ in range(3):
            for a1, a2 in sweep:
                self._write_action(np.array([a1, a2], dtype=np.float32))
                time.sleep(0.4)
        if _freed(stuck_pos):
            print("[HardwarePlateEnv] tilt sweep freed the ball")
            _consolidate_escape(gentle=False)
            return True

        # Last resort -- fast direction reversals add momentum/vibration a slow tilt
        # can't, but also shake the whole rig hard (see USB-drop note above), so only
        # reached if the gentler attempts above didn't already work. Confirmed directly
        # that the original version of this (6 cycles, 0.15s dwell, 2-point) never
        # freed a genuinely wedged ball even once -- 0.15s may not even be enough for
        # the servo to reach the commanded extreme before reversing, so the real
        # swept range was smaller than intended. Go through all 4 corners (not just
        # 2 opposite points), give each position enough time to actually get there,
        # and repeat the whole round-trip several times, checking after each full
        # round instead of only once at the very end.
        print("[HardwarePlateEnv] sweep didn't free it, trying a violent shake...")
        violent = [(-1.0, -1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)]
        for round_i in range(5):
            for a1, a2 in violent:
                self._write_action(np.array([a1, a2], dtype=np.float32))
                time.sleep(0.25)
            if _freed(stuck_pos):
                print(f"[HardwarePlateEnv] violent shake freed the ball (round {round_i + 1})")
                _consolidate_escape(gentle=False)
                return True
        self._level_plate()
        return _freed(stuck_pos)

    def _build_obs(self, xb, yb, vx, vy, alpha, beta):
        return np.array(
            [xb, yb, vx, vy, alpha, beta, self.goal[0] - xb, self.goal[1] - yb],
            dtype=np.float32,
        )

    # ---- Gym API -----------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """No human-in-the-loop: the ball is physically contained by the plate's
        frame, so there's no canonical 'start position' to place it at -- each
        episode just continues from wherever the ball currently is. Waits
        (indefinitely, logging periodically) for the ball to be visible rather than
        blocking on a person confirming placement, so training can run unattended.
        If this waits a long time, the most likely cause is the ball stuck in one of
        the known detection blind spots (resting on a corner marker, or under a
        loose paper edge) -- a physical problem, not something this loop can fix."""
        super().reset(seed=seed)
        self._level_plate()

        if self.fixed_goal == "center":
            self.goal = np.array([0.0, 0.0], dtype=np.float32)
        elif self.fixed_goal is not None:
            self.goal = np.array(self.fixed_goal, dtype=np.float32)
        else:
            self.goal = np.array(
                [
                    self.np_random.uniform(-self._x_half + self.goal_margin, self._x_half - self.goal_margin),
                    self.np_random.uniform(-self._y_half + self.goal_margin, self._y_half - self.goal_margin),
                ],
                dtype=np.float32,
            )
        print(f"[HardwarePlateEnv] goal: ({self.goal[0]:.3f}, {self.goal[1]:.3f})")

        UNSTICK_EVERY_S = 6.0  # retry the recovery sweep this often while still stuck
        LOG_EVERY_S = 5.0
        xb, yb, alpha, beta, ball_found = self._read_state()
        waited = 0.0
        last_log = 0.0
        last_unstick = 0.0
        while not ball_found:
            time.sleep(self.dt)
            waited += self.dt
            if getattr(self, "on_wait_tick", None) is not None:
                self.on_wait_tick()          # e.g. the controller's ball-lost alert timer / UI
            if not getattr(self, "recover_tilts", True):
                # 2026-09-27 (user): while the ball is missing the plate stays LEVEL -- no
                # recovery tilts (a lost ball is under the plate / out, not in a corner)
                self._write_action(np.zeros(2, dtype=np.float32))    # closed-loop level step (camera)
                xb, yb, alpha, beta, ball_found = self._read_state()
                continue
            if waited - last_log >= LOG_EVERY_S:
                print(f"[HardwarePlateEnv] still waiting for the ball to become visible "
                      f"({waited:.0f}s) -- likely stuck on a corner marker or under the "
                      f"paper edge")
                last_log = waited
            if waited - last_unstick >= UNSTICK_EVERY_S:
                if self.allow_unstick:
                    self._attempt_unstick()
                else:
                    last_xy = self._prev_ball if np.any(self._prev_ball) else None
                    self._gentle_recover(last_xy)
                last_unstick = waited
            xb, yb, alpha, beta, ball_found = self._read_state()

        # Probe whether the ball actually responds to a real tilt before starting the
        # episode -- it can be found (visible, ball_found=True) yet already wedged
        # against the frame/a corner marker from wherever the last episode left it,
        # in which case the wait loop above never even triggers (it only fires on
        # not-found). Confirmed directly: a fresh reset() returned a ball that then
        # sat frozen for the next 25+ steps before step()'s own detector caught it --
        # catching it here avoids starting an episode that's already doomed.
        probe_dir = np.sign(self.goal - np.array([xb, yb])).astype(np.float32)
        probe_dir = np.where(probe_dir == 0, 1.0, probe_dir)
        for attempt in range(2 if self.allow_unstick else 0):
            # The probe only exists to trigger _attempt_unstick(); with that disabled
            # a 1s full-tilt probe just throws a free ball across the flat plate.
            self._write_action(probe_dir)
            time.sleep(1.0)
            self._write_action(np.array([0.0, 0.0], dtype=np.float32))
            xb2, yb2, alpha, beta, ball_found = self._read_state()
            if not ball_found or np.hypot(xb2 - xb, yb2 - yb) > STUCK_MOVEMENT_THRESHOLD * 2:
                xb, yb = xb2, yb2
                break
            print(f"[HardwarePlateEnv] ball didn't respond to the reset probe tilt "
                  f"(attempt {attempt + 1}/2) -- it's likely already wedged, unsticking")
            self._attempt_unstick(stuck_pos=(xb, yb))
            xb, yb, alpha, beta, ball_found = self._read_state()

        self._prev_ball = np.array([xb, yb], dtype=np.float32)
        self._prev_t = time.time()
        self._step_count = 0
        self._lost_count = 0
        self._last_commanded_action = np.zeros(2, dtype=np.float32)  # plate is level at reset
        self._pos_history = []
        self._action_history = []
        return self._build_obs(xb, yb, 0.0, 0.0, alpha, beta), {}

    def step(self, action):
        t0 = time.time()
        action = np.clip(
            action,
            self._last_commanded_action - self.max_action_delta,
            self._last_commanded_action + self.max_action_delta,
        )
        self._last_commanded_action = action
        self._write_action(action)

        elapsed = time.time() - t0
        if elapsed < self.dt:
            time.sleep(self.dt - elapsed)

        xb, yb, alpha, beta, ball_found = self._read_state()
        # Detection-glitch filter (2026-09-26): a ball resting in the target "jumped"
        # 11 mm in one frame; the phantom ~0.3 m/s velocity made the D term slam the
        # plate to full tilt and really threw the ball into a corner. A jump larger
        # than the ball's recent motion can explain is treated as a missed frame
        # (last position held, like any not-found frame); fast real motion still
        # passes because the allowance grows with the previous speed.
        if ball_found and self._prev_t is not None and np.all(np.isfinite(self._prev_ball)):
            jump = float(np.hypot(xb - self._prev_ball[0], yb - self._prev_ball[1]))
            allow = max(0.012, 3.0 * getattr(self, "_prev_speed", 0.0) * max(time.time() - self._prev_t, self.dt) + 0.006)
            if jump > allow:
                print(f"[HardwarePlateEnv] ignoring implausible ball jump of {jump * 1000:.0f} mm")
                ball_found = False
        self._step_count += 1
        self._lost_count = 0 if ball_found else self._lost_count + 1

        if self._lost_count >= BALL_LOST_GRACE_FRAMES:
            self._recenter_motors()
            obs = self._build_obs(self._prev_ball[0], self._prev_ball[1], 0.0, 0.0, alpha, beta)
            return obs, LOST_BALL_PENALTY, True, False, {"status": "ball_lost", "ball_found": False}

        if not ball_found:
            # Within the grace period: hold the last known position, keep the episode alive.
            xb, yb = self._prev_ball

        self._pos_history.append((xb, yb))
        self._action_history.append(action)
        self._pos_history = self._pos_history[-STUCK_CHECK_WINDOW:]
        self._action_history = self._action_history[-STUCK_CHECK_WINDOW:]
        if len(self._pos_history) == STUCK_CHECK_WINDOW:
            xs = [p[0] for p in self._pos_history]
            ys = [p[1] for p in self._pos_history]
            pos_range = max(max(xs) - min(xs), max(ys) - min(ys))
            mean_action_mag = float(np.mean([np.abs(a).mean() for a in self._action_history]))
            # With unstick disabled (flat paper, no ledge) "not moving under tilt"
            # just means the tilt is weak -- terminating would re-level and throw
            # away the episode. Only act on it when the unstick escape is enabled.
            if (self.allow_unstick and pos_range < STUCK_MOVEMENT_THRESHOLD
                    and mean_action_mag > STUCK_ACTION_THRESHOLD):
                print(f"[HardwarePlateEnv] ball appears wedged (moved {pos_range*1000:.1f}mm over "
                      f"{STUCK_CHECK_WINDOW} steps despite mean |action|={mean_action_mag:.2f}) "
                      f"-- attempting to unstick")
                freed = self._attempt_unstick(stuck_pos=(xb, yb))
                self._pos_history = []
                self._action_history = []
                if not freed:
                    # Full escalation (targeted push, sweep, fast shake) genuinely
                    # didn't move it -- this is a real physical pocket, not something
                    # more retrying fixes. End the episode cleanly instead of looping
                    # detect->fail->detect->fail with no signal that anything is wrong.
                    print("[HardwarePlateEnv] ball still wedged after full unstick "
                          "escalation -- ending episode")
                    xb, yb, alpha, beta, _ = self._read_state()
                    obs = self._build_obs(xb, yb, 0.0, 0.0, alpha, beta)
                    return obs, LOST_BALL_PENALTY, True, False, {"status": "ball_wedged"}

        # Real elapsed time between frames, not the nominal 1/control_hz: the loop
        # actually runs at ~21 Hz (not 55), so dividing by self.dt inflated every
        # velocity -- and the PD's D term -- ~2.4x (measured 2026-09-26 in the
        # logs; it drove a ~1.45 Hz limit cycle around the goal).
        now = time.time()
        dt_meas = min(max(now - self._prev_t, 1e-3), 0.25) if self._prev_t is not None else self.dt
        self._prev_t = now
        vx = (xb - self._prev_ball[0]) / dt_meas
        vy = (yb - self._prev_ball[1]) / dt_meas
        self._prev_speed = float(np.hypot(vx, vy))
        self._prev_ball = np.array([xb, yb], dtype=np.float32)

        dist = float(np.hypot(self.goal[0] - xb, self.goal[1] - yb))
        reward = -dist
        status = "running"
        if dist < self.goal_tolerance:
            # Balancing task: staying inside the circle is the objective, not a
            # one-shot target to reach and stop -- reward it every step it holds,
            # but don't end the episode over it (only ball-lost/timeout do that).
            reward += IN_CIRCLE_BONUS
            status = "in_circle"

        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        obs = self._build_obs(xb, yb, vx, vy, alpha, beta)
        return obs, reward, terminated, truncated, {"status": status, "ball_found": bool(ball_found)}

    def close(self):
        if self._closed:
            return
        self._closed = True
        # If SIGINT (or any exception) interrupted a serial transfer mid-flight, the
        # SDK's port.is_using lock (protocol2_packet_handler.py) never gets cleared,
        # and every subsequent command -- including this safety shutdown -- fails
        # with "Port is in use!". This is exactly the shutdown path where we need
        # those commands to go through regardless, so force the lock clear first.
        if getattr(self, "port_handler", None) is not None:
            self.port_handler.is_using = False
        try:
            if getattr(self, "port_handler", None) is not None and self._motors_torqued:
                # Deliberately NOT recentering to the static calibrated value here --
                # that value isn't reliably level (motor3's axis has real hysteresis,
                # see _level_plate), so driving to it right before shutdown would
                # silently undo whatever position (e.g. a just-leveled one) the plate
                # was actually holding. Just disable torque in place; the plate settles
                # gently under gravity when unpowered, it doesn't snap anywhere.
                for dxl_id in DXL_IDS:
                    self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, 0)
                self._motors_torqued = False
        except Exception as e:
            print(f"[HardwarePlateEnv] error while disabling torque: {e}")
        try:
            if getattr(self, "port_handler", None) is not None:
                self.port_handler.closePort()
        except Exception:
            pass
        try:
            if getattr(self, "_reader_thread", None) is not None:
                self._reader_stop = True
                self._reader_thread.join(timeout=1.0)
            if getattr(self, "cap", None) is not None:
                self.cap.release()
        except Exception:
            pass

    def __del__(self):
        self.close()
