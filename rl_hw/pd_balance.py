#!/usr/bin/env python3
"""
Hand-tuned PD controller to balance the ball inside the drawn goal circle. Used as
a fast, direct proof that the hardware/vision stack actually works end to end,
independent of (and much faster to validate than) full SAC training.

The plate is a double-integrator plant for the ball: tilt (the action) drives ball
ACCELERATION via gravity (a ~ g*sin(tilt) ~ tilt for small angles), not position or
velocity directly. So full state feedback -- proportional on position error, plus a
derivative term on velocity for damping -- is the physically-motivated controller
structure here, not just a heuristic.

Runs across multiple episodes automatically: if one ends early (ball_wedged,
ball_lost), that's a single bad attempt, not the run being over -- it resets and
keeps going until it either racks up solid in-circle time or the episode budget
runs out.

Usage:
    python3 pd_balance.py [total_steps] [max_episodes]
"""
import sys
import os
import time
import json
import signal
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from plate_env import HardwarePlateEnv  # noqa: E402
from goal_circle import plate_to_pixel  # noqa: E402
from goal_circle import (detect_goal_circle, detect_on_frame, save_debug, save_last_goal,  # noqa: E402
                         pixel_to_plate, inside_region)
from ui_server import UIServer, FRAME_W, FRAME_H  # noqa: E402
from line_path import detect_line, PathTracker, V_LINE  # noqa: E402
import shapes  # noqa: E402
import calibrate  # noqa: E402
from maze_practice import MazePractice  # noqa: E402
from plate_env import set_position as _set_position  # noqa: E402
from rl_policy import RigPolicyController, ODILRigController, ODILFrictionCompRig  # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rl_sim")))
from odil_track import TrackPolicy  # noqa: E402  (numpy only)
MAZE_REAL = os.environ.get("PD_MAZE_REAL") == "1"   # the real maze board (walls, real holes)
from elevator import Elevator  # noqa: E402
from hole import detect_holes_stable, save_holes, near_hole, detour, HOLE_MARGIN_M  # noqa: E402
from plate_env import LEVEL_OFFSET_DEG  # noqa: E402

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pd_snapshots")
SNAPSHOT_EVERY_S = float(os.environ.get("PD_SNAPSHOT_S", "2.0"))  # e.g. 30 for long unattended runs
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pd_logs")
# Success criterion (Phase 3): ball held continuously inside the goal circle for
# this long, measured in wall-clock seconds, not steps -- the real step rate is
# well below the nominal 55Hz, so a step count would overstate the hold.
HOLD_TARGET_S = 10.0

# Single fixed (KP, KD) was a real tuning mistake, not just imprecise: lowering
# KP to stop overshoot AT the goal also weakens correction EVERYWHERE else,
# including wherever the plate's residual leveling bias is strongest (confirmed
# directly: the ball recovered to dist=0.128 from a corner, then drifted straight
# back to that exact same corner over the next ~30 steps under uninterrupted
# normal control -- a persistent bias steadily winning against a now-too-gentle
# correction, not overshoot). Gain-schedule instead: aggressive far from the
# goal (needs to win against real corner bias / reach across the plate), gentle
# only once close enough that overshoot-through-the-circle is the actual risk.
KP_FAR, KD_FAR = 5.0, 2.0
KP_NEAR, KD_NEAR = 2.5, 4.0
GAIN_BLEND_DIST = 0.09  # start easing toward the gentle gains at this distance
GAIN_BLEND_WIDTH = 0.05  # over this much additional distance
# 2026-09-26: slow integral on position error. The rig's true level point, fit
# from each run's own data (ball acceleration vs measured tilt, r~0.98), is
# stable within a run but moves 0.3-0.5deg between runs (beta0 3.00 -> 2.55 ->
# 2.98, alpha0 -0.82 -> -1.13 -> -1.41), so no constant LEVEL_OFFSET_DEG stays
# right; with KP_NEAR the residual bias parked the ball 25-50mm off-centre in a
# 45.5mm disc. Integrate only near the goal (anti-windup), real wall-clock dt,
# clamped to +-0.3 action (+-1.5deg), carried across episodes (the bias is).
KI = 1.0
I_ZONE = 0.10
I_MAX = 0.3  # 0.6 tried 2026-09-26: stick-slip (ball stuck ~1.5-2deg breakaway, then 50-80mm overshoot), in-circle 88.8% -> 76.8%; reverted
# 2026-09-26 ramp-kick & release (design panel + judge on the logs): on bare paper
# a still ball sinks into a slight dimple -> static breakaway ~1.5-2.4deg >> rolling
# ~0.3-0.6deg. With a plain integral the stored push was never released after
# breakaway (action unchanged 10 frames later), so the ball rolled through the goal
# and re-stuck 60-80mm away. Now: the integral learns only while the ball rolls;
# a separate kick ramps toward the goal while it is stuck and is dumped the moment
# the ball moves. Stuck is detected from POSITION (logged speed of a still ball is
# 7-16 mm/s, useless as a stillness test).
STUCK_WIN = 15          # frames (~0.66 s) for the stationarity test
STUCK_PTP = 0.0015      # m, per-axis position range => stationary (rest noise ptp <= 1.4 mm)
BREAK_DISP = 0.002      # m from anchor => broke free -> dump kick
R_DONE = 0.008          # m, stationary this close = success, never kick (0.005: a ball resting at 4.5 mm crossed it on noise and got kicked out 33 mm)
KICK_RATE = 0.6         # action/s (3 deg/s) ramp along the unit vector to the goal
KICK_MAX = 0.6          # action vector cap (3 deg); one stick held 87 s at |a|~0.47
KICK_START_FRAC = 0.6   # restart the ramp at 60% of the last successful breakaway kick
KICK_HOLD_S = 0.5       # give up if still stuck this long at the cap
KICK_LOCKOUT_S = 1.0    # after a release (2x after a give-up)
# The goal is detected from the red circle on the sheet at every startup
# (goal_circle.py), so a new sheet needs no edits. These are only a reference: the
# centre disc used on 2026-09-26. Set PD_FIXED_GOAL=1 to use them instead.
# Live goal tracking (swap the sheet mid-run): every GOAL_CHECK_S the red circle
# is re-detected on the frame the current plate pose came from. A new goal is
# adopted only after GOAL_CONFIRM consecutive checks agree (within GOAL_AGREE_M) on
# a circle clearly different from the current one (moved > GOAL_MOVE_M or radius
# changed > GOAL_RADIUS_CHANGE_M), so hands in view or a half-swapped sheet can't
# create a false goal. No visible circle -> keep the current goal.
# (A target-sized capture zone with KD 6 was tried 2026-09-26 on a 9 mm dot: no
# improvement, 23% vs 30% in the dot -- reverted.)
# Speed-reference approach (user's step 2): PD rewritten as kd*(v_ref - v) with
# v_ref = (kp/kd)*e, i.e. identical to kp*e - kd*v -- except that |v_ref| is capped,
# so a ball far from the goal is brought in at a limited speed instead of being
# flung at it (far gains alone ask for 2.5 m/s per m of error, 0.375 m/s at 15 cm).
V_REF_MAX = 0.08  # m/s
# Braking curve (research experiment #6): the allowed approach speed also shrinks
# with the distance left, v <= sqrt(2*A_BRAKE*(dist - r/2)), so the ball can stop
# at the target. With only the 8 cm/s cap a ball arriving at a 9 mm dot coasted
# ~10 cm on paper (rolling decel ~0.03 m/s^2) and overshot it by 42-52 mm.
A_BRAKE = 0.05    # m/s^2 -- gentle tilt-braking on top of rolling friction
# Timed pulse + planned brake (research workflow 2026-09-26; Yang & Tomizuka 1988
# adaptive pulse-width control, van de Wouw & Leine 2012 impulsive control under
# uncertain friction). Replaces the ramp-kick: the ramp pushed until motion was SEEN,
# 0.15 s late, so the freed ball carried too much energy and rolled past a small dot
# (82% of exits followed a kick). Now: a pulse just above breakaway for N frames,
# then an open-loop brake for N frames, then coast; afterwards the advance p toward
# the goal is measured and N adapted (N <- N*sqrt(d0/p)), per 3 cm plate cell. A
# pulse that doesn't free the ball makes the next one stronger.
PULSE_A0, PULSE_A_MAX = 0.40, 0.90   # pulse amplitude (action, 1.0 = 5 deg); 0.6 often failed to free the ball on the wall side of a near-edge dot
PULSE_N0, PULSE_N_MIN, PULSE_N_MAX = 3, 2, 8   # pulse width in frames
BRAKE_A = 0.12                       # ~0.6 deg: above rolling friction, below breakaway
COAST_MAX_S = 1.5
PULSE_LOCKOUT_S = 0.5
PULSE_CELL_M = 0.03
# learned per-cell pulse settings persist across runs (smooth from the first second)
PULSE_TABLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pulse_table.json"))
# Plate map (rl_hw/map_plate.py): per-area local level and breakaway tilt measured on
# the rig. The paper is not uniform: the local "level" varies by ~1 deg across the
# plate and breakaway between ~1.2 and 2.3 deg, which a single global offset and a
# fixed first-pulse strength can't serve. Off by default if the file is missing.
PLATE_MAP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plate_map.json"))
USE_PLATE_MAP = os.environ.get("PD_PLATE_MAP", "0") == "1"  # off: made the near-edge dot worse (2026-09-26)


class PlateMap:
    """Inverse-distance-weighted interpolation of the measured map samples."""

    def __init__(self, path):
        with open(path) as f:
            pts = json.load(f)["points"]
        self.pos = np.array([p["pos"] for p in pts])
        self.level = np.array([[p.get("level_x_deg") or 0.0, p.get("level_y_deg") or 0.0] for p in pts])
        self.static = np.array([np.nanmean([v for v in (p.get("static_x_deg"), p.get("static_y_deg")) if v is not None]
                                           or [np.nan]) for p in pts])

    def _w(self, xy):
        d = np.hypot(*(self.pos - np.asarray(xy)).T)
        return 1.0 / np.maximum(d, 0.01) ** 2

    def level_deg(self, xy):
        w = self._w(xy)
        return (w[:, None] * self.level).sum(0) / w.sum()

    def static_deg(self, xy):
        ok = np.isfinite(self.static)
        w = self._w(xy)[ok]
        return float((w * self.static[ok]).sum() / w.sum()) if ok.any() else None
GOAL_CHECK_S = 0.5
CLICK_R = 0.012   # target radius for a clicked point (m)
# Planned moves (2026-09-26, user: "one or two smooth motions, then balancing"): the
# hop-hop approach came from the ball slowing down, sinking into the paper and
# sticking, then being pushed free again. Instead, for targets more than MOVE_MIN_M
# away, plan ONE move -- accelerate, cruise, brake to a stop at the target (trapezoid
# speed profile) -- break the ball free once, then feed the planned acceleration +
# rolling friction forward so it keeps rolling the whole way, with light tracking
# feedback. If it still stops short, a second move is planned. The stiction pulses
# remain only for the last ~2 cm.
MOVE_MIN_M = 0.02
V_MOVE, A_MOVE = 0.10, 0.10     # m/s, m/s^2
ACC_PER_ACTION = 0.45           # m/s^2 per unit action (0.09 m/s^2/deg x 5 deg)
A_ROLL_FF = 0.03                # m/s^2 rolling friction to feed forward while moving
KP_MOVE, KD_MOVE = 6.0, 2.5     # tracking feedback on the planned path (action per m, per m/s)
MOVE_BOOST_RATE, MOVE_BOOST_MAX = 1.0, 0.7   # breakaway ramp (action/s, cap)
MOVE_ABORT_M = 0.035            # tracking error that ends the move (replanned if needed)
# Learned local level (2026-09-26): after a move got the ball to the target it often
# rolled back down a gentle local slope (~0.5 deg; the paper isn't flat), because the
# slow integral hadn't learned that spot yet. When the ball has rested in a target
# for BIAS_LEARN_S, the integral IS the local level there: store it per plate cell
# (persisted) and start from it whenever a new target lands in that cell.
BIAS_TABLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bias_table.json"))
BIAS_CELL_M = 0.03
BIAS_LEARN_S = 1.0


def trapezoid(D, vmax, amax):
    """Rest-to-rest speed profile over distance D: returns (T, f(t) -> (s, v, a))."""
    t_acc = vmax / amax
    if D < vmax * t_acc:                     # triangular
        t_acc = np.sqrt(D / amax)
        vmax = amax * t_acc
        t_flat = 0.0
    else:
        t_flat = (D - vmax * t_acc) / vmax
    T = 2 * t_acc + t_flat

    def f(t):
        t = min(max(t, 0.0), T)
        if t < t_acc:
            return 0.5 * amax * t * t, amax * t, amax
        if t < t_acc + t_flat:
            return 0.5 * amax * t_acc ** 2 + vmax * (t - t_acc), vmax, 0.0
        td = T - t
        return D - 0.5 * amax * td * td, amax * td, -amax
    return T, f
LINE_TOL = 0.008  # line mode: on-line tolerance (m)
# Hybrid (2026-09-27): the learned policy is great at the approach (one smooth ~1-2 s glide,
# no hops) but makes small jerky corrections near the target and sometimes settles just
# outside it. It drives until the ball is within HANDOVER of the target, then the classic
# near-field control (damping, learned local slope, stiction pulses) settles and holds.
# Hysteresis: back to the policy only if the ball is pushed out beyond HANDBACK.
HANDOVER_R_SCALE, HANDOVER_MIN = 2.5, 0.025
HANDBACK_SCALE = 2.5
KP_LINE, KD_LINE = 8.0, 2.5            # tracking feedback on the moving route reference
LINE_BOOST_RATE, LINE_BOOST_MAX = 1.0, 0.6  # breakaway push along the route when the ball won't follow
LINE_JOIN_M = 0.015                    # route following starts once the ball is this close to it
GOAL_CONFIRM = 3
GOAL_AGREE_M = 0.004
GOAL_MOVE_M = 0.008
GOAL_RADIUS_CHANGE_M = 0.005
GOAL = (-0.0078, 0.0060)
# holes in the paper (hole.py): targets route round them; a ball that falls in is
# reloaded by the elevator (motor 2) and the task continues
VIA_R = 0.012                          # target radius of a via point round a hole
VIA_REACH = 0.02                       # this close to a via point -> head for the next one
RELOAD_TIMEOUT_S = 60.0                # ball lost: elevator runs until the ball is seen again, at most 60 s (user)
RELOAD_UNITS = int(os.environ.get("PD_RELOAD_UNITS", "328"))      # elevator speed for a reload (~75 rpm)
RELOAD_SEEN_FRAMES = 5                 # ball visible on the paper this many frames in a row = reloaded
GOAL_TOLERANCE = 0.0455
# Direct feedback while watching this live: correcting X and Y together lets the
# combined vector point diagonally, straight at a corner, instead of going through
# the middle of an edge on the way back. Once either axis is out near the edge,
# stop blending -- fully prioritize bringing THAT axis back first before resuming
# normal 2D control, so the path runs through edge-middles, not corners.
EDGE_X, EDGE_Y = 0.10, 0.085
EDGE_PUSH_MIN, EDGE_PUSH_GAIN, EDGE_PUSH_MAX = 0.2, 20.0, 0.4  # action: 0.2 at the line, +0.1 per 5 mm, max 0.4 (2 deg)
# One specific point (-0.15, +0.125) keeps recurring as a trap: a real V-notch
# where the frame's two inner walls meet at a right angle -- UPDATE: a closer
# zoomed photo showed this is actually a raised wooden bezel LEDGE where the
# paper meets the frame wall, not a corner notch; the ball perches slightly
# above the play surface there. Confirmed directly, in isolation: the full
# escalation (targeted push + tilt sweep + violent shake, ~15s combined)
# displaced it only ~6mm from a cold stop at this spot -- a real, severe
# structural defect that no shake pattern tried so far (linear, biased-linear,
# orbital) reliably clears. That fix needs a physical correction (built up
# paper/shim height to remove the step) -- it's not something more code can
# solve. What code CAN do: don't let one truly-stuck episode burn the whole
# step budget on futile escapes when a fresh reset() elsewhere on the plate has
# a real chance (confirmed: several fresh starts this session reached
# dist<0.1). See max_episode_steps below.


_INSTANCE_LOCK = None


def _single_instance_or_exit():
    """Only one controller may drive the rig. 2026-09-26: a second instance opened
    the motor port while the first was running; the first crashed ('multiple access
    on port') and its shutdown switched motor torque OFF, so the survivor kept
    sending positions to unpowered motors -- 'running' in the UI, nothing moving."""
    global _INSTANCE_LOCK
    import fcntl
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".controller.lock"))
    _INSTANCE_LOCK = open(path, "a+")
    try:
        fcntl.flock(_INSTANCE_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _INSTANCE_LOCK.seek(0)
        other = _INSTANCE_LOCK.read().strip() or "?"
        sys.exit(f"another controller is already running (PID {other}) -- stop it first (Ctrl+C in its "
                 f"terminal, or: pkill -INT -f pd_balance.py). Not touching the motors.")
    _INSTANCE_LOCK.seek(0)
    _INSTANCE_LOCK.truncate()
    _INSTANCE_LOCK.write(str(os.getpid()))
    _INSTANCE_LOCK.flush()


LOST_ALERT_S = float(os.environ.get("PD_LOST_ALERT_S", "20"))


def _alert(msg):
    """Tell the user: a line in the log ("ALERT: ...", watched by the Claude session for
    phone pushes) + a macOS desktop notification with sound."""
    print(f"ALERT: {msg}", flush=True)
    try:
        import subprocess
        subprocess.Popen(["osascript", "-e", f'display notification "{msg}" with title "CyberRunner" sound name "Glass"'])
    except OSError:
        pass


def _sigterm(*_):
    raise KeyboardInterrupt   # a plain `kill` (or a background run, where Ctrl-C is ignored) shuts down cleanly


def back_waypoint(route, ball, arc=0.04, tol=0.003, near=0.025):
    """Next point for returning to the start of a walled maze: the route point furthest
    back (up to `arc` along the route) that the ball can reach in a straight line -- the
    chord may leave the route by at most `tol`. None if the ball is not near the route or
    already close to the start (then the start itself is the goal)."""
    d = np.hypot(*(route - ball).T)
    k = int(np.argmin(d))
    if d[k] > near:
        return None
    S = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))])
    if S[k] < arc:
        return None
    best = None
    for j in range(k - 1, -1, -1):
        if S[k] - S[j] > arc:
            break
        c = route[j] - ball
        n2 = max(float(c @ c), 1e-12)
        q = route[j:k + 1] - ball
        t = np.clip((q @ c) / n2, 0.0, 1.0)
        if np.hypot(*(q - t[:, None] * c).T).max() > tol:
            break
        best = route[j].copy()
    return best


def main():
    _single_instance_or_exit()
    signal.signal(signal.SIGTERM, _sigterm)
    total_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    # optional: hold target in s (0 = never stop early), episode length in steps
    hold_target = float(sys.argv[3]) if len(sys.argv) > 3 else HOLD_TARGET_S
    episode_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, time.strftime("pd_%Y%m%d_%H%M%S.csv"))
    log = open(log_path, "w")
    log.write("t,episode,step,xb,yb,vx,vy,alpha,beta,dist,in_circle,ball_found,a0,a1,status,i0,i1,k0,k1,kicking,"
              "c0r,c0c,c1r,c1c,c2r,c2c,c3r,c3c,goal_x,goal_y,goal_r\n")
    print(f"logging to {log_path}")
    # max_episode_steps default (600) let one truly-stuck episode (the ledge
    # defect, see note above) burn its entire step budget on repeated futile
    # escape attempts. 150 steps (~2.7s of real control time at 55Hz, though
    # wedge-detection/escalation stretches wall-clock well beyond that) is
    # still generous for genuine balancing but ends a hopeless episode fast
    # enough to let the remaining total_steps budget go toward fresh starts,
    # several of which land clear of the defect entirely.
    # 2026-09-26: glass removed, ledge gone -- the 150-step cap above (and the
    # unstick escalation) only existed for the ledge. 150 steps is ~3s of real
    # time, too short to ever contain a 10s hold: truncation -> reset() ->
    # re-level + probe tilt throws the ball out of the circle. No shaking at all.
    env = HardwarePlateEnv(
        fixed_goal=GOAL, goal_tolerance=GOAL_TOLERANCE, max_action_delta=0.5,
        max_episode_steps=episode_steps, allow_unstick=False,
    )
    live_goal = True
    goal_provisional = False
    goal_polygon = goal_contour_px = goal_px = goal_r_px = None
    ui = UIServer(int(os.environ.get("PD_UI_PORT", "8000"))) if os.environ.get("PD_UI") == "1" else None
    if ui is not None:
        env.frame_callback = ui.publish_frame
    running = ui is None           # with the UI, wait for Start (or a click)
    elevator = Elevator(env.port_handler, env.packet_handler) if ui is not None else None
    mode = "sheet"
    follower, line_off, line_msg = None, None, ""
    line_active, line_boost, line_ref = False, 0.0, None
    path_pts, path_px = [], []          # "Draw path" mode: clicked waypoints (plate m / image px)
    # learned controller (rl_sim/train_plate_ppo.py): PD_POLICY=<policy.npz> or UI toggle
    policy_path = os.environ.get("PD_POLICY") or os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "rl_sim", "runs", "plate_goal_v3", "policy.npz"))
    rl_ctrl = RigPolicyController(policy_path, LEVEL_OFFSET_DEG) if os.path.exists(policy_path) else None
    use_policy = bool(os.environ.get("PD_POLICY")) and rl_ctrl is not None
    policy_driving = True     # within the hybrid: True = policy approach, False = classic near field
    # ODIL controller (rl_sim/odil_plate.py): PD_ODIL=<odil_policy.npz> or UI toggle; PD_HYBRID=0 disables
    # the classic near-field settle for both learned controllers (pure policy all the way in)
    odil_path = os.environ.get("PD_ODIL") or os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "rl_sim", "runs", "odil_best", "odil_policy.npz"))
    # PD_ODIL_FC=0 disables the stiction compensation (pure ODIL policy)
    odil_cls = ODILRigController if os.environ.get("PD_ODIL_FC") == "0" else ODILFrictionCompRig
    odil_ctrl = odil_cls(odil_path) if os.path.exists(odil_path) else None
    # ODIL path-tracking policy (rl_sim/odil_track.py + finetune_track.py): used for line / path /
    # drawing modes when the ODIL controller is selected
    track_path = os.environ.get("PD_ODIL_TRACK") or os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "rl_sim", "runs", "odil_track_best", "odil_track_policy.npz"))
    track_ctrl = TrackPolicy(track_path) if os.path.exists(track_path) else None
    if track_ctrl is not None:
        print(f"ODIL tracking policy available: {track_path}")
    use_odil = bool(os.environ.get("PD_ODIL")) and odil_ctrl is not None
    hybrid = os.environ.get("PD_HYBRID", "1") == "1"
    if odil_ctrl is not None:
        print(f"ODIL policy available: {odil_path}" + (" (ACTIVE)" if use_odil else ""))
    if rl_ctrl is not None:
        print(f"learned policy available: {policy_path}" + (" (ACTIVE)" if use_policy else ""))
    move, move_request, n_moves = None, False, 0
    try:
        with open(BIAS_TABLE_PATH) as f:
            bias_table = {tuple(int(v) for v in k.split(",")): np.array(val) for k, val in json.load(f).items()}
    except (OSError, ValueError):
        bias_table = {}
    rest_since = None

    def bias_cell(xy):
        return (int(round(xy[0] / BIAS_CELL_M)), int(round(xy[1] / BIAS_CELL_M)))

    def bias_for(xy):
        """stored local level for the cell of xy (or the mean of its stored neighbours)"""
        c = bias_cell(xy)
        if c in bias_table:
            return bias_table[c].copy()
        near = [bias_table[(c[0] + i, c[1] + j)] for i in (-1, 0, 1) for j in (-1, 0, 1)
                if (c[0] + i, c[1] + j) in bias_table]
        return np.mean(near, axis=0) if near else None
    if os.environ.get("PD_GOAL"):
        # virtual goal for testing: PD_GOAL="x,y,r" in metres; disables live re-detection
        gx_, gy_, gr_ = (float(v) for v in os.environ["PD_GOAL"].split(","))
        goal, goal_tol, live_goal = (gx_, gy_), gr_, False
        env._level_plate()
        print(f"using virtual goal ({goal[0]:+.4f}, {goal[1]:+.4f}) r={goal_tol:.4f} (PD_GOAL, live tracking off)")
    elif os.environ.get("PD_FIXED_GOAL") == "1":
        goal, goal_tol = GOAL, GOAL_TOLERANCE
        print(f"using fixed goal ({goal[0]:+.4f}, {goal[1]:+.4f}) r={goal_tol:.4f} (PD_FIXED_GOAL=1)")
    else:
        # Level the plate first (closed loop): the cached startup posture can leave it
        # several degrees tilted, which rolls the ball away and gives a moving pose.
        env._level_plate()
        try:
            res = detect_goal_circle(env)
        except RuntimeError as e:
            if ui is None:
                env.close()
                log.close()
                sys.exit(f"could not find the goal circle on the sheet: {e}")
            print(f"no red region found ({e}) -- UI starts in click-to-target mode")
            res = {"center": (0.0, 0.0), "radius": 0.02, "px": None, "frame": None, "n_frames": 0,
                   "spread_m": 0.0, "source": "none", "polygon": None, "contour_px": None}
            mode, live_goal = "click", False
        goal, goal_tol = res["center"], res["radius"]
        goal_polygon, goal_contour_px = res.get("polygon"), res.get("contour_px")
        if res.get("px") is not None:
            goal_px, goal_r_px = res["px"][:2], res["px"][2]
        save_debug(res, os.path.join(LOG_DIR, os.path.basename(log_path).replace(".csv", "_goal.jpg")))
        print(f"goal circle detected ({res['source']}): centre ({goal[0]:+.4f}, {goal[1]:+.4f}) m, radius "
              f"{goal_tol * 1000:.1f} mm (median of {res['n_frames']} frames, spread {res['spread_m'] * 1000:.1f} mm)")
        goal_provisional = res["source"] == "occluded"
    env.fixed_goal = goal
    env.goal_tolerance = goal_tol
    holes = [] if os.environ.get("PD_HOLES") == "0" else detect_holes_stable(env)
    save_holes(holes)
    auto_reload = elevator is not None
    drop_left, drop_times, drop_prev, restore_after_reset = 0, [], None, False
    vias, final_target = [], (goal, goal_tol, goal_polygon, goal_contour_px, goal_px, goal_r_px, "start")
    for h in holes:
        print(f"hole at ({h['center'][0]:+.4f}, {h['center'][1]:+.4f}) m, radius {h['radius'] * 1000:.1f} mm "
              f"-> keep-out {(h['radius'] + HOLE_MARGIN_M) * 1000:.0f} mm")

    def hole_status(extra=""):
        base = (f"{len(holes)} hole(s) found" if holes else "no hole found") + (
            f" | drops {len(drop_times)}, last reload {drop_times[-1]:.1f} s" if drop_times else "")
        return base + (f" | {extra}" if extra else "")

    def publish_holes(extra=""):
        if ui is None:
            return
        ui.set_overlay(holes_px=[(h["px"][0], h["px"][1], h["px"][2],
                                  h["px"][2] * (h["radius"] + HOLE_MARGIN_M) / max(h["radius"], 1e-4)) for h in holes])
        ui.set_state(hole_msg=hole_status(extra), auto_reload=auto_reload)
    publish_holes()
    # Edge guard only where the ball is past the wall threshold AND further out than
    # the goal disc on that side -- a disc drawn near an edge stays reachable.
    def edge_limits(g, r, quiet=False):
        # Guard only well past the goal circle: with it right at the circle's edge
        # (e.g. at 10.4 cm for a 9 mm dot at x=9.5 cm) every small overshoot got a
        # full-tilt shove back -> overshoot the other way -> repeating pattern.
        # 3 cm beyond the circle, capped 1 cm short of the plate's edge.
        ex = max(EDGE_X, min(abs(g[0]) + r + 0.03, env._x_half - 0.01))
        ey = max(EDGE_Y, min(abs(g[1]) + r + 0.03, env._y_half - 0.01))
        if (ex > EDGE_X or ey > EDGE_Y) and not quiet:
            print(f"goal is near an edge: edge guard moved out to |x|>{ex:.3f}, |y|>{ey:.3f}")
        return ex, ey

    edge_x, edge_y = edge_limits(goal, goal_tol)

    def adopt_goal(center, radius, polygon=None, contour_px=None, px=None, r_px=None, why=""):
        """Switch the controller to a new target (sheet re-detection or a UI click)."""
        nonlocal goal, goal_tol, goal_polygon, goal_contour_px, goal_px, goal_r_px, edge_x, edge_y
        nonlocal hold_start, phase, kick, lockout_until, goal_provisional, n_goal_changes, move, move_request
        nonlocal integ, rest_since, policy_driving
        goal = (float(center[0]), float(center[1]))
        goal_tol = float(radius)
        goal_polygon, goal_contour_px, goal_px, goal_r_px = polygon, contour_px, px, r_px
        env.fixed_goal = goal
        env.goal = np.array(goal, dtype=np.float32)
        env.goal_tolerance = goal_tol
        edge_x, edge_y = edge_limits(goal, goal_tol)
        # new target: restart the hold and any stiction pulse; keep the learned level
        # bias (it belongs to the plate, not the target)
        hold_start = None
        phase, lockout_until = "idle", 0.0
        kick = np.zeros(2)
        pos_buf.clear()
        goal_cands.clear()
        goal_provisional = False
        move, move_request = None, True
        policy_driving = True
        b = bias_for(goal)
        if b is not None:
            integ = np.clip(b, -I_MAX, I_MAX)   # start from what this spot is known to need
        rest_since = None
        n_goal_changes += 1
        print(f"\n>>> NEW GOAL ({why}): centre ({goal[0]:+.4f}, {goal[1]:+.4f}) m, radius {goal_tol * 1000:.1f} mm <<<\n")

    def set_target(center, radius, polygon=None, contour_px=None, px=None, r_px=None, why="", from_xy=None):
        """adopt_goal, but routed round the holes: via points first, then the target"""
        nonlocal vias, final_target
        if px is None and ui is not None:
            # scripted / programmatic targets: find where to draw the target circle
            rc = plate_to_pixel(env, center)
            rc2 = plate_to_pixel(env, (center[0] + radius, center[1]), guess=rc) if rc is not None else None
            if rc is not None and rc2 is not None:
                px, r_px = (float(rc[1]), float(rc[0])), float(np.hypot(*(rc2 - rc)))
        final_target = (center, radius, polygon, contour_px, px, r_px, why)
        vias = detour(holes, from_xy, center) if (holes and from_xy is not None) else []
        if vias:
            print(f"  routing round the hole: {len(vias)} via point(s)")
            adopt_goal(vias[0], VIA_R, polygon, contour_px, px, r_px, why + ", via point")
        else:
            adopt_goal(center, radius, polygon, contour_px, px, r_px, why)

    def reload_ball(timeout):
        """The ball is lost (fell into a hole): run the elevator at RELOAD_UNITS, forward,
        only until the ball is back on the paper, then stop it. The plate stays LEVEL the
        whole time the elevator runs (user rule). -> (reloaded, seconds)."""
        t0, seen, units0, dir0 = time.time(), 0, elevator.units, elevator.direction
        elevator.units, elevator.direction = RELOAD_UNITS, 1     # not saved: the UI setting stays
        if elevator.on:
            elevator._w4(104, elevator.units)           # goal velocity (direct: set_speed would save it)
        else:
            elevator.start()
        started = True
        print(f"ball lost -- running the elevator at {elevator.units} units until it is back")
        env._servo_level(max_s=3.0)            # level on the camera-measured angle
        level = np.zeros(2, dtype=np.float32)
        while time.time() - t0 < timeout:
            env._write_action(level)               # closed-loop level step on the camera angle, every frame
            xr, yr, _, _, found = env._read_state()
            if env.camera_stale():
                print("reload: camera frozen -- elevator off")
                break
            # back = seen ON the board (corners too), not in a hole and not still in the elevator's
            # outlet above the board edge (on the maze board the ball sat in the outlet at y=143 mm
            # and "seen anywhere" stopped the elevator too early)
            ok = (found and near_hole(holes, (xr, yr), extra=-HOLE_MARGIN_M) is None
                  and abs(xr) < env._x_half + 0.005 and abs(yr) < env._y_half + 0.005)
            seen = seen + 1 if ok else 0
            track_lost(found)
            elevator.poll()
            if ui is not None:
                ui.set_state(**elevator.state(), ball=[float(xr), float(yr)] if found else None,
                             hole_msg=hole_status(f"reloading… {time.time() - t0:.0f} s"))
                ui.set_overlay(ball_px=getattr(env.pipeline.measurements.detector, "ball_pos", None) if found else None)
                with ui._lock:
                    stop_req = any(c.get("cmd") == "stop" or (c.get("cmd") == "elevator" and c.get("op") == "off")
                                   for c in ui._cmds)
                if stop_req:
                    print("reload aborted by the user")
                    break
            if seen >= RELOAD_SEEN_FRAMES or (started and not elevator.on):
                break
            time.sleep(0.03)
        if started:
            elevator.stop_verified("stopped: ball reloaded" if seen >= RELOAD_SEEN_FRAMES else "stopped: reload gave up")
        elevator.units, elevator.direction = units0, dir0
        if ui is not None:
            ui.set_state(**elevator.state())
        dt_ = time.time() - t0
        print(f"reload {'done' if seen >= RELOAD_SEEN_FRAMES else 'FAILED'} after {dt_:.1f} s")
        return seen >= RELOAD_SEEN_FRAMES, dt_

    def aim_at_hole():
        nonlocal vias
        h = min(holes, key=lambda h_: np.hypot(h_["center"][0] - xb, h_["center"][1] - yb))
        vias = []
        adopt_goal(h["center"], h["radius"], None, None, h["px"][:2], h["px"][2], "drop test: into the hole")

    xb = yb = 0.0
    draw_what = "path"
    draw_j0 = (0.0, 0)
    tour, tour_t0, cal_msg = [], 0.0, ""
    maze, maze_alt, maze_following, maze_fell = None, False, False, None
    maze_rest = None
    maze_blend = False
    maze_join_best, maze_join_t, maze_join_alerted, maze_jolts = None, 0.0, False, 0
    maze_back_wp = None                       # real maze: next waypoint back along the route
    run_jolt_at = None
    run_still_t, run_jolts = None, 0          # ball stopped mid-run (real maze): jolt it free
    maze_detour, maze_detoured = None, False
    relevel_times = []
    jerk_sum, jerk_n = 0.0, 0
    lost_xy = None
    lost_since, lost_alerted = None, False
    if ui is not None:
        env.recover_tilts = False            # ball missing -> plate stays level (no recovery tilts)


    def on_wait_tick():
        track_lost(False)
        if elevator is not None:
            elevator.poll()
            ui.set_state(**elevator.state(), ball=None)
    env.on_wait_tick = on_wait_tick

    def track_lost(found):
        """alert once when the ball has been out of sight for LOST_ALERT_S"""
        nonlocal lost_since, lost_alerted
        now_ = time.time()
        if found:
            if lost_alerted:
                _alert(f"ball found again after {now_ - lost_since:.0f} s")
            lost_since, lost_alerted = None, False
            return
        lost_since = lost_since or now_
        if not lost_alerted and now_ - lost_since >= LOST_ALERT_S:
            lost_alerted = True
            _alert(f"ball not found for {LOST_ALERT_S:.0f} s -- it may be stuck under the plate")
    goal_cands = []
    last_goal_check = 0.0
    n_goal_changes = 0
    t_start = time.time()
    integ = np.zeros(2)
    b0 = bias_for(goal)
    if b0 is not None:
        integ = np.clip(b0, -I_MAX, I_MAX)
    move_request = True   # plan the first move right away (no-op if already close)
    pos_buf = deque(maxlen=STUCK_WIN)
    kick = np.zeros(2)
    kicking = False
    anchor = None
    cap_t = None
    lockout_until = 0.0
    brk_mag = 0.0
    last_found = True
    phase, ph_steps, u, d0, start_pos, cell, t_coast = "idle", 0, np.zeros(2), 0.0, None, None, 0.0
    pulse_table = {}
    plate_map = None
    if USE_PLATE_MAP and os.path.exists(PLATE_MAP_PATH):
        plate_map = PlateMap(PLATE_MAP_PATH)
        print(f"using plate map with {len(plate_map.pos)} samples ({PLATE_MAP_PATH})")
    try:
        with open(PULSE_TABLE_PATH) as f:
            pulse_table = {tuple(int(v) for v in k.split(",")): list(val) for k, val in json.load(f).items()}
        print(f"loaded learned pulse settings for {len(pulse_table)} plate cells")
    except (OSError, ValueError):
        pass
    PHASE_CODE = {"idle": 0, "pulse": 1, "brake": 2, "coast": 3}
    t_prev = None
    best_hold = 0.0
    hold_start = None
    try:
        total_in_circle = 0
        total_taken = 0
        episode = 0
        last_snapshot = 0.0
        snap_i = 0
        while total_taken < total_steps and episode < max_episodes:
            episode += 1
            print(f"\n=== episode {episode} ===")
            if lost_xy is not None and auto_reload and elevator is not None:
                in_hole = bool(holes) and near_hole(holes, lost_xy, extra=0.015) is not None
                ok_, secs = reload_ball(RELOAD_TIMEOUT_S)
                if ok_ and in_hole:
                    drop_times.append(secs)
                    if mode == "drop":
                        drop_left -= 1
                        restore_after_reset = drop_left <= 0
                publish_holes("ball reloaded" if ok_ else ("reload failed -- ball not back" if in_hole else ""))
            lost_xy = None
            obs, info = env.reset()
            xb, yb = float(obs[0]), float(obs[1])
            if episode == 1 and holes and mode != "click":
                set_target(*final_target[:6], why="start", from_xy=(xb, yb))   # first move routes round the hole
            if mode == "drop" and drop_left > 0 and holes:
                aim_at_hole()
                publish_holes(f"drop test: {drop_left} to go")
            elif restore_after_reset:
                restore_after_reset = False
                pm, pt = drop_prev or ("click", None)
                mode, live_goal = (pm, pm == "sheet") if pm in ("sheet", "click") else ("click", False)
                if pt is not None and pm in ("sheet", "click"):
                    set_target(*pt[:6], why=pt[6] + " (after drop test)", from_xy=(xb, yb))
                else:
                    running = False
                if ui is not None:
                    ui.set_state(mode=mode)
                publish_holes(f"drop test done: {len(drop_times)} reloads, "
                              f"mean {np.mean(drop_times):.1f} s" if drop_times else "drop test done")
            pos_buf.clear()
            kick[:] = 0
            kicking, anchor, cap_t, last_found = False, None, None, True
            ep_in_circle = 0
            ep_steps = 0
            for i in range(total_steps - total_taken):
                if env.camera_stale():
                    # frozen camera: hold the plate level (motor positions, no camera needed),
                    # elevator off, nothing else until frames come back
                    print("CAMERA FROZEN -- plate held level, waiting for frames")
                    _alert("camera frozen -- rig paused with the plate level; re-plug the camera")
                    if elevator is not None and elevator.on:
                        elevator.stop_verified("stopped: camera frozen")
                    env.hold_session_level()
                    while env.camera_stale(after_s=0.0) and env._grab_frame() is None:
                        time.sleep(0.2)
                    print("camera frames are back -- continuing")
                    t_prev, move = None, None
                    obs, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
                xb, yb, vx, vy, alpha, beta, gx, gy = obs
                dt_real_prev = min(time.time() - t_prev, 0.2) if t_prev is not None else 0.034
                if ui is not None:
                    for c in ui.pop_commands():
                        kind = c.get("cmd")
                        if kind == "start":
                            if mode == "line" and follower is None:
                                line_msg = "No red line to follow -- draw one, then choose 'Follow red line' again"
                                ui.set_state(line_msg=line_msg)
                            else:
                                running = True
                        elif kind == "stop":
                            running = False
                        elif kind == "elevator" and elevator is not None:
                            op = c.get("op")
                            if op == "on":
                                elevator.set_speed(c.get("units"), c.get("dir"))
                                elevator.start()
                            elif op == "off":
                                elevator.stop()
                            elif op == "speed":
                                elevator.set_speed(c.get("units"), c.get("dir"))
                            ui.set_state(**elevator.state())
                        elif kind == "controller":
                            which = c.get("which")
                            if which == "learned" and rl_ctrl is None:
                                print("no trained policy found -- staying on the classic controller")
                            elif which == "odil" and odil_ctrl is None:
                                print("no ODIL policy found -- staying on the current controller")
                            else:
                                use_policy = which == "learned"
                                use_odil = which == "odil"
                                for ctl in (rl_ctrl, odil_ctrl):
                                    if ctl is not None:
                                        ctl.reset()
                                policy_driving = True
                                move, phase, kick = None, "idle", np.zeros(2)
                            ui.set_state(controller="odil" if use_odil else ("learned" if use_policy else "classic"))
                        elif kind == "hybrid":
                            hybrid = bool(c.get("on"))
                            policy_driving = True
                        elif kind == "mode" and c.get("mode") in ("sheet", "click", "line", "path"):
                            mode = c["mode"]
                            live_goal = mode == "sheet"
                            vias, drop_left = [], 0
                            follower, line_active = None, False
                            ui.set_overlay(path_px=None)
                            ui.set_state(mode=mode)
                            if mode == "line" and getattr(env, "_last_frame", None) is not None:
                                ln = detect_line(env, env._last_frame)
                                if ln is None:
                                    line_msg = "No red line found on the sheet"
                                    running = False
                                else:
                                    follower = PathTracker(ln["path"], ln["closed"], np.array([xb, yb]))
                                    line_active, line_boost = False, 0.0
                                    line_msg = (f"{'Loop' if ln['closed'] else 'Line'} of {ln['length'] * 100:.0f} cm found"
                                                f" -- press Start")
                                    goal_polygon = goal_contour_px = goal_px = goal_r_px = None
                                    ui.set_overlay(path_px=ln["path_px"], path_closed=ln["closed"], contour_px=None)
                                print(line_msg)
                                ui.set_state(line_msg=line_msg)
                            if mode == "sheet" and getattr(env, "_last_frame", None) is not None:
                                cand = detect_on_frame(env, env._last_frame)
                                if cand is not None:
                                    ex = cand[2]
                                    set_target(cand[0], cand[1], ex["polygon"], ex["contour_px"], ex["px"][:2],
                                               ex["px"][2], "sheet", from_xy=(xb, yb))
                        elif kind == "test_line":
                            # synthetic route for testing the following without a sheet
                            rr = float(c.get("r", 0.04))
                            if c.get("shape") == "s":
                                tt = np.linspace(0, 1, 200)
                                path = np.column_stack([-0.06 + 0.12 * tt, 0.03 * np.sin(2 * np.pi * tt)])
                                closed_ = False
                            else:
                                th = np.linspace(0, 2 * np.pi, 160, endpoint=False)
                                path, closed_ = np.column_stack([rr * np.cos(th), rr * np.sin(th)]), True
                            mode, live_goal = "line", False
                            follower = PathTracker(path, closed_, np.array([xb, yb]))
                            line_active, line_boost = False, 0.0
                            goal_polygon = goal_contour_px = goal_px = goal_r_px = None
                            line_msg = "Test S-line" if c.get("shape") == "s" else f"Test loop r={rr * 100:.0f} cm"
                            ui.set_state(mode=mode, line_msg=line_msg)
                            ui.set_overlay(path_px=None, contour_px=None)
                            running = True
                        elif kind == "draw_shape":
                            # the ball draws a shape / letters once (open path, drawing order),
                            # routed round the hole, shown on the live view
                            try:
                                cx, cy = float(c.get("x", -0.05)), float(c.get("y", 0.02))
                                size = float(c.get("size", 0.035))
                                if c.get("text"):
                                    path = shapes.text(str(c["text"]), (cx, cy), height=2 * size)
                                else:
                                    path = shapes.shape(str(c.get("shape", "circle")), (cx, cy), size)
                            except ValueError as err:
                                line_msg = str(err)
                                ui.set_state(line_msg=line_msg)
                                path = None
                            if path is not None:
                                lim = np.array([env._x_half - 0.015, env._y_half - 0.015])
                                path = np.clip(path, -lim, lim)
                                routed = [path[0]]
                                for p0, p1 in zip(path[:-1], path[1:]):
                                    routed += list(detour(holes, p0, p1)) + [p1]
                                path = shapes.resample(routed)
                                mode, live_goal = "path", False
                                follower = PathTracker(path, False, np.array([xb, yb]), keep_direction=True,
                                                       v=float(c.get("speed", V_LINE)))
                                line_active, line_boost = False, 0.0
                                goal_polygon = goal_contour_px = goal_px = goal_r_px = None
                                what = f"'{c['text']}'" if c.get("text") else str(c.get("shape", "circle"))
                                draw_what = f"{what} @ {float(c.get('speed', V_LINE)) * 1000:.0f} mm/s"
                                draw_j0 = (jerk_sum, jerk_n)
                                line_msg = f"Drawing {what} ({follower.L * 100:.0f} cm)"
                                pts_px = [plate_to_pixel(env, q) for q in path[::3]]
                                pts_px = [(q[1], q[0]) for q in pts_px if q is not None]
                                ui.set_overlay(path_px=np.array(pts_px) if len(pts_px) > 1 else None,
                                               path_closed=False, contour_px=None)
                                ui.set_state(mode=mode, line_msg=line_msg)
                                running = True
                        elif kind == "click" and mode == "path":
                            row, col = float(c.get("v", -1)) * FRAME_H, float(c.get("u", -1)) * FRAME_W
                            xy = pixel_to_plate(env, row, col)
                            if xy is not None and near_hole(holes, xy) is not None:
                                line_msg = "Too close to the hole -- click a bit further away"
                                ui.set_state(line_msg=line_msg)
                            elif xy is not None and abs(xy[0]) < env._x_half - 0.015 and abs(xy[1]) < env._y_half - 0.015:
                                path_pts.append(xy)
                                path_px.append((col, row))
                                line_msg = f"{len(path_pts)} point(s) -- click more, then Go"
                                ui.set_state(line_msg=line_msg)
                                ui.set_overlay(path_px=np.array(path_px), path_closed=False, contour_px=None)
                        elif kind == "path_clear":
                            path_pts, path_px, follower, line_active = [], [], None, False
                            line_msg = "Path cleared"
                            ui.set_state(line_msg=line_msg)
                            ui.set_overlay(path_px=None)
                        elif kind == "path_go" and mode == "path":
                            if len(path_pts) < 2:
                                line_msg = "Click at least 2 points first"
                            else:
                                routed = [np.asarray(path_pts[0], float)]
                                for p0, p1 in zip(path_pts[:-1], path_pts[1:]):
                                    routed += list(detour(holes, p0, p1)) + [np.asarray(p1, float)]
                                dense = [routed[0]]
                                for p0, p1 in zip(routed[:-1], routed[1:]):
                                    n = max(2, int(np.hypot(*(np.asarray(p1) - p0)) / 0.005))
                                    dense += [p0 + (np.asarray(p1) - p0) * k / n for k in range(1, n + 1)]
                                # drawn paths run in drawing order: start at the first clicked point
                                follower = PathTracker(np.array(dense), False, np.array([xb, yb]), keep_direction=True)
                                line_active, line_boost = False, 0.0
                                goal_polygon = goal_contour_px = goal_px = goal_r_px = None
                                line_msg = f"Driving the path ({follower.L * 100:.0f} cm)"
                                running = True
                            ui.set_state(line_msg=line_msg)
                        elif kind == "target_xy":
                            # scripted target in plate metres (data collection); same routing as a click
                            xy = np.array([float(c["x"]), float(c["y"])])
                            if near_hole(holes, xy) is None and abs(xy[0]) < env._x_half - 0.015 \
                                    and abs(xy[1]) < env._y_half - 0.015:
                                if mode != "click":
                                    mode, live_goal, follower, line_active = "click", False, None, False
                                    ui.set_state(mode=mode)
                                set_target(xy, float(c.get("r", CLICK_R)), why="scripted", from_xy=(xb, yb))
                                running = True
                        elif kind == "calibrate":
                            # self-calibration: level + motor response (blocking, ~25 s), then
                            # the ball tours a 3x3 grid so the classic controller learns the slopes
                            ui.set_state(cal_msg="calibrating: level and motor check…")
                            use_policy = use_odil = False
                            ui.set_state(controller="classic")
                            res, warns = calibrate.tilt_calibration(env, _set_position)
                            env._cal_tpd = None          # re-read the measured ticks per degree
                            if res is None:
                                cal_msg = "; ".join(warns)
                            else:
                                mot = res["motors"]
                                cal_msg = "level at m1 {} / m3 {}. ".format(res["level_ticks"]["1"], res["level_ticks"]["3"]) + \
                                    " ".join(f"motor {k}: {v['response_vs_expected'] * 100:.0f} % response, "
                                             f"{v['play_deg']:.1f} deg play." for k, v in mot.items()) + \
                                    (" WARNINGS: " + "; ".join(warns) if warns else " No warnings.")
                                tour = [np.array(q) for q in calibrate.TOUR if near_hole(holes, q, extra=0.01) is None]
                                mode, live_goal = "click", False
                                ui.set_state(mode=mode)
                                set_target(tour[0], 0.012, why="calibration tour", from_xy=(xb, yb))
                                tour_t0, running = time.time(), True
                                cal_msg += f" Slope tour: 1/{len(tour)}"
                            print("calibration:", cal_msg)
                            ui.set_state(cal_msg=cal_msg)
                            obs, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
                        elif kind == "maze_start":
                            # practise the real maze's route on white paper (maze/route.json)
                            try:
                                if maze is None:
                                    maze = MazePractice(env, plate_to_pixel)
                                maze_alt = bool(c.get("alternate", True))
                                first = c.get("controller", "odil")
                                use_odil, use_policy = first in ("odil", "blend"), False
                                maze_blend = first == "blend"
                                mode, live_goal, vias = "maze", False, []
                                follower = maze.start_run(first, PathTracker, (xb, yb))
                                line_active, line_boost, move, maze_following, maze_fell = False, 0.0, None, False, None
                                draw_j0 = (jerk_sum, jerk_n)
                                goal_polygon = goal_contour_px = goal_px = goal_r_px = None
                                ui.set_overlay(path_px=None, contour_px=None)
                                ui.set_state(mode=mode, maze_msg=maze.status())
                                running = True
                            except (OSError, ValueError, KeyError) as err:
                                ui.set_state(maze_msg=f"maze route not available: {err}")
                        elif kind == "track_policy":
                            # swap the ODIL tracking policy without a restart (learn_loop.py: a
                            # controller restart per round froze the camera, 2026-09-28)
                            p_ = c.get("path", "")
                            try:
                                track_ctrl = TrackPolicy(p_)
                                os.environ["PD_ODIL_TRACK"] = p_          # recorded with each maze run
                                print(f"ODIL tracking policy loaded: {p_}")
                            except (OSError, ValueError, KeyError) as err:
                                print(f"tracking policy {p_} not loaded: {err}")
                        elif kind == "maze_stop":
                            if maze is not None and maze.active:
                                maze.end_run("stopped")
                            mode, follower, line_active, maze_following = "click", None, False, False
                            ui.set_overlay(maze=None)
                            ui.set_state(mode=mode, maze_msg=(maze.status() if maze else ""))
                        elif kind == "drop_test":
                            if not holes:
                                publish_holes("no hole found -- press Find holes")
                            elif elevator is None:
                                publish_holes("no elevator connection -- drop test needs the UI controller")
                            else:
                                if mode != "drop":
                                    drop_prev = (mode, final_target)
                                mode, live_goal, follower, line_active = "drop", False, None, False
                                drop_left, auto_reload = max(1, int(c.get("n", 1))), True
                                ui.set_overlay(path_px=None)
                                aim_at_hole()
                                running = True
                                ui.set_state(mode=mode)
                                publish_holes(f"drop test: {drop_left} to go")
                        elif kind == "holes_detect":
                            holes = detect_holes_stable(env)
                            save_holes(holes)
                            publish_holes()
                        elif kind == "auto_reload":
                            auto_reload = bool(c.get("on"))
                            publish_holes()
                        elif kind == "click" and mode == "click":
                            row, col = float(c.get("v", -1)) * FRAME_H, float(c.get("u", -1)) * FRAME_W
                            xy = pixel_to_plate(env, row, col)
                            xy2 = pixel_to_plate(env, row, col + 10.0)
                            if (xy is None or abs(xy[0]) > env._x_half - 0.015
                                    or abs(xy[1]) > env._y_half - 0.015):
                                print(f"click at pixel ({col:.0f},{row:.0f}) is not on the plate -- ignored")
                            else:
                                px_per_m = 10.0 / max(np.hypot(*(xy2 - xy)), 1e-6) if xy2 is not None else 1000.0
                                if near_hole(holes, xy) is not None:
                                    print("click is inside the hole's keep-out zone -- ignored")
                                    publish_holes("that spot is too close to the hole -- use Drop into hole for that")
                                else:
                                    set_target(xy, CLICK_R, None, None, (col, row), CLICK_R * px_per_m, "click",
                                               from_xy=(xb, yb))
                                    running = True
                if vias and running and np.hypot(xb - vias[0][0], yb - vias[0][1]) < VIA_REACH:
                    vias.pop(0)
                    c_, r_, pg_, cp_, px_, rp_, why_ = final_target
                    if vias:
                        adopt_goal(vias[0], VIA_R, pg_, cp_, px_, rp_, "next via point")
                    else:
                        adopt_goal(c_, r_, pg_, cp_, px_, rp_, why_ + ", past the hole")
                line_ref = None
                if mode in ("line", "path", "maze") and follower is not None and running:
                    if not line_active:
                        # join: bring the ball to the route's current reference point first
                        # (a normal planned move), then start following
                        # the maze start is the true route start (not shifted by the learned offset)
                        join = follower.true_point(follower.s) if mode == "maze" else follower.point(follower.s)
                        join_goal = join
                        if mode == "maze" and maze_detour is not None:
                            # escape: first to a point away from the edge, then to the start
                            join_goal = maze_detour
                            if np.hypot(*(maze_detour - np.array([xb, yb]))) < 0.015:
                                print("  maze: detour point reached -> back to the start")
                                maze_detour = None
                                join_goal = join
                                maze_join_best, maze_jolts = None, 0          # fresh tries from here
                        if mode == "maze" and MAZE_REAL and maze is not None and maze_detour is None:
                            # real maze (2026-09-28): the straight line to the start runs through
                            # walls (a ball sat 20 min against one) -- go back ALONG the route,
                            # via in-sight route points up to 4 cm back, re-picked when reached
                            b_ = np.array([xb, yb])
                            if (maze_back_wp is None or np.hypot(*(maze_back_wp - b_)) < 0.010
                                    or np.hypot(*(maze_back_wp - b_)) > 0.06):
                                maze_back_wp = back_waypoint(maze.route, b_)
                            if maze_back_wp is not None:
                                join_goal = maze_back_wp
                        line_off = follower.off_line((xb, yb))
                        dj = float(np.hypot(*(join - np.array([xb, yb]))))
                        if (mode == "maze" and elevator is not None and last_found
                                and (abs(yb) > env._y_half + 0.005 or abs(xb) > env._x_half + 0.005)):
                            # the ball is still in the elevator's outlet above the board: push it out
                            print(f"  maze: ball at ({xb * 1000:.0f}, {yb * 1000:.0f}) mm is off the board (outlet) -> elevator")
                            reload_ball(RELOAD_TIMEOUT_S)
                            maze_join_best, maze_join_t = None, time.time()
                        if mode == "maze":
                            # returning to the start: alert once if it has not got closer for 30 s
                            # (a ball stuck on the paper seam sat still for 53 min unnoticed)
                            if maze_join_best is None or dj < maze_join_best - 0.01:
                                maze_join_best, maze_join_t, maze_jolts = dj, time.time(), 0
                            elif time.time() - maze_join_t > 30 and maze_jolts < 3:
                                # stuck (paper seam): rock it over -- tilt back 0.5 s so it rolls
                                # away from the edge, then snap to full tilt towards the start
                                # (user: "just jolt"); up to 3 tries, then alert
                                maze_jolts += 1
                                d = np.array(join_goal) - np.array([xb, yb])
                                d = d / max(np.hypot(*d), 1e-6)
                                print(f"  maze: stuck at ({xb * 1000:.0f}, {yb * 1000:.0f}) mm -- jolt {maze_jolts}/3")
                                env._hold_tilt((-0.5 * d).astype(np.float32), 0.5)
                                env._hold_tilt((1.0 * d).astype(np.float32), 1.0)
                                maze_join_t = time.time() - 20          # next try after 10 s
                                move, move_request = None, True
                            elif time.time() - maze_join_t > 30 and maze_detour is None and not maze_detoured:
                                # jolts did not help: go 4 cm towards the plate centre first
                                b_ = np.array([xb, yb])
                                maze_detour = b_ - 0.04 * b_ / max(np.hypot(*b_), 1e-6)
                                maze_detoured = True
                                maze_join_t = time.time()
                                print(f"  maze: detour via ({maze_detour[0] * 1000:.0f}, {maze_detour[1] * 1000:.0f}) mm")
                            elif time.time() - maze_join_t > 30 and not maze_join_alerted:
                                maze_join_alerted = True
                                _alert(f"maze: ball not getting back to the start (stuck at x={xb * 1000:.0f} "
                                       f"y={yb * 1000:.0f} mm, 3 jolts did not free it) -- paper seam?")
                            # a maze run starts only from REST at the start (a ball arriving at
                            # speed overshot through the wall below the start within 0.1-1 s)
                            if dj < 0.012 and np.hypot(vx, vy) < 0.02:
                                maze_rest = maze_rest or time.time()
                            else:
                                maze_rest = None
                            ready = maze_rest is not None and time.time() - maze_rest > 0.5
                        else:
                            ready = dj < LINE_JOIN_M
                        if ready:
                            line_active, move, maze_back_wp = True, None, None
                            maze_join_best, maze_join_alerted = None, False
                            maze_detour, maze_detoured = None, False
                        elif goal != (float(join_goal[0]), float(join_goal[1])):
                            goal, goal_tol = (float(join_goal[0]), float(join_goal[1])), LINE_TOL
                            env.goal = np.array(goal, dtype=np.float32)
                            env.fixed_goal, env.goal_tolerance = goal, goal_tol
                            move_request = True
                    if line_active:
                        line_ref = follower.update((xb, yb), max(dt_real_prev, 1e-3))
                        line_off = line_ref["off"]
                        goal, goal_tol = (float(line_ref["p"][0]), float(line_ref["p"][1])), LINE_TOL
                        env.goal = np.array(goal, dtype=np.float32)
                        env.fixed_goal, env.goal_tolerance = goal, goal_tol
                        if line_ref["finished"]:
                            line_active, line_ref = False, None  # hold at the end with normal balancing
                            if not line_msg.startswith("Reached the end") and mode != "maze":
                                acc = follower.accuracy()
                                line_msg = "Reached the end of the line" + (
                                    f" -- off the line: median {acc['median_mm']:.1f} mm, 90% within "
                                    f"{acc['p90_mm']:.1f} mm, max {acc['max_mm']:.1f} mm, jerk "
                                    f"{(jerk_sum - draw_j0[0]) / max(1, jerk_n - draw_j0[1]):.4f}" if acc else "")
                                print(line_msg)
                                if acc:
                                    try:
                                        with open(os.path.join(os.path.dirname(__file__), "..", "phase3_logs",
                                                               "draw_accuracy.jsonl"), "a") as f_:
                                            f_.write(json.dumps({"t": time.time(), "what": draw_what,
                                                                 "controller": "odil" if use_odil else "classic",
                                                                 "jerk": (jerk_sum - draw_j0[0]) / max(1, jerk_n - draw_j0[1]),
                                                                 "length_m": follower.L, **acc}) + "\n")
                                    except OSError:
                                        pass
                                if ui is not None:
                                    ui.set_state(line_msg=line_msg)
                    if mode == "maze" and maze is not None and maze.active:
                        result = None
                        if maze_following and not line_active:
                            result = "finished"              # the block above just reached the end
                        elif line_active and last_found:
                            maze_following = True
                            result = maze.step((xb, yb), True)
                            if maze.retry_req:
                                # stuck 15 s (real maze): back the reference up 2 cm so the ball rolls
                                # back and takes the spot again with some speed; fresh jolts
                                maze.retry_req = False
                                follower.s, follower.v = max(0.0, follower.s - 0.02), 0.0
                                run_still_t, run_jolts = None, 0
                                print(f"  maze: stuck at ({xb * 1000:.0f}, {yb * 1000:.0f}) mm -- retry {maze.retries} "
                                      f"(back 2 cm, then again)")
                                if maze.retries == 10:
                                    _alert(f"maze: ball stuck at x={xb * 1000:.0f} y={yb * 1000:.0f} mm, 10 retries so far")
                        if result:
                            jr = (jerk_sum - draw_j0[0]) / max(1, jerk_n - draw_j0[1])
                            print("maze:", maze.end_run(result, jr))
                            maze_fell = None
                            if result.startswith("fell into hole"):
                                hk = int(result.split()[-1])
                                maze_fell = maze.holes_px[hk] if hk < len(maze.holes_px) else None
                            maze_following = False
                            from maze_practice import CONTROLLERS as _MC
                            nxt = _MC[(_MC.index(maze.ctl) + 1) % len(_MC)] if maze_alt else maze.ctl
                            use_odil, use_policy = nxt in ("odil", "blend"), False
                            maze_blend = nxt == "blend"
                            if track_ctrl is not None:
                                track_ctrl.reset()
                            follower = maze.start_run(nxt, PathTracker, (xb, yb))   # back to the start
                            maze_rest = None
                            run_still_t, run_jolts = None, 0
                            line_active, line_boost, move, line_ref = False, 0.0, None, None
                            draw_j0 = (jerk_sum, jerk_n)
                        if ui is not None:
                            ui.set_state(maze_msg=maze.status(),
                                         controller=("odil+classic" if maze_blend else "odil") if use_odil else "classic")
                    edge_x, edge_y = edge_limits(goal, goal_tol, quiet=True)
                    goal_px = goal_r_px = None
                if ui is not None or (mode == "line" and follower is not None):
                    gx, gy = goal[0] - xb, goal[1] - yb
                dist_now = float(np.hypot(gx, gy))
                # 0 = fully "far" gains, 1 = fully "near" gains, smoothly blended
                # over GAIN_BLEND_WIDTH so there's no gain discontinuity right at
                # the switch point (which would itself show up as a small kick).
                near_frac = np.clip(
                    (GAIN_BLEND_DIST - dist_now) / GAIN_BLEND_WIDTH, 0.0, 1.0
                )
                kp = KP_FAR + near_frac * (KP_NEAR - KP_FAR)
                kd = KD_FAR + near_frac * (KD_NEAR - KD_FAR)
                t_now = time.time()
                dt_real = 0.0 if t_prev is None else min(t_now - t_prev, 0.2)
                t_prev = t_now
                e = np.array([gx, gy])
                stationary = False
                if last_found:  # frozen obs during not-found frames must not read as "still"
                    pos_buf.append((xb, yb))
                    P = np.array(pos_buf)
                    stationary = (len(P) == STUCK_WIN and np.ptp(P[:, 0]) < STUCK_PTP
                                  and np.ptp(P[:, 1]) < STUCK_PTP)
                    # Pulses work anywhere on the plate: with the approach speed capped, a
                    # STILL ball far from the goal only gets kd*V_REF_MAX = 0.16 (0.8 deg),
                    # below the ~2 deg breakaway -- it sat 11 cm away for a whole run.
                    if move is None and not line_active and running and dist_now > MOVE_MIN_M and (
                            move_request or (stationary and phase == "idle" and t_now >= lockout_until)):
                        u_m = e / max(dist_now, 1e-6)
                        T_m, prof = trapezoid(dist_now, V_MOVE, A_MOVE)
                        move = {"u": u_m, "start": np.array([xb, yb]), "D": dist_now, "T": T_m, "prof": prof,
                                "t0": None, "boost": 0.0, "t_plan": t_now}
                        move_request, phase, n_moves = False, "idle", n_moves + 1
                        print(f"  move {n_moves}: {dist_now * 1000:.0f} mm, planned {T_m:.1f} s")
                    if move is not None:
                        pass  # the planned move owns the ball; pulses wait
                    elif phase == "idle" and not line_active:
                        if stationary and dist_now > R_DONE and t_now >= lockout_until:
                            u = e / max(dist_now, 1e-6)
                            d0, start_pos = dist_now, np.array([xb, yb])
                            cell = (int(round(xb / PULSE_CELL_M)), int(round(yb / PULSE_CELL_M)))
                            a0 = PULSE_A0
                            if plate_map is not None and plate_map.static_deg((xb, yb)) is not None:
                                a0 = float(np.clip(1.1 * plate_map.static_deg((xb, yb)) / 5.0, 0.3, PULSE_A_MAX))
                            pulse_table.setdefault(cell, [a0, PULSE_N0])
                            phase, ph_steps = "pulse", 0
                    elif phase == "pulse":
                        ph_steps += 1
                        if ph_steps >= pulse_table[cell][1]:
                            phase, ph_steps = "brake", 0
                    elif phase == "brake":
                        ph_steps += 1
                        if ph_steps >= pulse_table[cell][1]:
                            phase, t_coast = "coast", t_now
                            pos_buf.clear()
                    elif phase == "coast" and (stationary or t_now - t_coast > COAST_MAX_S):
                        adv = float(np.dot(np.array([xb, yb]) - start_pos, u))
                        A, N = pulse_table[cell]
                        if adv < 0.001:
                            A = min(A + 0.10, PULSE_A_MAX)  # didn't break free -> stronger
                        else:
                            N = int(np.clip(round(N * np.sqrt(d0 / max(adv, 0.001))), PULSE_N_MIN, PULSE_N_MAX))
                        pulse_table[cell] = [A, N]
                        try:
                            with open(PULSE_TABLE_PATH, "w") as f:
                                json.dump({f"{k[0]},{k[1]}": v for k, v in pulse_table.items()}, f)
                        except OSError:
                            pass
                        print(f"  pulse: d0={d0 * 1000:.1f}mm advance={adv * 1000:.1f}mm -> cell {cell} A={A:.2f} N={N}")
                        phase = "idle"
                        lockout_until = t_now + PULSE_LOCKOUT_S
                        pos_buf.clear()
                if phase == "pulse":
                    kick = pulse_table[cell][0] * u
                elif phase == "brake":
                    kick = -BRAKE_A * u
                else:
                    kick = np.zeros(2)
                kicking = phase != "idle"
                # the integral learns the level bias only from a rolling ball, never stiction
                if (running and move is None and not line_active and not ((use_policy or use_odil) and policy_driving)
                        and dist_now < I_ZONE and not kicking and not stationary):
                    integ = np.clip(integ + KI * e * dt_real, -I_MAX, I_MAX)
                v_ref = (kp / kd) * e
                v_ref_mag = float(np.hypot(*v_ref))
                v_allow = min(V_REF_MAX, float(np.sqrt(2 * A_BRAKE * max(dist_now - 0.5 * goal_tol, 0.0))))
                if v_ref_mag > v_allow:
                    v_ref *= v_allow / max(v_ref_mag, 1e-9)
                pd = kd * (v_ref - np.array([vx, vy]))
                # per-area level feedforward from the plate map (action units: 5 deg = 1)
                ff = plate_map.level_deg((xb, yb)) / 5.0 if plate_map is not None else np.zeros(2)
                if phase in ("pulse", "brake"):
                    pd = pd - np.dot(pd, u) * u  # the pulse/brake own the goal direction
                action = np.clip(
                    np.array([pd[0] + integ[0] + kick[0] + ff[0], pd[1] + integ[1] + kick[1] + ff[1]], dtype=np.float32),
                    -0.8, 0.8,
                )
                if move is not None and running:
                    pos, vel = np.array([xb, yb]), np.array([vx, vy])
                    um = move["u"]
                    if move["t0"] is None:
                        # break free once: planned initial push + a slowly growing boost
                        move["boost"] = min(MOVE_BOOST_MAX, move["boost"] + MOVE_BOOST_RATE * dt_real)
                        push = (A_MOVE + A_ROLL_FF) / ACC_PER_ACTION + move["boost"]
                        a_move = um * push + integ
                        adv = float(np.dot(pos - move["start"], um))
                        if adv > 0.002 or float(np.dot(vel, um)) > 0.015:
                            # it rolls: start the plan where its speed/position match
                            v_now = max(float(np.dot(vel, um)), 0.0)
                            t_m = min(v_now / A_MOVE, move["T"] / 2)
                            s_m = move["prof"](t_m)[0]
                            move["t0"] = t_now - t_m
                            move["start"] = pos - um * s_m
                        elif t_now - move["t_plan"] > 3.0:
                            move = None      # couldn't break free: leave it to the pulses
                    if move is not None and move["t0"] is not None:
                        tm = t_now - move["t0"]
                        s_r, v_r, a_r = move["prof"](tm)
                        p_ref = move["start"] + um * s_r
                        ff_m = (a_r + (A_ROLL_FF if v_r > 0.005 else 0.0)) / ACC_PER_ACTION
                        a_move = um * ff_m + KP_MOVE * (p_ref - pos) + KD_MOVE * (um * v_r - vel) + integ
                        err = float(np.hypot(*(p_ref - pos)))
                        if tm > move["T"] or err > MOVE_ABORT_M:
                            if err > MOVE_ABORT_M:
                                print(f"  move {n_moves} ended early (tracking error {err * 1000:.0f} mm)")
                            move = None
                            lockout_until = t_now + 0.3
                            pos_buf.clear()
                    if move is not None:
                        action = np.clip(np.asarray(a_move, dtype=np.float32), -0.8, 0.8)
                if line_ref is not None and running:
                    pos, vel = np.array([xb, yb]), np.array([vx, vy])
                    th_ = line_ref["t_hat"]
                    bst = th_
                    if mode == "maze" and np.hypot(*(line_ref["p"] - pos)) > 0.003:
                        # real walls (2026-09-28): at a corner the route tangent at the reference
                        # points past the wall end -- the boost pressed the ball into the wall at
                        # full tilt. Push straight at the (in-sight, see los_tol) reference instead.
                        bst = (line_ref["p"] - pos) / np.hypot(*(line_ref["p"] - pos))
                    if np.hypot(*vel) < 0.01 and line_ref["lag"] > 0.005:
                        line_boost = min(LINE_BOOST_MAX, line_boost + LINE_BOOST_RATE * dt_real)
                    else:
                        line_boost = max(0.0, line_boost - 2 * LINE_BOOST_RATE * dt_real)
                    ff_l = (line_ref["a"] + (A_ROLL_FF * th_ if np.hypot(*line_ref["v"]) > 0.005 else 0.0)) / ACC_PER_ACTION
                    a_line = (ff_l + KP_LINE * (line_ref["p"] - pos) + KD_LINE * (line_ref["v"] - vel)
                              + integ + line_boost * bst)
                    action = np.clip(np.asarray(a_line, dtype=np.float32), -0.8, 0.8)
                    if use_odil and track_ctrl is not None:
                        a_odil = np.asarray(track_ctrl(line_ref, pos, vel, dt_real if dt_real > 0 else None), dtype=np.float32)
                        if mode == "maze" and maze_blend:
                            a_odil = 0.5 * a_odil + 0.5 * np.asarray(a_line, dtype=np.float32)   # ODIL + classic
                        # the same breakaway boost as the classic line law, in full: a ball stuck on the
                        # paper behind the reference stayed there (the ODIL tracker has no stiction term)
                        a_odil = a_odil + (1.0 if not (mode == "maze" and maze_blend) else 0.5) * line_boost * bst
                        action = np.clip(a_odil, -0.8, 0.8)
                    if mode == "maze" and maze_following:
                        # real maze (2026-09-28): the ball sat still at a wall end at ~4 deg for the
                        # whole 15 s stall timeout. After 4 s still, tilt back briefly and snap to
                        # full tilt at the reference (user: problem-solve a stuck ball); 3 tries
                        # still = moved < 3 mm in 4 s (the speed estimate jitters up to ~18 mm/s on a
                        # resting ball, which kept resetting a speed-based timer)
                        if run_still_t is None or np.hypot(*(pos - run_still_t[0])) > 0.003:
                            run_still_t = (pos.copy(), t_now)
                        elif (t_now - run_still_t[1] > 4.0
                              and np.hypot(*(line_ref["p"] - pos)) > 0.004
                              and (run_jolts < 3 or run_jolt_at is None or np.hypot(*(pos - run_jolt_at)) > 0.02)):
                            if run_jolt_at is None or np.hypot(*(pos - run_jolt_at)) > 0.02:
                                run_jolts = 0               # a new spot: fresh tries
                            run_jolt_at = pos.copy()
                            run_jolts += 1
                            d = line_ref["p"] - pos
                            d = d / max(np.hypot(*d), 1e-6)
                            # 1st straight at the reference, then 50 deg to either side: a ball
                            # resting against a wall end was pressed back onto it by every straight
                            # jolt (stuck at -128, 9 through 4 retries, 2026-09-28)
                            ang = np.radians((0.0, 50.0, -50.0)[(run_jolts - 1) % 3])
                            d = np.array([d[0] * np.cos(ang) - d[1] * np.sin(ang),
                                          d[0] * np.sin(ang) + d[1] * np.cos(ang)], dtype=np.float32)
                            print(f"  maze: run stalled at ({xb * 1000:.0f}, {yb * 1000:.0f}) mm -- jolt {run_jolts}/3")
                            env._hold_tilt(-0.4 * d, 0.3)
                            env._hold_tilt(1.0 * d, 0.6)
                            run_still_t = None
                elif track_ctrl is not None:
                    track_ctrl.reset()          # fresh observer / integral for the next line
                # Hard override once near an edge -- full brake straight back toward
                # center on whichever axis is in danger, overriding the blended 2D
                # goal-seeking above. Without this the combined X+Y correction can
                # point diagonally at a corner instead of cutting back through the
                # middle of the edge it's near.
                # Gentle, proportional wall guard (was a full +-1.0 = 5 deg slam, which
                # shot the ball back across the plate at 140-160 mm/s every time it
                # overshot toward the wall -- the recurring ~46 mm overshoots).
                # It guarantees a MINIMUM push away from the wall but never caps a
                # stronger one: a stiction pulse must still get through (overriding it
                # left a ball stuck at the sticky top edge for a whole run).
                for ax, pos, edge in ((0, xb, edge_x), (1, yb, edge_y)):
                    if mode == "maze":
                        break       # the maze start is 12 mm from the plate edge: the guard made it bounce
                    if abs(pos) > edge:
                        push = min(EDGE_PUSH_MAX, EDGE_PUSH_MIN + EDGE_PUSH_GAIN * (abs(pos) - edge))
                        away = -np.sign(pos)
                        action[ax] = away * max(push, away * action[ax])
                # A dedicated trap-zone override was tried here (both a steady
                # push and an in-loop oscillation toward the goal) and REMOVED --
                # confirmed directly to be counterproductive, not just ineffective.
                # env.step()'s own max_action_delta rate limit smooths any in-loop
                # oscillation enough that it keeps the ball moving JUST enough to
                # never trip plate_env.py's wedge detector, without ever moving it
                # enough to actually escape -- silently burning entire episodes
                # (600/600 steps, 0% in-circle) vibrating in place instead of ever
                # escalating to the real, un-rate-limited shake escape in
                # _attempt_unstick(), which is the only thing that has ever
                # actually freed the ball from this specific spot. Let the wedge
                # detector fire and do its job instead of masking it.
                # maze practice: getting to the start and resting there always uses ODIL + classic
                # settle (user; the best target-holding combination), whatever runs the route
                maze_join = mode == "maze" and not line_active and odil_ctrl is not None
                if ((use_policy or use_odil) and running and mode not in ("line", "path", "maze")) or (maze_join and running):
                    handover = max(HANDOVER_MIN, HANDOVER_R_SCALE * goal_tol)
                    if not hybrid and not maze_join:
                        policy_driving = True
                    elif policy_driving and dist_now < handover:
                        policy_driving = False
                        move, move_request, phase = None, False, "idle"
                        pos_buf.clear()
                        print(f"  hybrid: near field at {dist_now * 1000:.0f} mm -> classic settle")
                    elif not policy_driving and dist_now > HANDBACK_SCALE * handover:
                        policy_driving = True
                        for ctl in (rl_ctrl, odil_ctrl):
                            if ctl is not None:
                                ctl.reset()
                        print(f"  hybrid: ball {dist_now * 1000:.0f} mm out -> policy approach")
                    if policy_driving:
                        if use_odil or maze_join:
                            action = odil_ctrl.action(goal, (xb, yb), (vx, vy), dt_real if dt_real > 0 else None,
                                                      goal_tol).astype(np.float32)
                        else:
                            action = rl_ctrl.action(goal, goal_tol, (xb, yb), (vx, vy), alpha, beta).astype(np.float32)
                        move, move_request, phase, kick = None, False, "idle", np.zeros(2)
                if not running:
                    action = np.zeros(2, dtype=np.float32)   # stopped: hold the plate level
                    phase, kick, move = "idle", np.zeros(2), None
                # ball not seen, stopped, or elevator running -> level (camera-measured 0 deg)
                env.hold_level = ui is not None and (not running or not last_found
                                                     or (elevator is not None and elevator.on))
                if env.hold_level:
                    action = np.zeros(2, dtype=np.float32)
                    phase, kick, move = "idle", np.zeros(2), None
                if getattr(env, "motor_not_following", None) is not None and running:
                    # the motor sits at its cap and the angle does not move: the level reference has
                    # drifted -> re-level on the camera, drop this run, carry on; alert only if it
                    # keeps happening (3x in 5 min)
                    mid = env.motor_not_following
                    env.motor_not_following = None
                    env._capped_steps = {}
                    relevel_times = [t_ for t_ in relevel_times if time.time() - t_ < 300] + [time.time()]
                    print(f"  motor {mid} at its cap without the angle moving -> re-levelling ({len(relevel_times)}/3 in 5 min)")
                    env._servo_level(max_s=6.0)
                    env._session_level = {k: int(v) for k, v in env._cmd_ticks.items()}
                    if maze is not None and maze.active and mode == "maze":
                        maze.end_run("re-levelled")
                        follower = maze.start_run(maze.ctl, PathTracker, (xb, yb))
                        line_active, maze_following, maze_rest, move = False, False, None, None
                    if len(relevel_times) >= 3:
                        _alert(f"motor {mid}: re-levelled 3 times in 5 min -- paused, plate level")
                        running, mode, follower, line_active = False, "click", None, False
                        relevel_times = []
                        if ui is not None:
                            ui.set_state(running=False, mode=mode, maze_msg="PAUSED: re-levelled 3 times in 5 min")
                    action = np.zeros(2, dtype=np.float32)
                prev_applied = np.array(env._last_commanded_action, dtype=float)
                obs, reward, terminated, truncated, info = env.step(action)
                # smoothness (same "jerk" as rl_sim/eval_controllers): squared change of the
                # applied command per step, summed; clients take differences over a trip
                jerk_sum += float(np.sum((np.asarray(env._last_commanded_action, dtype=float) - prev_applied) ** 2))
                jerk_n += 1
                if rl_ctrl is not None:
                    rl_ctrl.record_applied(env._last_commanded_action)
                if odil_ctrl is not None:
                    odil_ctrl.record_applied(env._last_commanded_action, dt_real if dt_real > 0 else None)
                dist = float(np.hypot(obs[6], obs[7]))
                # Only a frame where the ball was actually detected counts: during
                # step()'s not-found grace frames obs holds the frozen last position.
                ball_found = bool(info.get("ball_found", True))
                last_found = ball_found
                track_lost(ball_found)
                if mode in ("line", "path", "maze") and follower is not None:
                    in_circle = ball_found and line_off is not None and line_off < LINE_TOL * 1.25
                elif goal_polygon is not None and len(goal_polygon) >= 3:
                    in_circle = ball_found and inside_region(goal_polygon, (obs[0], obs[1]))
                else:
                    in_circle = ball_found and dist < goal_tol
                if vias:
                    in_circle = False
                if info.get("status") == "ball_lost":
                    lost_xy = np.array(env._prev_ball, dtype=float)
                    if mode == "maze" and maze is not None and maze.active and maze_following:
                        # the real maze: the ball fell through a hole -> the run ends here; the
                        # reload brings it back and the next run starts from the start
                        k_ = None
                        if maze.holes:
                            k_ = int(np.argmin([np.hypot(*(lost_xy - c)) for c, _ in maze.holes]))
                        print("maze:", maze.end_run(f"fell into a real hole (near {k_})",
                                                    (jerk_sum - draw_j0[0]) / max(1, jerk_n - draw_j0[1])))
                        maze_following = False
                        follower = maze.start_run(maze.ctl, PathTracker, (xb, yb))
                        line_active, line_ref, move = False, None, None
                        draw_j0 = (jerk_sum, jerk_n)
                ep_in_circle += int(in_circle)
                ep_steps += 1
                # learn the local level where the ball rests in the target
                if running and move is None and ball_found and dist < R_DONE and phase == "idle" \
                        and np.hypot(obs[2], obs[3]) < 0.01:
                    rest_since = rest_since or time.time()
                    if time.time() - rest_since > BIAS_LEARN_S:
                        c = bias_cell(goal)
                        old = bias_table.get(c)
                        bias_table[c] = integ.copy() if old is None else 0.5 * old + 0.5 * integ
                        rest_since = time.time() + 1e9  # once per rest
                        try:
                            with open(BIAS_TABLE_PATH, "w") as f:
                                json.dump({f"{k[0]},{k[1]}": v.tolist() for k, v in bias_table.items()}, f)
                        except OSError:
                            pass
                else:
                    rest_since = None if not (rest_since and rest_since > time.time()) else rest_since

                now = time.time()
                # A lone missed detection neither extends nor breaks a hold (the
                # detector drops single frames on a stationary ball); a detected
                # frame outside the circle, or a real ball_lost, does break it.
                if in_circle:
                    if hold_start is None:
                        hold_start = now
                    best_hold = max(best_hold, now - hold_start)
                elif ball_found or info["status"] == "ball_lost":
                    hold_start = None
                if tour and running and ((hold_start and now - hold_start > 2.5) or now - tour_t0 > 15.0):
                    tour.pop(0)
                    if tour:
                        set_target(tour[0], 0.012, why="calibration tour", from_xy=(float(obs[0]), float(obs[1])))
                        tour_t0 = now
                        cal_msg = cal_msg.rsplit(" Slope tour:", 1)[0] + \
                            f" Slope tour: {len(calibrate.TOUR) - len(tour) + 1}/{len(calibrate.TOUR)}"
                    else:
                        cal_msg = cal_msg.rsplit(" Slope tour:", 1)[0] + f" Slope tour done ({len(bias_table)} areas learned)."
                        print("calibration:", cal_msg)
                    if ui is not None:
                        ui.set_state(cal_msg=cal_msg)
                if elevator is not None:
                    elevator.poll()
                    ui.set_state(**elevator.state())
                if ui is not None:
                    ui.set_state(controller=(("odil" if use_odil else "learned") + ("" if policy_driving else "+classic settle"))
                                 if (use_policy or use_odil) else "classic", hybrid=hybrid,
                                 running=running, mode=mode,
                                 ball=[float(obs[0]), float(obs[1])] if ball_found else None,
                                 goal=[goal[0], goal[1]], goal_r=goal_tol,
                                 dist=(line_off if (mode in ("line", "path", "maze") and follower is not None) else dist) if ball_found else None,
                                 in_target=bool(in_circle), hold=(now - hold_start) if hold_start else 0.0,
                                 jerk_sum=jerk_sum, jerk_n=jerk_n,
                                 hz=(1.0 / dt_real) if dt_real > 0 else None)
                    if mode == "maze" and maze is not None:
                        ui.set_overlay(maze=maze.overlay(maze_fell))
                    ui.set_overlay(ball_px=getattr(env.pipeline.measurements.detector, "ball_pos", None) if ball_found else None,
                                   goal_px=goal_px, goal_r_px=goal_r_px, contour_px=goal_contour_px)
                log.write(f"{now - t_start:.3f},{episode},{i},{obs[0]:.5f},{obs[1]:.5f},{obs[2]:.4f},"
                          f"{obs[3]:.4f},{obs[4]:.5f},{obs[5]:.5f},{dist:.5f},{int(in_circle)},{int(ball_found)},"
                          f"{action[0]:.3f},{action[1]:.3f},{info['status']},"
                          f"{integ[0]:.4f},{integ[1]:.4f},{kick[0]:.4f},{kick[1]:.4f},{4 if move is not None else PHASE_CODE[phase]},"
                          + (",".join(f"{v:.1f}" for v in env.last_inner_corners.ravel())
                             if getattr(env, "last_inner_corners", None) is not None else ",,,,,,,")
                          + f",{goal[0]:.5f},{goal[1]:.5f},{goal_tol:.5f}\n")

                if live_goal and mode == "sheet" and now - last_goal_check >= GOAL_CHECK_S and getattr(env, "_last_frame", None) is not None:
                    last_goal_check = now
                    cand = detect_on_frame(env, env._last_frame)
                    # a provisional goal (fitted with the ball covering part of it) is
                    # refined by the first clean view, however small the difference
                    mv, rc = (0.0005, 0.0005) if goal_provisional else (GOAL_MOVE_M, GOAL_RADIUS_CHANGE_M)
                    differs = cand is not None and (
                        np.hypot(cand[0][0] - goal[0], cand[0][1] - goal[1]) > mv
                        or abs(cand[1] - goal_tol) > rc)
                    if not differs:
                        goal_cands.clear()
                    else:
                        if goal_cands and np.hypot(*(cand[0] - goal_cands[-1][0])) > GOAL_AGREE_M:
                            goal_cands.clear()
                        goal_cands.append(cand)
                        if len(goal_cands) >= GOAL_CONFIRM:
                            c = np.median([cd[0] for cd in goal_cands], axis=0)
                            r_med = float(np.median([cd[1] for cd in goal_cands]))
                            ex = goal_cands[-1][2]
                            set_target(c, r_med, ex["polygon"], ex["contour_px"], ex["px"][:2], ex["px"][2], "sheet",
                                       from_xy=(float(obs[0]), float(obs[1])))
                            save_last_goal(goal, goal_tol, goal_polygon, goal_contour_px)

                if now - last_snapshot >= SNAPSHOT_EVERY_S:
                    frame = env._grab_frame()
                    if frame is not None:
                        path = os.path.join(SNAPSHOT_DIR, f"snap_{snap_i:04d}.jpg")
                        cv2.imwrite(path, frame)
                        snap_i += 1
                    last_snapshot = now
                if i % 10 == 0 or info["status"] not in ("running", "in_circle"):
                    print(f"  step {i:4d}: xb={obs[0]:+.4f} yb={obs[1]:+.4f} dist={dist:.4f} "
                          f"{'IN' if in_circle else '  '}  action=({action[0]:+.2f},{action[1]:+.2f})  "
                          f"I=({integ[0]:+.3f},{integ[1]:+.3f}) K=({kick[0]:+.2f},{kick[1]:+.2f}){'*' if kicking else ''}  "
                          f"status={info['status']}")
                if hold_target > 0 and best_hold >= hold_target:
                    print(f"  reached {hold_target:.0f}s continuous hold -- stopping")
                    break
                if terminated or truncated:
                    print(f"  episode {episode} ended at step {i}: {info['status']}")
                    break
            total_in_circle += ep_in_circle
            total_taken += ep_steps
            pct = 100 * ep_in_circle / max(1, ep_steps)
            print(f"  episode {episode} summary: {ep_in_circle}/{ep_steps} steps in circle ({pct:.1f}%)")
            hold_start = None  # a hold never spans a reset()
            if hold_target > 0 and best_hold >= hold_target:
                break

        overall_pct = 100 * total_in_circle / max(1, total_taken)
        print(f"\n=== overall: {total_in_circle}/{total_taken} steps in circle ({overall_pct:.1f}%) "
              f"across {episode} episode(s) ===")
        rate = total_taken / max(1e-6, time.time() - t_start)
        print(f"=== goal changes during run: {n_goal_changes} ===")
        print(f"=== longest continuous hold inside goal circle: {best_hold:.2f} s "
              f"(target {HOLD_TARGET_S:.0f} s, {'SUCCESS' if best_hold >= HOLD_TARGET_S else 'not reached'}), "
              f"~{rate:.1f} steps/s overall, log: {log_path} ===")
    finally:
        log.close()
        if elevator is not None:
            # always (not only if we think it is on): a stop interrupted mid-packet by
            # the shutdown signal once left it running at 320 after the controller exited
            try:
                elevator.stop_verified("stopped: controller shut down")
            except BaseException as e:
                print(f"elevator stop failed: {e}")
        env.close()


if __name__ == "__main__":
    main()
