#!/usr/bin/env python3
"""
Ball-on-plate: ODIL vs RL, both trained ONLY on the physical rig (2026-10-02).

One process owns the rig for the whole night (no controller restarts -- they froze the camera):

  ODIL  rounds:  drive the rig for --odil-min minutes (round 0: random tilts, no prior controller;
                 later rounds: the current ODIL policy to random targets) -> fit the physics to all
                 ODIL rig data so far (rl_sim/sysid_rig.py) -> train ODIL in that fitted model
                 (rl_sim/odil_plate_v9.py, 3-stage delay = the v6/v11 recipe) with the plate held
                 level -> test.
  SAC (RL):      Soft Actor-Critic from scratch on the rig (stable-baselines3), the same observation,
                 action limits and reward as the simulated RL (rl_sim/plate_goal_env.py); network
                 updates between episodes with the plate level, so the 29 Hz loop is never slowed;
                 test every --sac-test-min minutes of driving; checkpoints.
  Test:          the same 30 fixed random targets (seed 2027, r = 12 mm, away from the hole) for
                 every controller: reached (within 8 s), time to reach, share of the 4 s after
                 arrival inside, final distance, jerk (mean squared change of the applied command).
                 Every result -> phase3_logs/rig_learn.jsonl with the RIG MINUTES of training data.

Safety: motor caps fixed around the startup position (plate_env); this script aborts and releases
the motors if a tilt motor's command leaves +-1300 ticks of its start; a frozen camera holds the
plate level until frames return; a ball not visible -> plate level until it is seen again (no
elevator: on the plate the ball cannot leave).

  ../.venv-rl/bin/python3 rig_learn.py [--plan odil,sac] [--odil-rounds 3] [--odil-min 30] [--sac-hours 6]
                       [--sac-test-min 30] [--dry]      (--dry: 2-minute phases, for a supervised check)
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import zlib

import numpy as np

# no controller here asks for more than 4 deg (~800 ticks at ~200 ticks/deg); the default cap of
# 1200 ticks (~6 deg) let an untrained PPO ramp motor 1 by 1439 ticks in the dry run (watchdog stop)
os.environ.setdefault("PLATE_CAP_TICKS", "1000")
os.environ.setdefault("PLATE_MARKER_GUARD_M", "0.045")
os.environ.setdefault("PLATE_PLAUSIBLE_OFF_DEG", "4.0")   # tilt reading this far off the command ...
os.environ.setdefault("PLATE_PLAUSIBLE_FRAMES", "10")     # ... for 1/3 s = not trusted (hand in front of the camera)   # ball within 4.5 cm of a plate marker: pose not used
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SIM = os.path.join(ROOT, "rl_sim")
HW = os.path.join(ROOT, "rl_hw")
sys.path.insert(0, HW)                   # the rig: plate_env, hole, rl_policy (shared with parts 1-2)
sys.path.insert(0, SIM)
from plate_env import HardwarePlateEnv                                       # noqa: E402
from hole import detect_holes_stable, near_hole                              # noqa: E402
from rl_policy import ODILFrictionCompRig                                    # noqa: E402
from plate_goal_env import build_obs, ACTION_SCALE, N_HIST, POLICY_RATE, SMOOTH_COEF   # noqa: E402

LOG = os.path.join(ROOT, "phase3_logs", "rig_learn.jsonl")
DATA = os.path.join(HERE, "data")                      # rig recordings + test results (not in git)
LEVEL = (-1.1, 2.55)                     # (alpha, beta) deg, as in rl_policy / sysid_rig
GOAL_X, GOAL_Y, R_TEST = 0.09, 0.07, 0.012
SAC_BUFFER = 1_000_000                     # replay memory (steps); --sac-buffer
RESUME = None                              # (checkpoint .zip, rig minutes already driven); --resume
WATCH_TICKS = 1300
DIP_XY, DIP_R = (0.015, -0.011), 0.02   # the taped-over hole (ball rested at (13..16, -11..-12) mm)
DT = 1.0 / 29.0


def log(rec):
    rec = {"t": time.time(), **rec}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


class Rig:
    """the plate, the safety checks, the data log"""

    def __init__(self):
        if subprocess.run(["pgrep", "-f", "pd_bal" + "ance.py"], capture_output=True).stdout.strip():
            raise SystemExit("pd_balance.py is running -- one controller per U2D2; stop it first")
        self.env = HardwarePlateEnv(fixed_goal=(0.0, 0.0), max_action_delta=0.5,
                                    max_episode_steps=10 ** 9, allow_unstick=False)
        self.env.recover_tilts = False
        self.start_ticks = dict(self.env._cmd_ticks)
        self.first_ticks = dict(self.start_ticks)      # never re-based (absolute bound for re-levels)
        self.n_relevel = 0
        self.holes = detect_holes_stable(self.env)
        # live view for the phone relay (rl_hw/remote_view.py reads :8000): camera image + the target
        from ui_server import UIServer
        self.ui = UIServer(8000)
        self.env.frame_callback = self._on_frame
        self.vid, self.vid_label, self.vid_ov, self.vid_text = None, None, {}, ""
        self.ui.set_state(running=True, mode="click")
        self.csv = None
        self.applied_prev = np.zeros(2)
        self.obs = None
        signal.signal(signal.SIGTERM, lambda *_: self.shutdown("SIGTERM"))

    # ---- safety -------------------------------------------------------------------------
    def relevel(self, why):
        """user-approved 2026-10-03: the commanded ticks wound up (PPO: motors 340-470 ticks short of
        their command, the watchdog tripped after 26 min although the camera showed the plate following):
        re-read where the motors ARE (as at startup), level on the camera, and only a camera-confirmed
        level becomes the new reference. Fails -> stop, torque on."""
        from state_est_control import read_motor_positions, set_position
        from plate_env import LEVEL_OFFSET_DEG
        env = self.env
        self.n_relevel += 1
        log({"event": "relevel", "why": why, "n": self.n_relevel, "cmd_ticks": dict(env._cmd_ticks)})
        # a ball on a plate marker blocks the camera pose: roll it off first (open loop, up to 5 deg)
        t0 = time.time()
        while time.time() - t0 < 12:
            xb, yb, _, _, found = env._read_state()
            if not (found and getattr(env, "ball_at_marker", False)):
                break
            mag = min(1.0, 0.35 + 0.33 * ((time.time() - t0) % 3.0))
            env._write_action((-np.sign([xb, yb]) * mag).astype(np.float32))
            time.sleep(env.dt)
        present = read_motor_positions(env.port_handler, env.packet_handler, [1, 3])
        if any(present.get(k) is None for k in (1, 3)):
            self.shutdown("relevel: could not read the motor positions")
        for k in (1, 3):
            set_position(env.port_handler, env.packet_handler, k, int(present[k]))
            env._cmd_ticks[k] = int(present[k])
        env._session_level = {k: int(present[k]) for k in (1, 3)}
        env._stall, env._tilt_target = {}, LEVEL_OFFSET_DEG
        env._target_hist = [LEVEL_OFFSET_DEG] * 12
        env._last_commanded_action = np.zeros(2, dtype=np.float32)
        time.sleep(0.5)
        env._servo_level(max_s=20.0)
        a, b = env._meas_tilt if env._meas_tilt is not None else (np.nan, np.nan)
        ok = env._pose_ok and abs(a - LEVEL_OFFSET_DEG[0]) < 0.5 and abs(b - LEVEL_OFFSET_DEG[1]) < 0.5
        if not ok:
            self.shutdown(f"relevel failed (alpha {a:+.2f} beta {b:+.2f})")
        if any(abs(env._cmd_ticks[k] - self.first_ticks[k]) > 3000 for k in (1, 3)):
            self.shutdown(f"relevel: level now {dict(env._cmd_ticks)}, > 3000 ticks from the session start {self.first_ticks}")
        self.start_ticks = dict(env._cmd_ticks)
        log({"event": "relevel ok", "n": self.n_relevel, "present": {k: int(present[k]) for k in (1, 3)},
             "level_ticks": dict(env._cmd_ticks), "seconds": round(time.time() - t0, 1)})

    def check(self):
        for k, t0 in self.start_ticks.items():
            if abs(self.env._cmd_ticks.get(k, t0) - t0) > WATCH_TICKS - 300:
                self.relevel(f"motor {k} command {self.env._cmd_ticks.get(k)} within 300 of the +-{WATCH_TICKS} watchdog")
                break
        for k, t0 in self.start_ticks.items():
            if abs(self.env._cmd_ticks.get(k, t0) - t0) > WATCH_TICKS:
                self.shutdown(f"motor {k} command {self.env._cmd_ticks.get(k)} left +-{WATCH_TICKS} of {t0}", release=True)
        if self.env.camera_stale():
            print("camera frozen -- plate held level, waiting", flush=True)
            self.env.hold_session_level()
            while self.env.camera_stale(after_s=0.0) and self.env._grab_frame() is None:
                time.sleep(0.2)
            print("frames back", flush=True)

    def shutdown(self, why, release=False):
        """release=True only for a runaway (watchdog). Otherwise the motors keep holding: releasing
        lets the plate sag under its own weight (-14 deg in the dry run) and it must be re-levelled."""
        print("SHUTDOWN:", why, flush=True)
        try:
            self.stop_csv()
        except Exception:
            pass
        from state_est_control import ADDR_TORQUE_ENABLE
        for k in ((1, 3) if release else ()):
            try:
                self.env.packet_handler.write1ByteTxRx(self.env.port_handler, k, ADDR_TORQUE_ENABLE, 0)
            except Exception:
                pass
        log({"event": "shutdown", "why": why, "released": release})
        os._exit(1)

    # ---- camera video (demo phase) ------------------------------------------------------------
    def _on_frame(self, frame):
        self.ui.publish_frame(frame)
        if self.vid_label is None or frame is None:
            return
        import cv2
        img = frame.copy()
        h, w = img.shape[:2]
        if self.vid is None:
            p = os.path.join(DATA, "..", "report", "video", self.vid_label["file"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            self.vid = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"avc1"), 29, (w, h))
            if not self.vid.isOpened():
                self.vid = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 29, (w, h))
        ov = self.vid_ov
        if ov.get("path_px") is not None:
            cv2.polylines(img, [np.asarray(ov["path_px"], dtype=np.int32)], False, (60, 200, 60), 2, cv2.LINE_AA)
        if ov.get("ref_px") is not None:
            cv2.circle(img, (int(ov["ref_px"][0]), int(ov["ref_px"][1])), 4, (0, 140, 255), -1, cv2.LINE_AA)
        if ov.get("goal_px") is not None:
            c = (int(ov["goal_px"][0]), int(ov["goal_px"][1]))
            cv2.circle(img, c, max(3, int(ov["goal_r_px"])), (60, 200, 60), 2, cv2.LINE_AA)
            cv2.drawMarker(img, c, (60, 200, 60), cv2.MARKER_CROSS, 10, 1)
        col = self.vid_label["color"]
        cv2.rectangle(img, (0, 0), (w, 46), (255, 255, 255), -1)
        sc = 0.9
        while sc > 0.4 and cv2.getTextSize(self.vid_label["title"], cv2.FONT_HERSHEY_SIMPLEX, sc, 2)[0][0] > w - 20:
            sc -= 0.05                                  # the whole method name must fit the frame
        cv2.putText(img, self.vid_label["title"], (10, 32), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, h - 28), (w, h), (255, 255, 255), -1)
        cv2.putText(img, f"{self.vid_label['sub']}   {self.vid_text}", (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (40, 40, 40), 1, cv2.LINE_AA)
        self.vid.write(img)

    def start_video(self, file, title, sub, color):
        self.stop_video()
        self.vid_label = {"file": file, "title": title, "sub": sub, "color": color}

    def stop_video(self):
        self.vid_label = None
        if self.vid is not None:
            self.vid.release()
        self.vid = None

    # ---- one control step ---------------------------------------------------------------
    def step(self, action, tag=""):
        """apply an env action (|a| <= 1, x 5 deg); -> (pos, vel, alpha, beta, found)"""
        self.check()
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        # corner shield (every controller): near a plate marker the tilt reading cannot be trusted
        # (see plate_env MARKER_GUARD_M) -> steer the ball out towards the middle, open loop
        p = getattr(self, "last_pos", None)
        if p is not None and np.hypot(abs(p[0]) - 0.14175, abs(p[1]) - 0.11925) < 0.06:
            # ramp (2026-10-03, PPO): a ball resting in the corner against two walls stayed 263 s at a
            # constant 1.4 deg away-tilt (stiction ~1.6 deg) -> 1.4 deg rising 1 deg/s to 3.2 deg
            # 2026-10-04: ramp ONLY while the ball is stuck (< 2 cm/s): ramping on a rolling ball shot it
            # into the opposite corner and back (shield in control 32 % of PPO's frames)
            lv = getattr(self, "last_vel", None)
            stuck = lv is None or float(np.hypot(*lv)) < 0.02
            self.shield_t = getattr(self, "shield_t", 0.0) + DT if stuck else 0.0
            a = -np.sign(p) * min(0.8, 0.35 + 0.25 * self.shield_t)
            self.shield_n = getattr(self, "shield_n", 0) + 1
        else:
            self.shield_t = 0.0
        # dip twitch (user, 2026-10-04): the taped-over hole near the centre is a small dip; a ball resting in
        # it while the target is elsewhere gets a short twitch towards the target (3.2 deg, 3 frames), at most
        # once a second, the same for every controller; counted in self.twitch_n
        g = getattr(self.env, "goal", None)
        lv = getattr(self, "last_vel", None)
        tw = getattr(self, "twitch_left", 0)
        if tw > 0:
            a = self.twitch_dir * 0.8
            self.twitch_left = tw - 1
        elif (p is not None and g is not None and lv is not None
              and np.hypot(p[0] - DIP_XY[0], p[1] - DIP_XY[1]) < DIP_R
              and np.hypot(*(np.asarray(g, float) - p)) > 0.015 and float(np.hypot(*lv)) < 0.01):
            self.dip_still = getattr(self, "dip_still", 0.0) + DT
            if self.dip_still > 1.0 and time.time() - getattr(self, "twitch_t", 0.0) > 1.0:
                d_ = np.asarray(g, float) - p
                self.twitch_dir = d_ / max(np.hypot(*d_), 1e-6)
                self.twitch_left, self.twitch_t, self.dip_still = 2, time.time(), 0.0
                self.twitch_n = getattr(self, "twitch_n", 0) + 1
                a = self.twitch_dir * 0.8
        else:
            self.dip_still = 0.0
        obs, _, _, _, info = self.env.step(a.astype(np.float32))
        found = bool(info.get("ball_found", info.get("status") != "ball_lost"))
        # a frame or two without the ball (camera hiccup, edge) is not a lost ball: hold the last
        # position; only 10 misses in a row (1/3 s) count as lost (the dry run lost 6 of 6 trips)
        pos_ = np.array(obs[:2], float)
        if found:
            self.miss, self.last_pos = 0, pos_
            self.last_vel = np.array(obs[2:4], float)
        else:
            self.miss = getattr(self, "miss", 0) + 1
            if self.miss < 10 and getattr(self, "last_pos", None) is not None:
                obs = np.array(obs, dtype=np.float32).copy()
                obs[0:2], obs[2:4] = self.last_pos, 0.0
                found = True
        applied = np.asarray(getattr(self.env, "_last_commanded_action", a), dtype=float)
        jerk = float(np.sum((applied - self.applied_prev) ** 2))
        self.applied_prev = applied
        if self.csv is not None:
            self.csv.writerow([f"{time.time() - self.t_csv:.4f}", 1, int(self.k_csv), *(f"{v:.5f}" for v in obs[:6]),
                               f"{a[0]:.4f}", f"{a[1]:.4f}", int(found), "running" if found else "lost", tag])
            self.k_csv += 1
            if self.k_csv % 300 == 0:
                self.fcsv.flush()                     # a hard stop must not lose the last seconds
        return np.array(obs[:2], float), np.array(obs[2:4], float), float(obs[4]), float(obs[5]), found, jerk

    def show(self, goal, r=R_TEST, text=""):
        """draw the current target on the live view (green circle)"""
        from goal_circle import plate_to_pixel
        try:
            a = plate_to_pixel(self.env, np.asarray(goal, float))
            b = plate_to_pixel(self.env, np.asarray(goal, float) + [r, 0.0])
            if a is not None and b is not None:
                self.ui.set_overlay(goal_px=(float(a[1]), float(a[0])),
                                    goal_r_px=float(np.hypot(b[0] - a[0], b[1] - a[1])))
                self.vid_ov = {"goal_px": (float(a[1]), float(a[0])), "goal_r_px": float(np.hypot(b[0] - a[0], b[1] - a[1]))}
            if text:
                self.ui.set_state(hole_msg=text)
                self.vid_text = text
        except Exception:
            pass

    def start_csv(self, name):
        self.stop_csv()
        os.makedirs(DATA, exist_ok=True)
        fn = os.path.join(DATA, f"{name}.csv")
        if os.path.exists(fn):
            # never overwrite a recording (2026-10-04: session 2's "sac1" truncated session 1's sac1.csv)
            fn = os.path.join(DATA, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self.fcsv = open(fn, "w", newline="")
        self.csv = csv.writer(self.fcsv)
        self.csv.writerow(["t", "episode", "step", "xb", "yb", "vx", "vy", "alpha", "beta", "a0", "a1",
                           "ball_found", "status", "tag"])
        self.t_csv, self.k_csv = time.time(), 0
        return fn

    def stop_csv(self):
        if self.csv is not None:
            self.fcsv.close()
            self.csv = None

    # ---- ball lost ----------------------------------------------------------------------
    def reload(self):
        """the ball is not visible (on the plate it cannot leave: lifted, occluded, under the
        camera's blind spot): hold the plate level until it is seen on the board for 10 frames.
        No elevator on the ball-on-plate rig. A long absence is logged once."""
        t0, seen, noted = time.time(), 0, False
        while seen < 10:
            pos, _, _, _, found, _ = self.step(np.zeros(2), "wait_ball")
            # on the plate rig the ball cannot be anywhere else: a ball pressed against the frame sits a
            # little outside the marker-to-marker extents and must still count (it waited forever)
            on = found and abs(pos[0]) < self.env._x_half + 0.02 and abs(pos[1]) < self.env._y_half + 0.02
            seen = seen + 1 if on else 0
            if not noted and time.time() - t0 > 120:
                log({"event": "ball not visible for 2 min -- waiting, plate level"})
                noted = True
        return True

    def level_idle(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.step(np.zeros(2), "idle")


# ---- controllers: (goal, radius, pos, vel, alpha, beta) -> env action -----------------------
class RandomTilts:
    """round-0 excitation without any prior controller: smooth random tilts (Ornstein-Uhlenbeck),
    with a pull back towards the middle near the frame so the ball keeps crossing the plate"""

    def __init__(self, rng):
        self.rng, self.u = rng, np.zeros(2)

    def reset(self):
        self.u = np.zeros(2)

    def __call__(self, goal, radius, pos, vel, alpha, beta):
        # dry run 2026-10-02: +-0.6 (3 deg) raced the ball frame to frame (lost ~10 % of frames,
        # most time against the frame) -> gentler: +-0.35, pull back from 7 cm, brake above 15 cm/s
        self.u += -self.u * DT / 1.5 + 0.2 * np.sqrt(DT) * self.rng.normal(size=2)
        over = np.maximum(np.abs(pos) - np.array([0.07, 0.055]), 0.0)          # how far past 7 / 5.5 cm
        back = np.clip(-np.sign(pos) * 4.0 * over, -0.3, 0.3)
        if np.all(np.abs(pos) > [0.09, 0.07]):
            # a corner: a plate marker sits there and the ball next to it corrupts the tilt reading
            # (session 1: -10 deg, a 4-min runaway) -> leave it decisively, along both axes
            back = -np.sign(pos) * 0.35
            self.u[:] = 0.0
        sp = float(np.hypot(*vel))
        brake = -0.25 * vel / sp * min(1.0, (sp - 0.15) / 0.1) if sp > 0.15 else 0.0
        return np.clip(self.u + back + brake, -0.35, 0.35)


class ODILCtl:
    def __init__(self, path):
        self.c = ODILFrictionCompRig(path)

    def reset(self):
        self.c.reset()

    def __call__(self, goal, radius, pos, vel, alpha, beta):
        return np.asarray(self.c.action(goal, pos, vel, None, radius), dtype=float)


class SACCtl:
    """deterministic SAC or PPO policy with the RL observation and rate limit (as rl_policy.RigPolicyController)"""

    def __init__(self, model):
        self.m, self.hist = model, [np.zeros(2)] * N_HIST

    def reset(self):
        self.hist = [np.zeros(2)] * N_HIST

    def __call__(self, goal, radius, pos, vel, alpha, beta):
        tilt = np.array([np.degrees(beta) - LEVEL[1], -(np.degrees(alpha) - LEVEL[0])])
        a, _ = self.m.predict(build_obs(goal, radius, pos, vel, tilt, self.hist), deterministic=True)
        a = np.clip(np.asarray(a, float) * ACTION_SCALE, self.hist[-1] - POLICY_RATE, self.hist[-1] + POLICY_RATE)
        self.hist = self.hist[1:] + [a.copy()]
        return a


class SimPPOCtl:
    """the simulation-trained PPO v3 (part 1), exactly as the rig runs it (rl_policy.RigPolicyController)"""

    def __init__(self, path):
        from rl_policy import RigPolicyController
        self.c = RigPolicyController(path, LEVEL)

    def reset(self):
        self.c.reset()

    def __call__(self, goal, radius, pos, vel, alpha, beta):
        a = np.asarray(self.c.action(goal, radius, pos, vel, alpha, beta), dtype=float)
        self.c.record_applied(a)
        return a


def references(rig, dry, run="ref"):
    """the part-1 controllers (trained in simulation) on the same targets: anchors every session and
    links 'trained in simulation' to 'trained only on the rig'"""
    n = 6 if dry else 30
    test(rig, ODILCtl(os.path.join(SIM, "runs", "odil_v11", "odil_policy.npz")), "ref_odil_v11_sim", 0, n=n, run=run)
    test(rig, SimPPOCtl(os.path.join(SIM, "runs", "plate_goal_v3", "policy_final.npz")), "ref_ppo_v3_sim", 0, n=n, run=run)


def random_target(rng, rig):
    for _ in range(100):
        g = np.array([rng.uniform(-GOAL_X, GOAL_X), rng.uniform(-GOAL_Y, GOAL_Y)])
        if near_hole(rig.holes, g, extra=0.01) is None:
            return g
    return np.zeros(2)


def drive(rig, ctl, minutes, rng, tag, goal_s=6.0):
    """drive `ctl` to a new random target every goal_s seconds for `minutes` of DRIVING time"""
    driven, t_goal, goal = 0.0, -1e9, np.zeros(2)
    ctl.reset()
    while driven < 60 * minutes:
        t = time.time()
        if t - t_goal > goal_s:
            goal, t_goal = random_target(rng, rig), t
            rig.env.goal = goal.astype(np.float32)
            if not isinstance(ctl, RandomTilts):
                rig.show(goal, R_TEST, tag)
            else:
                rig.ui.set_overlay(goal_px=None, goal_r_px=None)
        if rig.obs is None:
            pos, vel, al, be, found, _ = rig.step(np.zeros(2), tag)
        else:
            pos, vel, al, be = rig.obs
            found = True
        a = ctl(goal, R_TEST, pos, vel, al, be)
        pos, vel, al, be, found, _ = rig.step(a, tag)
        rig.obs = (pos, vel, al, be) if found else None
        driven += time.time() - t
        if not found:
            rig.obs = None
            rig.reload()
            ctl.reset()
    return driven / 60.0


def test(rig, ctl, name, rig_minutes, n=30, seed=2027, run=None, r=R_TEST):
    """the same n targets for every controller -> metrics, logged with the rig minutes; `r` is the
    target radius the controller is told and that counts as reached (12 mm unless a retest)"""
    tag = "" if abs(r - R_TEST) < 1e-9 else f"_r{1000 * r:.0f}mm"
    # the targets are FROZEN (data/test_targets.json, session 1): a hole detected in a later session
    # must not change them (2026-10-03: holes 0 -> 1 would have shifted the random draw)
    fz = os.path.join(DATA, "test_targets.json")
    if os.path.exists(fz) and seed == 2027:
        targets = [np.array(g, dtype=float) for g in json.load(open(fz))["targets"]][:n]
    else:
        rng = np.random.default_rng(seed)
        targets = [random_target(rng, rig) for _ in range(n)]
    R = []
    prev_csv = rig.csv is not None
    if not prev_csv:                                  # tests are recorded too: any tolerance can be
        rig.start_csv(f"test_{run}_{name}_{int(rig_minutes)}min{tag}")   # evaluated afterwards
    for k_, g in enumerate(targets):
        rig.env.goal = g.astype(np.float32)
        rig.show(g, r, f"test {name}{tag}: target {k_ + 1}/{n}")
        ctl.reset()
        t0, t_in, inside, jerks, lost = time.time(), None, [], [], False
        dists, times = [], []
        pos, vel, al, be, found, _ = rig.step(np.zeros(2), "test")
        while True:
            if not found:
                lost = True
                break
            t = time.time() - t0
            if t_in is None and t > 8.0:
                break
            if t_in is not None and t - t_in > 4.0:
                break
            a = ctl(g, r, pos, vel, al, be)
            pos, vel, al, be, found, j = rig.step(a, "test")
            jerks.append(j)
            d = float(np.hypot(*(g - pos)))
            dists.append(d)
            times.append(time.time() - t0)
            if t_in is None and d < r:
                t_in = time.time() - t0
            elif t_in is not None:
                inside.append(d < r)
        R.append({"target": [float(g[0]), float(g[1])], "min_mm": 1000 * float(min(dists)) if dists else None,
                  "t_within": {str(tol): next((tt for tt, dd in zip(times, dists) if dd < tol / 1000), None)
                               for tol in (5, 10, 15, 20, 30)},
                  "reached": t_in is not None, "t_reach": t_in, "lost": lost,
                  "inside_after": float(np.mean(inside)) if inside else 0.0,
                  "final_mm": 1000 * float(np.hypot(*(g - pos))) if found else None,
                  "jerk": float(np.mean(jerks)) if jerks else None})
        if lost:
            rig.reload()
    if not prev_csv:
        rig.stop_csv()
    ok = [r for r in R if r["reached"]]
    rec = {"event": "test", "method": name, "run": run, "radius_mm": round(1000 * r, 1),
           "twitches_total": int(getattr(rig, "twitch_n", 0)), "shield_frames_total": int(getattr(rig, "shield_n", 0)), "rig_minutes": round(rig_minutes, 1), "n": n,
           "reached": len(ok), "lost": sum(r["lost"] for r in R),
           "t_reach_med": float(np.median([r["t_reach"] for r in ok])) if ok else None,
           "inside_after_mean": float(np.mean([r["inside_after"] for r in ok])) if ok else 0.0,
           "final_mm_med": float(np.median([r["final_mm"] for r in R if r["final_mm"] is not None])) if R else None,
           "jerk_med": float(np.median([r["jerk"] for r in R if r["jerk"] is not None])),
           "trips": R}
    log({k: v for k, v in rec.items() if k != "trips"})
    os.makedirs(DATA, exist_ok=True)
    fj = os.path.join(DATA, f"test_{run}_{name}_{int(rig_minutes)}min{tag}.json")
    if os.path.exists(fj):                         # never overwrite a result (see start_csv)
        fj = fj[:-5] + f"_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fj, "w") as f:
        json.dump(rec, f)
    return rec


# ---- ODIL ---------------------------------------------------------------------------------
def odil_rounds(rig, rounds, minutes, rng, dry, run="odil1"):
    files, rig_min, policy = [], 0.0, None
    py = os.path.join(ROOT, ".venv-rl", "bin", "python3")
    for r in range(rounds):
        ctl = RandomTilts(rng) if policy is None else ODILCtl(policy)
        fn = rig.start_csv(f"{run}_r{r}")
        rig_min += drive(rig, ctl, minutes, rng, f"{run}_r{r}")
        rig.stop_csv()
        files.append(fn)
        rig.env._write_action(np.zeros(2, dtype=np.float32))
        # fit the physics to ALL ODIL rig data so far
        t0 = time.time()
        out = subprocess.run([py, "sysid_rig.py", *files], cwd=SIM, capture_output=True, text=True)
        try:
            fit = json.load(open(os.path.join(SIM, "rig_sysid.json")))
        except (OSError, ValueError):
            log({"event": "sysid failed", "stderr": out.stderr[-500:]})
            break
        k, ar = float(fit["k_acc"]), float(fit["a_roll"])
        taus = [fit[f"servo_{a}"]["tau_s"] for a in ("x", "y")]
        brk = fit.get("breakaway_deg", {}).get("median") or 1.8
        env = dict(os.environ, ODIL_THREADS="6", ODIL_N_STAGES="3",
                   ODIL_K_RANGE=f"{0.9 * k:.4f},{1.1 * k:.4f}", ODIL_A_ROLL=f"{ar:.4f}",
                   ODIL_TAU_RANGE=f"{max(0.01, 0.7 * min(taus)):.3f},{max(0.02, 1.3 * max(taus)):.3f}",
                   ODIL_STATIC_RANGE=f"{0.7 * brk:.2f},{1.3 * brk:.2f}")
        name = f"{run}_rig_r{r}" + ("_dry" if dry else "")
        iters = "60" if dry else "1500"
        # train while the plate is held level (a background thread would fight the camera for the CPU)
        proc = subprocess.Popen([py, "odil_plate_v9.py", iters, name], cwd=SIM, env=env,
                                stdout=open(os.path.join(DATA, f"{name}.log"), "w"), stderr=subprocess.STDOUT)
        while proc.poll() is None:
            rig.level_idle(2.0)
        cand = os.path.join(SIM, "runs", name, "odil_policy.npz")
        log({"event": "odil fitted+trained", "run": run, "round": r, "rig_minutes": round(rig_min, 1),
             "k_acc": k, "a_roll": ar, "tau_s": taus, "breakaway_deg": brk,
             "train_minutes": round((time.time() - t0) / 60, 1), "ok": os.path.exists(cand)})
        if not os.path.exists(cand):
            break
        policy = cand
        test(rig, ODILCtl(policy), "odil", rig_min, n=6 if dry else 30, run=run)
    return policy


# ---- SAC ----------------------------------------------------------------------------------
def rl(rig, algo, hours, test_min, rng, dry, run="sac1"):
    """model-free RL on the rig: SAC (off-policy, replays its memory) or PPO (on-policy, the RL
    that was used in simulation). Network updates happen with the plate level."""
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.callbacks import BaseCallback

    EP = 450

    class RigGoalEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9 + 2 * N_HIST,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            self.driven = 0.0

        def reset(self, *, seed=None, options=None):
            if rig.obs is None:
                rig.reload()
            self.goal, self.radius = random_target(rng, rig), float(rng.uniform(0.008, 0.03))
            rig.env.goal = self.goal.astype(np.float32)
            rig.show(self.goal, self.radius, f"{algo} learning: {self.driven / 60:.0f} rig min")
            self.applied, self.hist, self.k = np.zeros(2), [np.zeros(2)] * N_HIST, 0
            pos, vel, al, be, found, _ = rig.step(np.zeros(2), "sac")
            self.state = (pos, vel, al, be)
            rig.obs = self.state if found else None
            return self._obs(), {}

        def _obs(self):
            pos, vel, al, be = self.state
            tilt = np.array([np.degrees(be) - LEVEL[1], -(np.degrees(al) - LEVEL[0])])
            return build_obs(self.goal, self.radius, pos, vel, tilt, self.hist)

        def step(self, action):
            t = time.time()
            a = np.clip(np.asarray(action, float), -1, 1) * ACTION_SCALE
            prev = self.applied
            self.applied = np.clip(a, prev - POLICY_RATE, prev + POLICY_RATE)
            pos, vel, al, be, found, _ = rig.step(self.applied, "sac")
            self.hist = self.hist[1:] + [self.applied.copy()]
            self.k += 1
            self.driven += time.time() - t
            if not found:
                rig.obs = None
                return self._obs(), -5.0, True, False, {}
            self.state = (pos, vel, al, be)
            rig.obs = self.state
            d = float(np.hypot(*(self.goal - pos)))
            inside, still = d < self.radius, np.hypot(*vel) < 0.01
            wall = abs(pos[0]) > 0.13 or abs(pos[1]) > 0.108
            r = (-d / 0.1 + (1.0 if inside else 0.0) + (0.5 if inside and still else 0.0)
                 - SMOOTH_COEF * float(np.sum((self.applied - prev) ** 2))
                 - 0.01 * float(np.sum(self.applied ** 2)) - (0.5 if wall else 0.0))
            return self._obs(), r, False, self.k >= EP, {}

    env = RigGoalEnv()
    seed = int(rng.integers(1 << 30))
    if algo == "sac":
        model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=SAC_BUFFER, batch_size=256,
                    gamma=0.99, tau=0.005, learning_starts=60 if dry else 5_000,
                    train_freq=(1, "episode"), gradient_steps=-1,   # updates between episodes, plate level
                    policy_kwargs=dict(net_arch=[256, 256]), verbose=0, device="cpu", seed=seed)
    else:
        # the simulated-RL settings (rl_sim/train_plate_ppo.py), one rig instead of 8 simulators
        model = PPO("MlpPolicy", env, n_steps=256 if dry else 2048, batch_size=64, n_epochs=10,
                    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    policy_kwargs=dict(net_arch=[256, 256]), verbose=0, device="cpu", seed=seed)
    resumed = False
    if RESUME is not None and os.path.basename(RESUME[0]).startswith(algo):
        # continue the same run from its last checkpoint (policy + optimiser state); rig minutes go on
        model = (SAC if algo == "sac" else PPO).load(RESUME[0], env=env, device="cpu")
        buf = os.path.join(os.path.dirname(RESUME[0]), "sac_buffer_latest.pkl")
        if algo == "sac" and os.path.exists(buf):
            model.load_replay_buffer(buf)
        env.driven = 60.0 * RESUME[1]
        resumed = True
        log({"event": "resume", "run": run, "checkpoint": RESUME[0], "rig_minutes": RESUME[1]})
    out_dir = os.path.join(SIM, "runs", f"{run}_rig" + ("_dry" if dry else ""))
    if not resumed and os.path.isdir(out_dir) and any(n.endswith(".zip") for n in os.listdir(out_dir)):
        raise SystemExit(f"{out_dir} already holds checkpoints -- use another --tag (or --resume); not overwriting")
    os.makedirs(out_dir, exist_ok=True)
    next_test = [env.driven / 60.0 + test_min]

    class Every(BaseCallback):
        def _on_step(self):
            m = env.driven / 60.0
            if m >= next_test[0]:
                rig.env._write_action(np.zeros(2, dtype=np.float32))
                model.save(os.path.join(out_dir, f"{algo}_{int(m)}min"))
                if algo == "sac":   # SAC's memory is not in the .zip: keep it so a resume keeps its experience
                    model.save_replay_buffer(os.path.join(out_dir, "sac_buffer_latest.pkl"))
                test(rig, SACCtl(model), algo, m, n=6 if dry else 30, run=run)
                next_test[0] += test_min
                env.reset()
            return m < 60 * hours

        def _on_rollout_end(self):
            rig.env._write_action(np.zeros(2, dtype=np.float32))      # level while the network updates
            # latest state after every batch (~70 s for PPO): a stop loses at most one batch; resumed by
            # tools/ppo_supervisor.py with --resume <latest> <rig min>
            model.save(os.path.join(out_dir, f"{algo}_latest"))
            with open(os.path.join(out_dir, f"{algo}_latest.json"), "w") as f_:
                json.dump({"rig_minutes": env.driven / 60.0, "t": time.time(), "num_timesteps": int(model.num_timesteps)}, f_)

    model.learn(total_timesteps=10 ** 8, callback=Every(), reset_num_timesteps=not resumed)
    model.save(os.path.join(out_dir, f"{algo}_final"))
    return model


# ---- retest: every saved policy again at a SMALL target ---------------------------------------
def retest(rig, r, dry, max_per_run=12, runs=None, mins=None):
    """the precision question (2026-10-03): ODIL's breakaway boost (stiction compensation) only acts
    while the ball is still and further out than R, so at R = 12 mm a ball stuck ~8 mm off is left
    there (ODIL v9 itself aims at the centre: ODIL_END_FREE = 0); SAC/PPO are pulled to the centre by
    -d/0.1. All take the radius as an input -> tell every saved policy 'the target is r'.
    `runs` (2026-10-05, session 2): only these runs (tagged names like s2c_sac1), no sim references;
    `mins`: only these RL checkpoints (rig minutes). Without them: session 1 as before."""
    import glob
    import re
    from stable_baselines3 import SAC, PPO
    n = 6 if dry else 30
    pre = r"(?:\w+_)?" if runs else ""                 # session 1 names are untagged
    if not runs:
        test(rig, ODILCtl(os.path.join(SIM, "runs", "odil_v11", "odil_policy.npz")), "ref_odil_v11_sim", 0, n=n, run="ref", r=r)
        test(rig, SimPPOCtl(os.path.join(SIM, "runs", "plate_goal_v3", "policy_final.npz")), "ref_ppo_v3_sim", 0, n=n, run="ref", r=r)
    for d in sorted(glob.glob(os.path.join(SIM, "runs", "*odil*_rig_r*"))):
        m = re.match(rf"({pre}odil\d+)_rig_r(\d+)$", os.path.basename(d))
        if m and (not runs or m.group(1) in runs) and os.path.exists(os.path.join(d, "odil_policy.npz")):
            test(rig, ODILCtl(os.path.join(d, "odil_policy.npz")), "odil", 30.0 * (int(m.group(2)) + 1), n=n,
                 run=m.group(1), r=r)
    for d in sorted(glob.glob(os.path.join(SIM, "runs", "*_rig"))):
        m = re.match(rf"({pre}(sac|ppo)\d+)_rig$", os.path.basename(d))
        if not m or (runs and m.group(1) not in runs):
            continue
        ck = sorted(((int(re.search(r"_(\d+)min", p).group(1)), p) for p in glob.glob(os.path.join(d, "*min.zip"))))
        if mins:
            ck = [c for c in ck if c[0] in mins]
        if len(ck) > max_per_run:                      # evenly spread, always the last one
            idx = sorted(set(np.linspace(0, len(ck) - 1, max_per_run).round().astype(int)))
            ck = [ck[i] for i in idx]
        algo = m.group(2)
        for mins, p in ck:
            model = (SAC if algo == "sac" else PPO).load(p, device="cpu")
            test(rig, SACCtl(model), algo, mins, n=n, run=m.group(1), r=r)


# ---- demo: real camera footage of the controllers -----------------------------------------------
def demo(rig, dry, n=10):
    """film each controller through the rig camera on the same first n frozen test targets"""
    from stable_baselines3 import SAC
    n = 3 if dry else n
    BLUE, RED, GREY = (235, 99, 37), (38, 38, 220), (90, 90, 90)
    clips = [("demo_odil_rig_60min.mp4", "ODIL - trained only on the rig (60 rig min)", BLUE,
              lambda: ODILCtl(os.path.join(SIM, "runs", "odil1_rig_r1", "odil_policy.npz")), "odil_rig60"),
             ("demo_sac_rig_120min.mp4", "SAC (RL) - trained only on the rig (120 rig min)", RED,
              lambda: SACCtl(SAC.load(os.path.join(SIM, "runs", "sac1_rig", "sac_120min.zip"), device="cpu")), "sac_rig120"),
             ("demo_odil_v11_sim.mp4", "ODIL v11 - trained in simulation", BLUE,
              lambda: ODILCtl(os.path.join(SIM, "runs", "odil_v11", "odil_policy.npz")), "odil_v11_sim"),
             ("demo_ppo_v3_sim.mp4", "PPO v3 (RL) - trained in simulation", RED,
              lambda: SimPPOCtl(os.path.join(SIM, "runs", "plate_goal_v3", "policy_final.npz")), "ppo_v3_sim")]
    for file, title, col, make, name in clips:
        rig.start_video(file, title, "real rig camera, real time, green circle = target (12 mm)", col)
        try:
            test(rig, make(), name, 0, n=n, run="demo")
        finally:
            rig.stop_video()


# ---- demo: ODIL following a line (real camera footage) -------------------------------------------
def demo_track(rig, dry, shapes_=("circle", "square", "star"), speed=0.03):
    """the part-1 ODIL path tracker (rl_sim/runs/odil_track_best, + closed-loop refinement) draws
    shapes like pd_balance.py's draw mode: PathTracker reference, ODIL action, the same breakaway
    boost along the route; filmed through the rig camera, accuracy logged"""
    sys.path.insert(0, SIM)
    import shapes
    from line_path import PathTracker
    from odil_track import TrackPolicy
    from goal_circle import plate_to_pixel
    track = TrackPolicy(os.path.join(SIM, "runs", "odil_track_best", "odil_track_policy.npz"))
    to_balance = ODILCtl(os.path.join(SIM, "runs", "odil_v11", "odil_policy.npz"))
    BLUE = (235, 99, 37)
    for name in (shapes_[:1] if dry else shapes_):
        path = shapes.resample(shapes.shape(name, (-0.05, 0.02), 0.035))
        px = []
        for p in path:
            q = plate_to_pixel(rig.env, np.asarray(p, float))
            if q is not None:
                px.append((float(q[1]), float(q[0])))
        rig.start_video(f"demo_odil_line_{name}.mp4", f"ODIL path tracker - following a {name}",
                        f"real rig camera, real time, green line = path, orange dot = moving reference ({speed * 100:.0f} cm/s)",
                        BLUE)
        rig.vid_ov = {"path_px": px}
        # bring the ball to the start of the path (ODIL balance controller), up to 12 s
        to_balance.reset()
        pos, vel, al, be, found, _ = rig.step(np.zeros(2), "demo_track")
        t0 = time.time()
        while time.time() - t0 < 12 and found and np.hypot(*(path[0] - pos)) > 0.006:
            pos, vel, al, be, found, _ = rig.step(to_balance(path[0], 0.006, pos, vel, al, be), "demo_track")
        if not found:
            rig.stop_video()
            continue
        follower = PathTracker(path, False, pos, keep_direction=True, v=speed)
        track.reset()
        boost, offs, t_prev, t0 = 0.0, [], time.time(), time.time()
        while time.time() - t0 < 60:
            now = time.time()
            dt = max(now - t_prev, 1e-3)
            t_prev = now
            ref = follower.update(pos, dt)
            if ref.get("finished"):
                break
            if np.hypot(*vel) < 0.01 and ref["lag"] > 0.005:          # pd_balance LINE_BOOST_RATE / _MAX
                boost = min(0.6, boost + 1.0 * dt)
            else:
                boost = max(0.0, boost - 2.0 * dt)
            a = np.clip(np.asarray(track(ref, pos, vel, dt), float) + boost * ref["t_hat"], -0.8, 0.8)
            q = plate_to_pixel(rig.env, np.asarray(ref["p"], float))
            if q is not None:
                rig.vid_ov["ref_px"] = (float(q[1]), float(q[0]))
            pos, vel, al, be, found, _ = rig.step(a, "demo_track")
            if not found:
                break
            offs.append(1000 * float(np.min(np.hypot(*(path - pos).T))))
        # hold the end for 2 s so the clip ends calmly
        t1 = time.time()
        while time.time() - t1 < 2 and found:
            pos, vel, al, be, found, _ = rig.step(to_balance(path[-1], 0.012, pos, vel, al, be), "demo_track")
        rig.stop_video()
        rig.vid_ov = {}
        if offs:
            log({"event": "track demo", "shape": name, "speed_mm_s": 1000 * speed, "seconds": round(time.time() - t0, 1),
                 "median_mm": round(float(np.median(offs)), 2), "p90_mm": round(float(np.percentile(offs, 90)), 2),
                 "max_mm": round(float(np.max(offs)), 2)})


# ---- shape test: trained controllers follow 5 shapes (real camera footage) -----------------------
SHAPES = ("star", "heart", "circle", "square", "figure8")       # figure8 = the infinity symbol


def shape_test(rig, dry, reps=3, speed=0.03, radius=0.008):
    """every method's best rig-trained controller (best 12-mm test: most reached, then inside) follows the
    same moving reference along 5 shapes, like pd_balance's draw mode. Goal-reaching controllers (ODIL, SAC,
    PPO) get the moving reference point as their goal; the ODIL path tracker trained only from the ODIL rig
    data (reuse, 0 extra rig minutes) gets the reference directly. Interleaved: rep -> shape -> controller."""
    sys.path.insert(0, SIM)
    import shapes
    from line_path import PathTracker
    from odil_track import TrackPolicy
    from goal_circle import plate_to_pixel
    from stable_baselines3 import SAC, PPO
    R = os.path.join(SIM, "runs")
    ctls = [("odil_rig60", "ODIL (goal-reaching) - trained only on the rig, 60 rig min", (235, 99, 37),
             ODILCtl(os.path.join(R, "odil1_rig_r1", "odil_policy.npz"))),
            ("sac_rig330", "SAC (RL) - trained only on the rig, 330 rig min", (38, 38, 220),
             SACCtl(SAC.load(os.path.join(R, "sac1_rig", "sac_330min.zip"), device="cpu"))),
            ("ppo_rig600", "PPO (RL) - trained only on the rig, 600 rig min", (38, 38, 220),
             SACCtl(PPO.load(os.path.join(R, "ppo1_rig", "ppo_600min.zip"), device="cpu"))),
            ("odil_tracker_reuse", "ODIL path tracker - from the ODIL rig data only (0 extra rig min)", (180, 60, 20),
             TrackPolicy(os.path.join(R, "reuse_odil1_track", "odil_track_policy.npz")))]
    approach = ODILCtl(os.path.join(R, "odil1_rig_r1", "odil_policy.npz"))   # not used for the tracker's approach
    for rep in range(1 if dry else reps):
        for name in (SHAPES[:1] if dry else SHAPES):
            path = shapes.resample(shapes.shape(name, (-0.05, 0.02), 0.035))
            px = []
            for p_ in path:
                q = plate_to_pixel(rig.env, np.asarray(p_, float))
                if q is not None:
                    px.append((float(q[1]), float(q[0])))
            for tag, title, col, ctl in ctls:
                film = rep == 0
                if film:
                    rig.start_video(f"shape_{name}_{tag}.mp4", title,
                                    f"real rig camera, real time | {name} | green = path, orange = moving reference 3 cm/s", col)
                rig.vid_ov = {"path_px": px}
                is_tracker = isinstance(ctl, TrackPolicy)
                ctl.reset()
                reacher = approach if is_tracker else ctl
                reacher.reset()
                pos, vel, al, be, found, _ = rig.step(np.zeros(2), "shape")
                t0 = time.time()
                while time.time() - t0 < 12 and found and np.hypot(*(path[0] - pos)) > 0.008:   # go to the start
                    pos, vel, al, be, found, _ = rig.step(reacher(path[0], radius, pos, vel, al, be), "shape")
                ctl.reset()
                follower = PathTracker(path, False, pos, keep_direction=True, v=speed)
                boost, offs, jerks, t_prev, t1, ok = 0.0, [], [], time.time(), time.time(), False
                while found and time.time() - t1 < 60:
                    now = time.time()
                    dt = max(now - t_prev, 1e-3)
                    t_prev = now
                    ref = follower.update(pos, dt)
                    if ref.get("finished"):
                        ok = True
                        break
                    if is_tracker:
                        if np.hypot(*vel) < 0.01 and ref["lag"] > 0.005:
                            boost = min(0.6, boost + 1.0 * dt)
                        else:
                            boost = max(0.0, boost - 2.0 * dt)
                        a = np.clip(np.asarray(ctl(ref, pos, vel, dt), float) + boost * ref["t_hat"], -0.8, 0.8)
                    else:
                        a = ctl(np.asarray(ref["p"], float), radius, pos, vel, al, be)
                    q = plate_to_pixel(rig.env, np.asarray(ref["p"], float))
                    if q is not None:
                        rig.vid_ov["ref_px"] = (float(q[1]), float(q[0]))
                    pos, vel, al, be, found, j = rig.step(a, "shape")
                    jerks.append(j)
                    offs.append(1000 * float(np.min(np.hypot(*(path - pos).T))))
                prog = float(min(follower.s, follower.L) / follower.L)
                if film:
                    t2 = time.time()
                    while time.time() - t2 < 1.5:
                        pos, vel, al, be, found, _ = rig.step(np.zeros(2), "shape")
                    rig.stop_video()
                rig.vid_ov = {}
                log({"event": "shape test", "rep": rep + 1, "shape": name, "controller": tag, "finished": ok,
                     "progress": round(prog, 3), "seconds": round(time.time() - t1, 1), "found": bool(found),
                     "median_mm": round(float(np.median(offs)), 2) if offs else None,
                     "p90_mm": round(float(np.percentile(offs, 90)), 2) if offs else None,
                     "max_mm": round(float(np.max(offs)), 2) if offs else None,
                     "jerk": round(float(np.mean(jerks)), 5) if jerks else None})
                if not found:
                    rig.reload()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="ref,odil,sac")
    ap.add_argument("--ppo-hours", type=float, default=30)
    ap.add_argument("--ppo-test-min", type=float, default=60)
    ap.add_argument("--odil-rounds", type=int, default=3)
    ap.add_argument("--odil-min", type=float, default=30)
    ap.add_argument("--sac-hours", type=float, default=6)
    ap.add_argument("--sac-test-min", type=float, default=30)
    ap.add_argument("--retest-mm", type=float, default=5)
    ap.add_argument("--retest-runs", default="", help="only these runs, e.g. s2c_sac1,s2c_odil1")
    ap.add_argument("--retest-min", default="", help="only these RL checkpoints (rig min), e.g. 60,120")
    # session 1 used 300k (= ~172 min of driving, from rl_sim/train_sac.py) and declined after ~240 min;
    # 1M is the stable-baselines3 default (~575 min) -> tests whether FIFO forgetting caused it
    ap.add_argument("--sac-buffer", type=int, default=1_000_000)
    ap.add_argument("--tag", default="", help="session label prefixed to every run name, e.g. s2 -> s2_sac1")
    ap.add_argument("--resume", nargs=2, metavar=("CHECKPOINT", "RIG_MIN"), help="continue an RL run")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    global SAC_BUFFER
    SAC_BUFFER = a.sac_buffer
    global RESUME
    RESUME = (a.resume[0], float(a.resume[1])) if a.resume else None
    if a.dry:
        a.odil_rounds, a.odil_min, a.sac_hours, a.sac_test_min = 1, 2, 4 / 60, 2
        a.ppo_hours, a.ppo_test_min = 4 / 60, 2
    os.makedirs(DATA, exist_ok=True)
    rig = Rig()
    log({"event": "start", "plan": a.plan, "dry": a.dry, "sac_buffer": SAC_BUFFER, "start_ticks": rig.start_ticks, "holes": len(rig.holes)})
    count = {}
    try:
        for phase in a.plan.split(","):
            count[phase] = count.get(phase, 0) + 1
            run = (f"{a.tag}_" if a.tag else "") + f"{phase}{count[phase]}" + ("_dry" if a.dry else "")
            rng = np.random.default_rng(zlib.crc32(run.encode()))       # each run its own, reproducible seed
            log({"event": "phase", "run": run})
            if phase in ("ref", "sac", "ppo"):
                rig.start_csv(run)                     # everything recorded (ODIL starts its own per round)
            if phase == "ref":
                references(rig, a.dry, run)
            elif phase == "odil":
                rig.stop_csv()
                odil_rounds(rig, a.odil_rounds, a.odil_min, rng, a.dry, run)
            elif phase == "shapes":
                rig.start_csv(run)
                shape_test(rig, a.dry)
            elif phase == "demo_track":
                demo_track(rig, a.dry)
            elif phase == "demo":
                demo(rig, a.dry)
            elif phase == "retest":
                retest(rig, a.retest_mm / 1000.0, a.dry,
                       runs=[s for s in a.retest_runs.split(",") if s] or None,
                       mins=[int(s) for s in a.retest_min.split(",") if s] or None)
            elif phase in ("sac", "ppo"):
                hours = a.sac_hours if phase == "sac" else a.ppo_hours
                rl(rig, phase, hours, a.sac_test_min if phase == "sac" else a.ppo_test_min, rng, a.dry, run)
        rig.stop_csv()
        log({"event": "done"})
        rig.shutdown("finished")
    except BaseException as e:
        import traceback
        traceback.print_exc()
        log({"event": "error", "error": repr(e), "trace": traceback.format_exc()[-1500:]})
        rig.shutdown(f"error: {e!r}")


if __name__ == "__main__":
    main()
