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
WATCH_TICKS = 1300
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
        self.holes = detect_holes_stable(self.env)
        # live view for the phone relay (rl_hw/remote_view.py reads :8000): camera image + the target
        from ui_server import UIServer
        self.ui = UIServer(8000)
        self.env.frame_callback = self.ui.publish_frame
        self.ui.set_state(running=True, mode="click")
        self.csv = None
        self.applied_prev = np.zeros(2)
        self.obs = None
        signal.signal(signal.SIGTERM, lambda *_: self.shutdown("SIGTERM"))

    # ---- safety -------------------------------------------------------------------------
    def check(self):
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

    # ---- one control step ---------------------------------------------------------------
    def step(self, action, tag=""):
        """apply an env action (|a| <= 1, x 5 deg); -> (pos, vel, alpha, beta, found)"""
        self.check()
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        obs, _, _, _, info = self.env.step(a.astype(np.float32))
        found = bool(info.get("ball_found", info.get("status") != "ball_lost"))
        # a frame or two without the ball (camera hiccup, edge) is not a lost ball: hold the last
        # position; only 10 misses in a row (1/3 s) count as lost (the dry run lost 6 of 6 trips)
        pos_ = np.array(obs[:2], float)
        if found:
            self.miss, self.last_pos = 0, pos_
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
            if text:
                self.ui.set_state(hole_msg=text)
        except Exception:
            pass

    def start_csv(self, name):
        self.stop_csv()
        os.makedirs(DATA, exist_ok=True)
        fn = os.path.join(DATA, f"{name}.csv")
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


def references(rig, dry):
    """the part-1 controllers (trained in simulation) on the same targets: anchors every session and
    links 'trained in simulation' to 'trained only on the rig'"""
    n = 6 if dry else 30
    test(rig, ODILCtl(os.path.join(SIM, "runs", "odil_v11", "odil_policy.npz")), "ref_odil_v11_sim", 0, n=n, run="ref")
    test(rig, SimPPOCtl(os.path.join(SIM, "runs", "plate_goal_v3", "policy_final.npz")), "ref_ppo_v3_sim", 0, n=n, run="ref")


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


def test(rig, ctl, name, rig_minutes, n=30, seed=2027, run=None):
    """the same n targets for every controller -> metrics, logged with the rig minutes"""
    rng = np.random.default_rng(seed)
    targets = [random_target(rng, rig) for _ in range(n)]
    R = []
    prev_csv = rig.csv is not None
    if not prev_csv:                                  # tests are recorded too: any tolerance can be
        rig.start_csv(f"test_{run}_{name}_{int(rig_minutes)}min")   # evaluated afterwards
    for k_, g in enumerate(targets):
        rig.env.goal = g.astype(np.float32)
        rig.show(g, R_TEST, f"test {name}: target {k_ + 1}/{n}")
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
            a = ctl(g, R_TEST, pos, vel, al, be)
            pos, vel, al, be, found, j = rig.step(a, "test")
            jerks.append(j)
            d = float(np.hypot(*(g - pos)))
            dists.append(d)
            times.append(time.time() - t0)
            if t_in is None and d < R_TEST:
                t_in = time.time() - t0
            elif t_in is not None:
                inside.append(d < R_TEST)
        R.append({"target": [float(g[0]), float(g[1])], "min_mm": 1000 * float(min(dists)) if dists else None,
                  "t_within": {str(tol): next((tt for tt, dd in zip(times, dists) if dd < tol / 1000), None)
                               for tol in (10, 15, 20, 30)},
                  "reached": t_in is not None, "t_reach": t_in, "lost": lost,
                  "inside_after": float(np.mean(inside)) if inside else 0.0,
                  "final_mm": 1000 * float(np.hypot(*(g - pos))) if found else None,
                  "jerk": float(np.mean(jerks)) if jerks else None})
        if lost:
            rig.reload()
    if not prev_csv:
        rig.stop_csv()
    ok = [r for r in R if r["reached"]]
    rec = {"event": "test", "method": name, "run": run, "rig_minutes": round(rig_minutes, 1), "n": n,
           "reached": len(ok), "lost": sum(r["lost"] for r in R),
           "t_reach_med": float(np.median([r["t_reach"] for r in ok])) if ok else None,
           "inside_after_mean": float(np.mean([r["inside_after"] for r in ok])) if ok else 0.0,
           "final_mm_med": float(np.median([r["final_mm"] for r in R if r["final_mm"] is not None])) if R else None,
           "jerk_med": float(np.median([r["jerk"] for r in R if r["jerk"] is not None])),
           "trips": R}
    log({k: v for k, v in rec.items() if k != "trips"})
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, f"test_{run}_{name}_{int(rig_minutes)}min.json"), "w") as f:
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
        model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=300_000, batch_size=256,
                    gamma=0.99, tau=0.005, learning_starts=60 if dry else 5_000,
                    train_freq=(1, "episode"), gradient_steps=-1,   # updates between episodes, plate level
                    policy_kwargs=dict(net_arch=[256, 256]), verbose=0, device="cpu", seed=seed)
    else:
        # the simulated-RL settings (rl_sim/train_plate_ppo.py), one rig instead of 8 simulators
        model = PPO("MlpPolicy", env, n_steps=256 if dry else 2048, batch_size=64, n_epochs=10,
                    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    policy_kwargs=dict(net_arch=[256, 256]), verbose=0, device="cpu", seed=seed)
    out_dir = os.path.join(SIM, "runs", f"{run}_rig" + ("_dry" if dry else ""))
    os.makedirs(out_dir, exist_ok=True)
    next_test = [test_min]

    class Every(BaseCallback):
        def _on_step(self):
            m = env.driven / 60.0
            if m >= next_test[0]:
                rig.env._write_action(np.zeros(2, dtype=np.float32))
                model.save(os.path.join(out_dir, f"{algo}_{int(m)}min"))
                test(rig, SACCtl(model), algo, m, n=6 if dry else 30, run=run)
                next_test[0] += test_min
                env.reset()
            return m < 60 * hours

        def _on_rollout_end(self):
            rig.env._write_action(np.zeros(2, dtype=np.float32))      # level while the network updates

    model.learn(total_timesteps=10 ** 8, callback=Every())
    model.save(os.path.join(out_dir, f"{algo}_final"))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="ref,odil,sac")
    ap.add_argument("--ppo-hours", type=float, default=30)
    ap.add_argument("--ppo-test-min", type=float, default=60)
    ap.add_argument("--odil-rounds", type=int, default=3)
    ap.add_argument("--odil-min", type=float, default=30)
    ap.add_argument("--sac-hours", type=float, default=6)
    ap.add_argument("--sac-test-min", type=float, default=30)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.dry:
        a.odil_rounds, a.odil_min, a.sac_hours, a.sac_test_min = 1, 2, 4 / 60, 2
        a.ppo_hours, a.ppo_test_min = 4 / 60, 2
    os.makedirs(DATA, exist_ok=True)
    rig = Rig()
    log({"event": "start", "plan": a.plan, "dry": a.dry, "start_ticks": rig.start_ticks, "holes": len(rig.holes)})
    count = {}
    try:
        for phase in a.plan.split(","):
            count[phase] = count.get(phase, 0) + 1
            run = f"{phase}{count[phase]}" + ("_dry" if a.dry else "")
            rng = np.random.default_rng(zlib.crc32(run.encode()))       # each run its own, reproducible seed
            log({"event": "phase", "run": run})
            if phase in ("ref", "sac", "ppo"):
                rig.start_csv(run)                     # everything recorded (ODIL starts its own per round)
            if phase == "ref":
                references(rig, a.dry)
            elif phase == "odil":
                rig.stop_csv()
                odil_rounds(rig, a.odil_rounds, a.odil_min, rng, a.dry, run)
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
