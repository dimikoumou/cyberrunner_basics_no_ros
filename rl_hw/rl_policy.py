"""
Run a policy trained in rl_sim (train_plate_ppo.py) on the real rig (2026-09-27).

Loads the exported numpy weights (runs/<name>/policy.npz: tanh MLP + linear head),
so the rig needs no PyTorch, and builds exactly the simulator's observation from
the real env's measurements (see rl_sim/plate_goal_env.py):
  pos, vel      -> ball position/velocity from the camera (m, m/s)
  tilt          -> camera-measured plate tilt in the simulator's axes, level offset
                   removed: x <- beta - beta0, y <- -(alpha - alpha0)   (deg)
  act history   -> the last N applied actions (after the env's rate limit)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rl_sim")))
from plate_goal_env import build_obs, ACTION_SCALE, N_HIST, POLICY_RATE  # noqa: E402


class NumpyPolicy:
    def __init__(self, path):
        w = np.load(path)
        n = int(w["n_hidden"])
        self.layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(n)]
        self.out = (w["Wout"], w["bout"])
        self.path = path

    def __call__(self, obs):
        h = np.asarray(obs, dtype=np.float64)
        for W, b in self.layers:
            h = np.tanh(W @ h + b)
        return np.clip(self.out[0] @ h + self.out[1], -1.0, 1.0)


class RigPolicyController:
    def __init__(self, path, level_offset_deg):
        self.policy = NumpyPolicy(path)
        self.a0, self.b0 = level_offset_deg          # (alpha, beta) offset in deg
        self.hist = [np.zeros(2)] * N_HIST

    def reset(self):
        self.hist = [np.zeros(2)] * N_HIST

    def action(self, goal, radius, pos, vel, alpha_rad, beta_rad):
        tilt = np.array([np.degrees(beta_rad) - self.b0, -(np.degrees(alpha_rad) - self.a0)])
        obs = build_obs(goal, radius, np.asarray(pos), np.asarray(vel), tilt, self.hist)
        a = self.policy(obs) * ACTION_SCALE
        prev = self.hist[-1]
        return np.clip(a, prev - POLICY_RATE, prev + POLICY_RATE)   # same rate limit as in training

    def record_applied(self, applied):
        """call after env.step() with env._last_commanded_action"""
        self.hist = self.hist[1:] + [np.asarray(applied, dtype=float).copy()]


class ODILRigController:
    """ODIL policy (rl_sim/odil_plate.py) on the rig: features [(goal-p)/0.1, v/0.1,
    th/5, th_lag/5] where th/th_lag are the model's tilt states, advanced by an observer
    from the commands actually applied (same as rl_sim/eval_controllers.ODILController).
    Output: commanded tilt in deg (|u| <= 4 deg) -> env action = u/5."""
    TAU, DT = 0.045, 1.0 / 29.0

    def __init__(self, path):
        w = np.load(path)
        n = sum(1 for k in w.files if k.startswith("W") and k[1:].isdigit())
        self.layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(n)]
        self.path = path
        # v5 (odil_plate_v5.py): 10 inputs, the last two a leaky integral of the goal error
        self.n_in = int(w["n_in"]) if "n_in" in w.files else 8
        self.tz = float(w["tz"]) if "tz" in w.files else 1.5
        self.z_scale = float(w["z_scale"]) if "z_scale" in w.files else 0.02
        self._t_prev = None
        self.reset()

    def reset(self):
        self.th = np.zeros(2)
        self.thl = np.zeros(2)
        self.z = np.zeros(2)
        self._t_prev = None

    def action(self, goal, pos, vel, dt=None, radius=0.012):
        rel = np.asarray(goal) - np.asarray(pos)
        feat = np.concatenate([rel / 0.1, np.asarray(vel) / 0.1, self.th / 5.0, self.thl / 5.0])
        if self.n_in >= 10:
            self.z += (rel / self.z_scale - self.z / self.tz) * (dt or self.DT)
            feat = np.concatenate([feat, self.z])
        if self.n_in == 11:                              # v7: target radius
            feat = np.concatenate([feat, [radius / 0.03]])
        h = feat
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        u_deg = 5.0 * 0.8 * np.tanh(h)
        return u_deg / 5.0

    def record_applied(self, applied, dt=None):
        dt = dt or self.DT
        u_deg = 5.0 * np.asarray(applied, dtype=float)
        self.th += (u_deg - self.th) * min(1.0, dt / self.TAU)
        self.thl += (self.th - self.thl) * min(1.0, dt / self.TAU)


class ODILFrictionCompRig(ODILRigController):
    """ODIL + stiction compensation on the rig (rl_sim/odil_friction_comp.py): when the
    ball has stayed within 1.5 mm for ~0.4 s outside the target, a breakaway boost along
    the goal direction ramps up (6 deg/s, max 2 deg) until it moves, then is dropped."""
    STILL_PTP, STILL_N, RAMP, BOOST_MAX, OUTSIDE = 0.0015, 12, 6.0, 2.0, 1.0

    def reset(self):
        super().reset()
        self.hist, self.boost = [], 0.0

    def action(self, goal, pos, vel, dt=None, radius=0.012):
        a = super().action(goal, pos, vel, dt, radius)
        rel = np.asarray(goal, dtype=float) - np.asarray(pos, dtype=float)
        d = float(np.hypot(*rel))
        self.hist = (self.hist + [np.asarray(pos, dtype=float).copy()])[-self.STILL_N:]
        H = np.array(self.hist)
        still = len(H) == self.STILL_N and np.ptp(H[:, 0]) < self.STILL_PTP and np.ptp(H[:, 1]) < self.STILL_PTP
        if still and d > self.OUTSIDE * radius:
            self.boost = min(self.BOOST_MAX, self.boost + self.RAMP * (dt or self.DT))
        elif not still:
            self.boost = 0.0
        if self.boost > 0:
            a = np.clip(a + (rel / max(d, 1e-6)) * self.boost / 5.0, -1.0, 1.0)
        return a
