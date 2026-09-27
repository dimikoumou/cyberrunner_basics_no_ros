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
from plate_goal_env import build_obs, ACTION_SCALE, N_HIST  # noqa: E402


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
        return self.policy(obs) * ACTION_SCALE      # what goes to env.step()

    def record_applied(self, applied):
        """call after env.step() with env._last_commanded_action"""
        self.hist = self.hist[1:] + [np.asarray(applied, dtype=float).copy()]
