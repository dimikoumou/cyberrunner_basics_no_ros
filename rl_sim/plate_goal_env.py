"""
Goal-reaching gym environment on top of plate_sim.PlateSim (2026-09-27): the policy
sees what the real controller sees (camera-measured, delayed, noisy) and outputs a
tilt command; it must bring the ball into a target circle anywhere in the open area
and keep it there. Observation/action conventions match rl_hw (deployment wrapper
in rl_hw/rl_policy.py builds the same observation from the real env).

Observation (17):
  [ (goal - ball)/0.1 (2), vel/0.1 (2), ball/0.14 (2), measured tilt/5deg (2),
    goal radius/0.03 (1), last 4 applied actions (8) ]
Action (2): in [-1, 1]; scaled by ACTION_SCALE (the rig's controllers cap at 0.8)
            -> env rate limit 0.5/step -> target tilt = 5 deg * applied action.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from plate_sim import PlateSim, DT_NOM

ACTION_SCALE = 0.8
MAX_ACTION_DELTA = 0.5
GOAL_X, GOAL_Y = 0.09, 0.07          # open area (>= ~5 cm from the frame)
START_X, START_Y = 0.12, 0.10
R_RANGE = (0.008, 0.03)
EP_STEPS = 450                        # ~15 s
N_HIST = 4


def build_obs(goal, radius, pos, vel, tilt_deg, act_hist):
    return np.concatenate([(np.asarray(goal) - pos) / 0.1, np.asarray(vel) / 0.1, np.asarray(pos) / 0.14,
                           np.asarray(tilt_deg) / 5.0, [radius / 0.03],
                           np.asarray(act_hist, dtype=float).ravel()]).astype(np.float32)


class PlateGoalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, randomize=True, seed=None):
        super().__init__()
        self.randomize = randomize
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9 + 2 * N_HIST,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def _new_goal(self):
        self.goal = np.array([self.rng.uniform(-GOAL_X, GOAL_X), self.rng.uniform(-GOAL_Y, GOAL_Y)])
        self.radius = float(self.rng.uniform(*R_RANGE))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.sim = PlateSim(self.rng, randomize=self.randomize)
        start = np.array([self.rng.uniform(-START_X, START_X), self.rng.uniform(-START_Y, START_Y)])
        self.sim.reset(start)
        self._new_goal()
        self.switch_at = int(self.rng.integers(150, 350)) if self.rng.random() < 0.5 else -1
        self.applied = np.zeros(2)
        self.hist = [np.zeros(2)] * N_HIST
        self.k = 0
        self.pos, self.vel, self.tilt = start.copy(), np.zeros(2), np.zeros(2)
        return self._obs(), {}

    def _obs(self):
        return build_obs(self.goal, self.radius, self.pos, self.vel, self.tilt, self.hist)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1) * ACTION_SCALE
        prev = self.applied
        self.applied = np.clip(a, prev - MAX_ACTION_DELTA, prev + MAX_ACTION_DELTA)
        dt = float(np.clip(self.rng.normal(DT_NOM, 0.003), 0.028, 0.045))
        pos, vel, tilt, found = self.sim.step(5.0 * self.applied, dt=dt)
        self.pos, self.vel, self.tilt = pos, vel, tilt
        self.hist = self.hist[1:] + [self.applied.copy()]
        self.k += 1
        if self.k == self.switch_at:
            self._new_goal()
        true_pos, true_vel = self.sim.pos, self.sim.vel
        d = float(np.hypot(*(self.goal - true_pos)))
        inside = d < self.radius
        still = np.hypot(*true_vel) < 0.01
        wall = abs(true_pos[0]) > 0.13 or abs(true_pos[1]) > 0.108
        r = (-d / 0.1 + (1.0 if inside else 0.0) + (0.5 if inside and still else 0.0)
             - 0.05 * float(np.sum((self.applied - prev) ** 2)) - 0.01 * float(np.sum(self.applied ** 2))
             - (0.5 if wall else 0.0))
        truncated = self.k >= EP_STEPS
        return self._obs(), r, False, truncated, {"dist": d, "inside": inside, "still": still}
