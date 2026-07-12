"""
Gymnasium environment wrapping the maze physics.

Interface is chosen to match the REAL system so a policy transfers to hardware:
    observation (8,), all normalized to ~[-1, 1]:
        [ x, y,                      # ball position (from state estimator)
          vx, vy,                    # ball velocity (finite-diff of position)
          alpha, beta,               # current plate angles (motor feedback)
          dx_next, dy_next ]         # vector to next path checkpoint (path is known)
    action (2,) in [-1, 1]:
        [ target_alpha, target_beta ] -> plate tilt commands (map to motors on HW)

Reward is potential-based on distance-to-go ALONG the path, plus checkpoint/goal
bonuses, a hole penalty, and a small step penalty (encourages finishing quickly).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from maze_sim import MazeSim
from maze_layout import default_maze

V_NORM = 1.0  # nominal max speed for normalizing velocity (m/s)


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, cfg=None, fps=60, max_steps=800, render_mode=None):
        super().__init__()
        self.cfg = cfg if cfg is not None else default_maze()
        self.sim = MazeSim(self.cfg, fps=fps)
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        # precompute path segment lengths for the distance-to-go potential
        cps = self.cfg.checkpoints
        self._seg_len = np.linalg.norm(np.diff(cps, axis=0), axis=1) if len(cps) > 1 else np.zeros(0)
        self.progress_scale = 20.0
        self._steps = 0
        self._prev_potential = 0.0

    # ---- helpers --------------------------------------------------------
    def _path_remaining(self):
        """Distance from the ball to the goal measured along the remaining path."""
        idx = self.sim.checkpoint_idx
        d = self.sim.dist_to_next_checkpoint()
        if idx < len(self._seg_len):
            d += self._seg_len[idx:].sum()
        return d

    def _obs(self):
        s = self.sim.state()
        pos, vel, ang = s["pos"], s["vel"], s["angles"]
        nxt = self.cfg.checkpoints[self.sim.checkpoint_idx]
        d = nxt - pos
        return np.array([
            pos[0] / self.cfg.hx, pos[1] / self.cfg.hy,
            vel[0] / V_NORM,      vel[1] / V_NORM,
            ang[0] / self.cfg.max_tilt, ang[1] / self.cfg.max_tilt,
            d[0] / (2 * self.cfg.hx),   d[1] / (2 * self.cfg.hy),
        ], dtype=np.float32)

    # ---- gym API --------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # small random start jitter for robustness
        jitter = self.np_random.uniform(-0.005, 0.005, size=2)
        self.sim.reset(start=self.cfg.start + jitter)
        self._steps = 0
        self._max_cp = 0
        self._prev_potential = -self._path_remaining()
        return self._obs(), {}

    def step(self, action):
        cp_before = self.sim.checkpoint_idx
        self.sim.step(action)
        self._steps += 1

        potential = -self._path_remaining()
        reward = self.progress_scale * (potential - self._prev_potential)
        self._prev_potential = potential
        reward -= 0.01                                   # step penalty (finish faster)
        reward -= 0.002 * float(np.sum(np.square(action)))  # mild control effort penalty

        # explicit bonus each time a new checkpoint is reached (discrete signal on
        # top of the dense shaping)
        cp_after = self.sim.checkpoint_idx
        reward += 3.0 * (cp_after - cp_before)
        self._max_cp = max(self._max_cp, cp_after)

        terminated = False
        status = self.sim.status
        if status == "goal":
            reward += 20.0
            terminated = True
        elif status == "hole":
            reward -= 5.0        # softer than before: don't make stalling the safe choice
            terminated = True

        truncated = self._steps >= self.max_steps
        info = {"status": status, "checkpoint_idx": cp_after, "max_cp": self._max_cp}
        return self._obs(), float(reward), terminated, truncated, info

    # ---- rendering ------------------------------------------------------
    def render(self):
        return self._render_frame()

    def _render_frame(self, px_per_m=2000):
        import cv2
        cfg = self.cfg
        W = int(2 * cfg.hx * px_per_m) + 40
        H = int(2 * cfg.hy * px_per_m) + 40
        img = np.full((H, W, 3), 245, np.uint8)

        def to_px(p):
            x = int((p[0] + cfg.hx) * px_per_m) + 20
            y = int((cfg.hy - p[1]) * px_per_m) + 20
            return (x, y)

        # checkpoints / path (faint)
        cps = cfg.checkpoints
        for i in range(len(cps) - 1):
            cv2.line(img, to_px(cps[i]), to_px(cps[i + 1]), (220, 220, 220), 1)
        # walls
        for w in self.sim.walls:
            cv2.line(img, to_px(w[:2]), to_px(w[2:]), (60, 60, 60), 3)
        # holes
        for h in cfg.holes:
            cv2.circle(img, to_px(h), int(cfg.hole_radius * px_per_m), (30, 30, 30), -1)
        # goal
        cv2.circle(img, to_px(cps[-1]), int(cfg.goal_radius * px_per_m), (0, 170, 0), 2)
        # next checkpoint marker
        cv2.circle(img, to_px(cps[self.sim.checkpoint_idx]),
                   int(cfg.checkpoint_radius * px_per_m), (255, 180, 0), 1)
        # ball
        cv2.circle(img, to_px(self.sim.pos), int(cfg.ball_radius * px_per_m), (0, 0, 220), -1)
        # HUD
        a, b = np.degrees(self.sim.angles)
        cv2.putText(img, f"a={a:+.1f} b={b:+.1f} cp={self.sim.checkpoint_idx}/{len(cps)-1}",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return img
