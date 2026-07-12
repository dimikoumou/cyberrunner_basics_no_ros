"""
2D physics simulation of the CyberRunner labyrinth: a steel ball rolling on a
plane that tilts about two axes (alpha, beta), with maze walls, holes, and a goal.

Coordinate frame (matches the real state estimator's output):
    - x, y in METERS, origin at the center of the play field, x right, y up.
    - plate angles alpha, beta in RADIANS.
    - The ball position/velocity the sim reports are the same quantities the real
      EstimationPipeline produces, so a policy trained here consumes the same
      observation on hardware (sim-to-real).

Pure numpy -- no ML dependency. The physics is deliberately simple but captures
the parts that matter for control: gravity-from-tilt, rolling inertia, damping,
wall collisions (circle vs. segment), holes (fall = fail), and a goal region.
"""
from dataclasses import dataclass, field
import numpy as np

G = 9.81                 # gravity, m/s^2
ROLL_FACTOR = 5.0 / 7.0  # solid-sphere rolling-without-slipping acceleration factor


@dataclass
class MazeConfig:
    # play field half-extents (meters); field spans [-hx, hx] x [-hy, hy]
    hx: float = 0.13
    hy: float = 0.11
    ball_radius: float = 0.0065      # ~13 mm steel ball
    hole_radius: float = 0.009
    # walls: array (N,4) of segments [x1,y1,x2,y2] in meters
    walls: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    # holes: array (M,2) of centers [x,y]; radius = hole_radius
    holes: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    # ordered path checkpoints (K,2); last one is the goal
    checkpoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    start: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    goal_radius: float = 0.012
    checkpoint_radius: float = 0.014
    # control / actuation
    max_tilt: float = 0.10           # rad (~5.7 deg) max plate tilt each axis
    max_tilt_rate: float = 3.0       # rad/s servo slew-rate limit
    # dynamics
    damping: float = 1.2             # viscous damping (1/s), models rolling resistance
    restitution: float = 0.15        # wall bounciness (0=stick, 1=elastic)
    wall_half_thickness: float = 0.002


class MazeSim:
    def __init__(self, cfg: MazeConfig, fps: int = 60, substeps: int = 8):
        self.cfg = cfg
        self.dt = 1.0 / fps
        self.substeps = substeps
        # boundary walls (the field edges)
        hx, hy = cfg.hx, cfg.hy
        border = np.array([
            [-hx, -hy,  hx, -hy],
            [ hx, -hy,  hx,  hy],
            [ hx,  hy, -hx,  hy],
            [-hx,  hy, -hx, -hy],
        ])
        self.walls = np.vstack([border, cfg.walls]) if len(cfg.walls) else border
        self.reset()

    def reset(self, start=None):
        self.pos = np.array(self.cfg.start if start is None else start, dtype=float)
        self.vel = np.zeros(2)
        self.angles = np.zeros(2)          # [alpha, beta], current plate tilt
        self.target_angles = np.zeros(2)
        self.checkpoint_idx = 0
        self.status = "playing"            # "playing" | "goal" | "hole"
        return self.state()

    def state(self):
        return {
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "angles": self.angles.copy(),
            "checkpoint_idx": self.checkpoint_idx,
            "status": self.status,
        }

    # ---- dynamics -------------------------------------------------------
    def set_target_angles(self, alpha, beta):
        mt = self.cfg.max_tilt
        self.target_angles = np.clip(np.array([alpha, beta]), -mt, mt)

    def _slew_angles(self, dt):
        # servos can't jump instantly: move current angle toward target, rate-limited
        max_step = self.cfg.max_tilt_rate * dt
        delta = np.clip(self.target_angles - self.angles, -max_step, max_step)
        self.angles += delta

    def _acceleration(self):
        # A plate tilted by alpha (about x) / beta (about y) makes the ball roll
        # downhill. Small-angle-consistent: use sin of the tilt on each axis.
        alpha, beta = self.angles
        ax = ROLL_FACTOR * G * np.sin(beta)
        ay = -ROLL_FACTOR * G * np.sin(alpha)
        acc = np.array([ax, ay]) - self.cfg.damping * self.vel
        return acc

    def step(self, action):
        """action: [target_alpha_norm, target_beta_norm] in [-1, 1]."""
        action = np.asarray(action, dtype=float).reshape(2)
        self.set_target_angles(*(np.clip(action, -1, 1) * self.cfg.max_tilt))
        dt_sub = self.dt / self.substeps
        for _ in range(self.substeps):
            self._slew_angles(dt_sub)
            self.vel += self._acceleration() * dt_sub
            self.pos += self.vel * dt_sub
            self._resolve_walls()
            if self._check_hole():
                self.status = "hole"
                return self.state()
        self._update_checkpoint()
        return self.state()

    # ---- collisions -----------------------------------------------------
    def _resolve_walls(self):
        r = self.cfg.ball_radius + self.cfg.wall_half_thickness
        for _ in range(2):  # a couple of relaxation iterations for corners
            for w in self.walls:
                cp = _closest_point_on_segment(self.pos, w[:2], w[2:])
                d = self.pos - cp
                dist = np.linalg.norm(d)
                if dist < r and dist > 1e-9:
                    n = d / dist
                    # push out of the wall
                    self.pos = cp + n * r
                    # kill inward velocity component, apply restitution
                    vn = self.vel @ n
                    if vn < 0:
                        self.vel -= (1 + self.cfg.restitution) * vn * n
                elif dist <= 1e-9:
                    # exactly on the segment: nudge along segment normal
                    seg = w[2:] - w[:2]
                    n = np.array([-seg[1], seg[0]])
                    n = n / (np.linalg.norm(n) + 1e-9)
                    self.pos = cp + n * r

    def _check_hole(self):
        if len(self.cfg.holes) == 0:
            return False
        d = np.linalg.norm(self.cfg.holes - self.pos, axis=1)
        # ball falls in if its center gets within (hole_radius - small margin)
        return bool(np.any(d < self.cfg.hole_radius))

    def _update_checkpoint(self):
        cps = self.cfg.checkpoints
        if len(cps) == 0:
            return
        nxt = cps[self.checkpoint_idx]
        if np.linalg.norm(self.pos - nxt) < self.cfg.checkpoint_radius:
            if self.checkpoint_idx >= len(cps) - 1:
                self.status = "goal"
            else:
                self.checkpoint_idx += 1

    def dist_to_next_checkpoint(self):
        cps = self.cfg.checkpoints
        if len(cps) == 0:
            return 0.0
        return float(np.linalg.norm(self.pos - cps[self.checkpoint_idx]))


def _closest_point_on_segment(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab
