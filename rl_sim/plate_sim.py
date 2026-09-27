"""
Ball-on-plate simulator of the real CyberRunner plate rig (2026-09-27), for training
a goal-reaching policy: "click anywhere and the ball goes there", learned instead of
hand-tuned PID/planning.

Everything here is modelled on measurements from the rig (rl_hw logs, 2026-09-26):
  - control loop ~29 Hz (camera 30 fps), jittery dt
  - action a in [-1, 1]^2 -> target tilt 5 deg * a (ball x <- tilt_x, ball y <- tilt_y;
    the real env maps these onto beta / -alpha); the env rate-limits the action
    change to 0.5 per step
  - the closed-loop tilt servo follows the target with a delay of ~2-3 steps and
    a small first-order lag; the camera adds ~1-2 more steps (total ~4-5 steps, i.e.
    0.13-0.16 s command -> seen)
  - ball acceleration ~0.09 m/s^2 per deg of tilt; rolling friction ~0.03 m/s^2
  - paper stiction: a ball at rest sinks into the paper; the tilt needed to break it
    free grows from ~0.4 deg to 1.2-2.6 deg within ~0.3 s of rest
  - local slope field: the paper isn't flat (plate map: local level varies by ~1 deg)
  - plate half-extents 0.1417 x 0.1192 m (ball centre limit a bit less), weak wall bounce
  - camera: ~0.3 mm position noise, occasional missed frames (last position held),
    velocity = finite difference of noisy positions over the real dt
All physical parameters are domain-randomised per episode around those values.

The environment is pure numpy (runs in both the rig venv and the training venv).
"""
import numpy as np

X_HALF, Y_HALF = 0.1417, 0.1192
BALL_LIMIT = (X_HALF - 0.007, Y_HALF - 0.007)
DT_NOM = 1.0 / 29.0
SUBSTEPS = 8

# nominal values (fitted/checked against rig logs in fit_plate_sim.py)
NOMINAL = dict(
    k_acc=0.11,          # m/s^2 per deg (fit on rig logs: 0.10-0.11)
    a_roll=0.025,        # m/s^2 (fit: 0.02-0.035)
    static_roll_deg=0.4,     # breakaway right after stopping
    static_rest_deg=1.8,     # breakaway after resting a while (dimple)
    dimple_tau=0.3,      # s
    servo_delay=2,       # steps from command to plate motion (fit: total 2-3 incl. camera)
    servo_tau=0.04,      # s first-order lag of the plate tilt
    cam_delay=1,         # steps camera latency
    slope_amp_deg=0.6,   # local slope field amplitude
    level_offset_deg=0.3,  # global level error (the controller's level offset is not exact)
    wall_e=0.2,          # restitution
    pos_noise=0.0003,    # m
    tilt_noise_deg=0.15,
    miss_prob=0.01,
)

RANDOMIZE = dict(
    k_acc=(0.085, 0.125),
    a_roll=(0.015, 0.05),
    static_roll_deg=(0.25, 0.6),
    static_rest_deg=(1.2, 2.6),
    dimple_tau=(0.15, 0.6),
    servo_delay=(1, 3),
    servo_tau=(0.02, 0.07),
    cam_delay=(1, 1),
    slope_amp_deg=(0.0, 1.0),
    level_offset_deg=(0.0, 0.6),
    wall_e=(0.05, 0.4),
    pos_noise=(0.0002, 0.0006),
    tilt_noise_deg=(0.05, 0.25),
    miss_prob=(0.0, 0.03),
)


class SlopeField:
    """Smooth random local slope (deg per axis) over the plate: a few random
    low-frequency sinusoids, like the measured plate map (+-~1 deg)."""

    def __init__(self, rng, amp_deg, n=4):
        self.k = rng.uniform(8, 35, size=(n, 2)) * rng.choice([-1, 1], size=(n, 2))   # rad/m
        self.ph = rng.uniform(0, 2 * np.pi, size=(n, 2))
        self.a = rng.normal(0, amp_deg / np.sqrt(n), size=(n, 2))

    def __call__(self, xy):
        arg = self.k[:, 0] * xy[0] + self.k[:, 1] * xy[1]
        return np.array([np.sum(self.a[:, 0] * np.sin(arg + self.ph[:, 0])),
                         np.sum(self.a[:, 1] * np.sin(arg + self.ph[:, 1]))])


class PlateSim:
    """Physics + sensing of one episode. step(action) advances one control step and
    returns the camera measurement the controller would see."""

    def __init__(self, rng, params=None, randomize=True):
        self.rng = rng
        p = dict(NOMINAL)
        if randomize:
            for k, (lo, hi) in RANDOMIZE.items():
                p[k] = int(rng.integers(lo, hi + 1)) if isinstance(lo, int) else float(rng.uniform(lo, hi))
        if params:
            p.update(params)
        self.p = p
        self.slope = SlopeField(rng, p["slope_amp_deg"])
        off = rng.normal(0, 1, 2)
        self.level_off = off / max(np.hypot(*off), 1e-9) * p["level_offset_deg"]

    def reset(self, pos, vel=(0.0, 0.0), tilt=(0.0, 0.0)):
        p = self.p
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.tilt = np.array(tilt, dtype=float)          # actual plate tilt (deg, x/y)
        self.rest_t = 1.0                                 # starts resting in a dimple
        self.cmd_hist = [np.array(tilt, dtype=float)] * (p["servo_delay"] + 1)
        hist_len = p["cam_delay"] + 1
        self.meas_hist = [(self.pos.copy(), self.tilt.copy())] * hist_len
        self.last_meas_pos = self.pos.copy()
        self.t = 0.0

    def _static_deg(self):
        p = self.p
        f = 1.0 - np.exp(-self.rest_t / p["dimple_tau"])
        return p["static_roll_deg"] + (p["static_rest_deg"] - p["static_roll_deg"]) * f

    def step(self, target_tilt_deg, dt=None):
        """target_tilt_deg: commanded tilt (deg, x/y) after the env's rate limit."""
        p = self.p
        dt = dt if dt is not None else DT_NOM
        self.cmd_hist.append(np.array(target_tilt_deg, dtype=float))
        cmd = self.cmd_hist.pop(0)                        # what reaches the plate now
        h = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            self.tilt += (cmd - self.tilt) * min(1.0, h / max(p["servo_tau"], 1e-4))
            eff = self.tilt + self.level_off + self.slope(self.pos)     # deg of gravity drive
            drive = p["k_acc"] * eff
            speed = np.hypot(*self.vel)
            if speed < 0.002:
                # at rest: stays unless the drive beats the (growing) static friction
                if np.hypot(*drive) < p["k_acc"] * self._static_deg():
                    self.vel[:] = 0.0
                    self.rest_t += h
                    continue
                self.rest_t = 0.0
                acc = drive - p["a_roll"] * drive / max(np.hypot(*drive), 1e-9)
            else:
                self.rest_t = 0.0
                acc = drive - p["a_roll"] * self.vel / speed
            new_v = self.vel + acc * h
            if speed >= 0.002 and np.dot(new_v, self.vel) < 0 and np.hypot(*drive) < p["a_roll"]:
                new_v[:] = 0.0                            # friction stops it, doesn't reverse it
            self.vel = new_v
            self.pos += self.vel * h
            for ax in (0, 1):
                lim = BALL_LIMIT[ax]
                if abs(self.pos[ax]) > lim:
                    self.pos[ax] = np.sign(self.pos[ax]) * lim
                    if self.vel[ax] * np.sign(self.pos[ax]) > 0:
                        self.vel[ax] = -p["wall_e"] * self.vel[ax]
        self.t += dt
        # camera: delayed, noisy, sometimes missing
        self.meas_hist.append((self.pos.copy(), self.tilt.copy()))
        m_pos, m_tilt = self.meas_hist.pop(0)
        found = self.rng.random() >= p["miss_prob"]
        if found:
            m_pos = m_pos + self.rng.normal(0, p["pos_noise"], 2)
            m_tilt = m_tilt + self.rng.normal(0, p["tilt_noise_deg"], 2)
        else:
            m_pos = self.last_meas_pos.copy()
        vel_meas = (m_pos - self.last_meas_pos) / dt
        self.last_meas_pos = m_pos
        return m_pos, vel_meas, m_tilt, found
