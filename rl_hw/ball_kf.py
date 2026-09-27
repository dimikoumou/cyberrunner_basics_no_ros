"""
Delay-compensated Kalman filter for the ball (research experiment #4, 2026-09-26).

Why: the camera image is ~0.13-0.16 s old when it arrives (4-5 loop steps), and
velocity from frame differences is noisy (5-7 mm/s std on a ball at rest). Near a
small target the controller therefore reacts to where the ball WAS, and brakes /
releases too late. This filter estimates position and velocity per axis from the
camera, and then rolls the estimate forward through the tilt commands that were
already sent but whose effect the camera hasn't seen yet -- the Smith-predictor
idea, as in Matous et al. (IFAC 2019, augmented-state KF for vision delay) and the
CyberRunner work, which conditions on past actions for the same reason.

Model per axis (x uses beta, y uses alpha; both are "ball acceleration from the
commanded tilt along that axis"):
    p' = v
    v' = K_ACC * (5 deg * a_cmd(t - D)) - rolling friction * sign(v)
where a_cmd is the applied action (after the env's rate limit) and the level offset
is already inside the tilt servo, so action 0 = level. The measurement is position.
"""
import numpy as np

K_ACC = 0.09         # m/s^2 per deg of tilt (fitted from the logs: 0.087-0.099)
DEG_PER_ACTION = 5.0  # TILT_MAX_DEG in plate_env
A_ROLL = 0.03         # m/s^2 rolling-friction deceleration (fitted: 0.026-0.036)
V_ROLL_MIN = 0.005    # below this speed, friction is not applied (ball at rest)


class BallKF:
    def __init__(self, delay_steps=5, q_pos=1e-7, q_vel=4e-4, r_pos=(0.00025) ** 2):
        self.delay = delay_steps
        self.q = np.diag([q_pos, q_vel])
        self.r = r_pos
        self.x = None                       # (2 axes, [p, v])
        self.P = None
        self.cmd_hist = []                  # applied actions, oldest first
        self.dt_hist = []

    def reset(self, pos):
        self.x = np.array([[pos[0], 0.0], [pos[1], 0.0]], dtype=float)
        self.P = np.array([np.diag([1e-6, 1e-2]) for _ in range(2)])
        self.cmd_hist, self.dt_hist = [], []

    @staticmethod
    def _accel(a_cmd, v):
        acc = K_ACC * DEG_PER_ACTION * a_cmd
        if abs(v) > V_ROLL_MIN:
            acc -= A_ROLL * np.sign(v)
        return acc

    def _predict(self, x, P, a_cmd, dt):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        acc = self._accel(a_cmd, x[1])
        x = np.array([x[0] + x[1] * dt + 0.5 * acc * dt * dt, x[1] + acc * dt])
        P = F @ P @ F.T + self.q * max(dt, 1e-3) / 0.034
        return x, P

    def step(self, meas_pos, found, applied_action, dt):
        """Call once per control step AFTER env.step(): meas_pos is the (x, y) the
        camera reported this step (seeing the plate as it was `delay` steps ago),
        applied_action the action that was just sent. Returns (pos_now, vel_now):
        the estimate rolled forward to the present through the unseen commands."""
        if self.x is None:
            if not found:
                return None, None
            self.reset(meas_pos)
        self.cmd_hist.append(np.asarray(applied_action, dtype=float))
        self.dt_hist.append(dt)
        self.cmd_hist = self.cmd_hist[-(self.delay + 2):]
        self.dt_hist = self.dt_hist[-(self.delay + 2):]
        # the command acting on the frame we just received was sent `delay` steps ago
        a_seen = self.cmd_hist[0] if len(self.cmd_hist) <= self.delay else self.cmd_hist[-1 - self.delay]
        for ax in range(2):
            self.x[ax], self.P[ax] = self._predict(self.x[ax], self.P[ax], a_seen[ax], dt)
            if found:
                H = np.array([1.0, 0.0])
                S = H @ self.P[ax] @ H + self.r
                Kg = self.P[ax] @ H / S
                self.x[ax] = self.x[ax] + Kg * (meas_pos[ax] - self.x[ax][0])
                self.P[ax] = self.P[ax] - np.outer(Kg, H @ self.P[ax])
        # roll forward through the commands the camera hasn't seen yet
        pos_now, vel_now = np.zeros(2), np.zeros(2)
        unseen = self.cmd_hist[-self.delay:] if len(self.cmd_hist) > self.delay else self.cmd_hist[1:]
        for ax in range(2):
            p, v = self.x[ax]
            for a in unseen:
                acc = self._accel(a[ax], v)
                p, v = p + v * dt + 0.5 * acc * dt * dt, v + acc * dt
            pos_now[ax], vel_now[ax] = p, v
        return pos_now, vel_now

    def filtered(self):
        """Filtered (not rolled-forward) position and velocity at the camera's time."""
        return self.x[:, 0].copy(), self.x[:, 1].copy()
