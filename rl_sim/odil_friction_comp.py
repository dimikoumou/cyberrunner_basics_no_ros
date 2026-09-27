"""
ODIL + stiction compensation (2026-09-27): the ODIL policy drives; when the ball has been
still for STILL_S outside the target, a breakaway boost along the goal direction ramps up
(RAMP deg/s, max BOOST_MAX deg) until the ball moves, then is removed -- the classic
controller's pulse idea (Yang & Tomizuka) as a plain friction-compensation add-on.
Works in the simulator eval (obs layout of plate_goal_env) and is ported to the rig.
"""
import numpy as np

from eval_controllers import ODILController, DT


class ODILFrictionComp(ODILController):
    STILL_PTP, STILL_N, RAMP, BOOST_MAX = 0.0015, 12, 6.0, 2.0   # 1.5 mm over ~0.4 s (as classic)
    OUTSIDE = 1.0                                                # boost only this x radius or further out

    def reset(self):
        super().reset()
        self.hist, self.boost = [], 0.0

    def __call__(self, obs):
        rel, v = np.asarray(obs[0:2]) * 0.1, np.asarray(obs[2:4]) * 0.1
        radius = float(obs[8]) * 0.03
        a = super().__call__(obs)                      # env action (x ACTION_SCALE = deg/5)
        d = float(np.hypot(*rel))
        self.hist = (self.hist + [rel.copy()])[-self.STILL_N:]     # goal fixed -> rel tracks the ball
        H = np.array(self.hist)
        still = len(H) == self.STILL_N and np.ptp(H[:, 0]) < self.STILL_PTP and np.ptp(H[:, 1]) < self.STILL_PTP
        if still and d > self.OUTSIDE * radius:
            self.boost = min(self.BOOST_MAX, self.boost + self.RAMP * DT)
        elif not still:
            self.boost = 0.0
        if self.boost > 0:
            from plate_goal_env import ACTION_SCALE
            a = np.clip(a + (rel / max(d, 1e-6)) * self.boost / 5.0 / ACTION_SCALE, -1, 1)
        return a
