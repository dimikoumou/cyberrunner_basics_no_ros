#!/usr/bin/env python3
"""
Hole-avoidance test in the simulator (2026-09-27): every episode puts a hole (radius 8 mm,
keep-out 16 mm) on the straight start -> goal line. The ball "falls" if its centre comes
within the hole radius. Compares ODIL ignoring the hole, ODIL + via points round it (the rig's
current routing), and ODIL v12 which has the hole as input (no via points).
"""
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import eval_controllers as e  # noqa: E402
from odil_friction_comp import ODILFrictionComp  # noqa: E402
from plate_goal_env import PlateGoalEnv  # noqa: E402
from odil_plate_v12 import hole_features  # noqa: E402

HOLE_R, RK = 0.008, 0.016
sys.path.insert(0, __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", "rl_hw")))
import types  # noqa: E402
sys.modules.setdefault("cv2", types.ModuleType("cv2"))            # hole.detour needs neither
_gc = types.ModuleType("goal_circle"); _gc.pixel_to_plate = None
sys.modules.setdefault("goal_circle", _gc)
import hole as holemod  # noqa: E402


class HoleODIL(ODILFrictionComp):
    """ODIL + fc; v12 policies (14 inputs) get the hole features appended"""
    hole = None

    def __call__(self, obs):
        if self.n_in == 14:
            p = np.asarray(obs[4:6]) * 0.14
            self._extra = hole_features(p, self.hole, RK)
        return super().__call__(obs)


class Vias:
    """drive to via points round the hole first (hole.py detour), then the goal"""
    def __init__(self, ctrl):
        self.c = ctrl

    def reset(self):
        self.c.reset()
        self.vias = None

    def __call__(self, obs):
        p = np.asarray(obs[4:6]) * 0.14
        goal = p + np.asarray(obs[0:2]) * 0.1
        if self.vias is None:
            self.vias = list(holemod.detour([{"center": tuple(self.hole), "radius": HOLE_R}], p, goal))
        if self.vias and np.hypot(*(self.vias[0] - p)) < 0.02:
            self.vias.pop(0)
        o = np.array(obs, dtype=float)
        if self.vias:
            o[0:2] = (self.vias[0] - p) / 0.1
        return self.c(o)


def run(ctrl, n=200, seed=777):
    env = PlateGoalEnv(seed=seed)
    fell = reached = 0
    t_reach, jerks = [], []
    for ep in range(n):
        o, _ = env.reset(seed=seed + ep)
        env.switch_at = -1
        p0, g = env.sim.pos.copy(), env.goal.copy()
        rng = np.random.default_rng(seed + ep)
        if np.hypot(*(g - p0)) < 0.06:
            continue
        d = (g - p0) / np.hypot(*(g - p0))
        H = p0 + rng.uniform(0.35, 0.65) * (g - p0) + np.array([-d[1], d[0]]) * rng.uniform(-0.5, 0.5) * RK
        for c in (ctrl, getattr(ctrl, "c", None)):
            if c is not None:
                c.hole = H
        ctrl.hole = H
        ctrl.reset()
        prev, dj, got, f = np.zeros(2), [], None, False
        for k in range(300):
            o, r, te, tr, info = env.step(ctrl(o))
            dj.append(float(np.sum((env.applied - prev) ** 2)))
            prev = env.applied.copy()
            if np.hypot(*(env.sim.pos - H)) < HOLE_R:
                f = True
                break
            if got is None and info["inside"]:
                got = k
        fell += f
        if not f and got is not None:
            reached += 1
            t_reach.append(got * e.DT)
        jerks.append(np.mean(dj))
    m = fell + reached + (n - fell - reached)
    return dict(n=len(jerks), fell=fell / len(jerks), reached=reached / len(jerks),
                t_reach=float(np.median(t_reach)) if t_reach else None, jerk=float(np.mean(jerks)))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    for name, c in [("v11+fc, ignores hole", HoleODIL("runs/odil_v11/odil_policy.npz")),
                    ("v11+fc + via points", Vias(HoleODIL("runs/odil_v11/odil_policy.npz"))),
                    ("v12+fc, hole as input", HoleODIL("runs/odil_v12/odil_policy.npz"))]:
        r = run(c, n)
        print(f"{name:24s} episodes {r['n']}  fell {r['fell'] * 100:4.0f} %  reached {r['reached'] * 100:4.0f} %  "
              f"t_reach {r['t_reach']:.2f}s  jerk {r['jerk']:.4f}")
