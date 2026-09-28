#!/usr/bin/env python3
"""
Path-tracking test in the full plate simulator (2026-09-27): the ball follows the drawing
shapes (rl_hw/shapes.py) with the rig's PathTracker reference; the distance of the TRUE ball
to the line is measured (median / p90 / max, mm). Controllers:
  classic -- the rig's line law (pd_balance): feed-forward of the reference acceleration +
             rolling friction, KP/KD on the reference, breakaway boost along the route
  odil    -- an ODIL tracking policy (odil_track.py) given the reference p/v/a
Rig rate limit (0.5 action per step), camera delay/noise, stiction: all from plate_sim.

  python3 eval_track.py [n_sims] [policy.npz]
"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "rl_hw"))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
_gc = types.ModuleType("goal_circle")
_gc.red_mask = _gc.pixel_to_plate = None
sys.modules.setdefault("goal_circle", _gc)
from line_path import PathTracker  # noqa: E402
import shapes  # noqa: E402
from plate_sim import PlateSim  # noqa: E402

DT = 1.0 / 29.0
KP_LINE, KD_LINE, BOOST_RATE, BOOST_MAX = 8.0, 2.5, 1.0, 0.6
ACC_PER_ACTION, A_ROLL_FF = 0.45, 0.03


class Classic:
    def reset(self):
        self.boost = 0.0

    def __call__(self, ref, pos, vel):
        th = ref["t_hat"]
        if np.hypot(*vel) < 0.01 and ref["lag"] > 0.005:
            self.boost = min(BOOST_MAX, self.boost + BOOST_RATE * DT)
        else:
            self.boost = max(0.0, self.boost - 2 * BOOST_RATE * DT)
        ff = (ref["a"] + (A_ROLL_FF * th if np.hypot(*ref["v"]) > 0.005 else 0.0)) / ACC_PER_ACTION
        return ff + KP_LINE * (ref["p"] - pos) + KD_LINE * (ref["v"] - vel) + self.boost * th


def run(ctrl, shape, speed, n=20, seed=11):
    offs, done, times = [], 0, []
    for k in range(n):
        rng = np.random.default_rng(seed + k)
        sim = PlateSim(rng)
        path = shapes.shape(shape, (-0.03, 0.01), 0.035)
        sim.reset(path[0])
        tr = PathTracker(path, False, path[0], keep_direction=True, v=speed)
        ctrl.reset()
        applied = np.zeros(2)
        pos, vel = path[0].copy(), np.zeros(2)
        for step in range(int(40 / DT)):
            ref = tr.update(pos, DT)
            a = np.clip(ctrl(ref, pos, vel), -0.8, 0.8)
            applied = np.clip(a, applied - 0.5, applied + 0.5)
            pos, vel, _, _ = sim.step(5.0 * applied, dt=DT)
            offs.append(tr.off_line(sim.pos))
            if ref["finished"]:
                done += 1
                times.append(step * DT)
                break
    o = np.array(offs) * 1000
    return np.median(o), np.percentile(o, 90), np.max(o), done / n, (np.median(times) if times else float("nan"))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ctrls = {"classic": Classic()}
    if len(sys.argv) > 2:
        from odil_track import TrackPolicy
        ctrls["odil"] = TrackPolicy(sys.argv[2])
    for shape in ("circle", "square", "star"):
        for speed in (0.04, 0.02):
            for name, c in ctrls.items():
                m, p90, mx, dn, tm = run(c, shape, speed, n)
                print(f"{shape:7s} {speed * 100:.0f} cm/s  {name:8s} median {m:5.1f} mm  p90 {p90:5.1f} mm  "
                      f"max {mx:5.1f} mm  completed {dn * 100:3.0f} %  time {tm:4.1f} s")
