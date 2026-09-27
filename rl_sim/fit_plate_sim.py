#!/usr/bin/env python3
"""
Calibrate plate_sim.py against real rig logs: replay the logged commands from the
logged starting state, 1 s at a time, and compare predicted vs measured ball
position. Grid-searches tilt->acceleration gain, rolling friction and the total
command->camera delay (identifiable only as a sum, so camera delay is folded into
the servo delay here).

Usage:  python3 fit_plate_sim.py log1.csv [log2.csv ...]
"""
import csv
import itertools
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from plate_sim import PlateSim  # noqa: E402

HORIZON = 30          # steps (~1 s)
MAX_ACTION_DELTA = 0.5


def load(fn):
    r = list(csv.DictReader(open(fn)))
    g = lambda k: np.array([float(q[k]) if q.get(k) not in (None, "") else np.nan for q in r])
    t, x, y, fd, ep = g("t"), g("xb"), g("yb"), g("ball_found"), g("episode")
    a = np.column_stack([g("a0"), g("a1")])
    # the env rate-limits the logged action by 0.5/step, then the servo clips to +-1
    applied = np.zeros_like(a)
    prev = np.zeros(2)
    for i in range(len(a)):
        if i and ep[i] != ep[i - 1]:
            prev = np.zeros(2)
        prev = np.clip(np.clip(a[i], prev - MAX_ACTION_DELTA, prev + MAX_ACTION_DELTA), -1, 1)
        applied[i] = prev
    return t, np.column_stack([x, y]), fd, ep, applied


def segments(t, pos, fd, ep, stride=15):
    out = []
    for i0 in range(10, len(t) - HORIZON - 1, stride):
        sl = slice(i0 - 2, i0 + HORIZON + 1)
        if fd[sl].min() < 1 or ep[sl].min() != ep[sl].max():
            continue
        moved = np.max(np.hypot(*(pos[i0:i0 + HORIZON] - pos[i0]).T))
        if moved < 0.01:              # only snippets where the ball really moves
            continue
        if np.max(np.abs(pos[i0:i0 + HORIZON])) > 0.12:   # skip wall contacts
            continue
        out.append(i0)
    return out


def replay(params, data, i0):
    t, pos, fd, ep, applied = data
    sim = PlateSim(np.random.default_rng(0), params=dict(params, slope_amp_deg=0.0, level_offset_deg=0.0,
                                                         pos_noise=0.0, tilt_noise_deg=0.0, miss_prob=0.0,
                                                         cam_delay=0), randomize=False)
    d = sim.p["servo_delay"]
    v0 = (pos[i0] - pos[i0 - 2]) / max(t[i0] - t[i0 - 2], 1e-3)
    sim.reset(pos[i0], v0, tilt=5.0 * applied[i0 - d - 1] if i0 - d - 1 >= 0 else (0, 0))
    sim.rest_t = 0.0 if np.hypot(*v0) > 0.005 else 1.0
    sim.cmd_hist = [5.0 * applied[i0 - d + k] for k in range(d)] + [5.0 * applied[i0]]
    sim.cmd_hist = sim.cmd_hist[-(d + 1):]
    preds = []
    for k in range(1, HORIZON + 1):
        m_pos, _, _, _ = sim.step(5.0 * applied[i0 + k], dt=t[i0 + k] - t[i0 + k - 1])
        preds.append(m_pos.copy())
    preds = np.array(preds)
    meas = pos[i0 + 1:i0 + HORIZON + 1]
    tt = t[i0 + 1:i0 + HORIZON + 1] - t[i0]
    # unknown local slope: allow a constant extra acceleration per snippet (least squares)
    basis = 0.5 * tt ** 2
    b = ((meas - preds) * basis[:, None]).sum(0) / max((basis ** 2).sum(), 1e-12)
    b = np.clip(b, -0.12, 0.12)          # at most ~1.3 deg of unknown slope
    return np.hypot(*(meas - preds - np.outer(basis, b)).T)


def main():
    datas = [load(fn) for fn in sys.argv[1:]]
    segs = [(d, i0) for d in datas for i0 in segments(*d[:4])]
    print(f"{len(segs)} moving 1-s snippets from {len(datas)} logs")
    grid = dict(k_acc=[0.07, 0.08, 0.09, 0.10, 0.11], a_roll=[0.02, 0.035, 0.05, 0.07], servo_delay=[2, 3, 4, 5])
    results = []
    for k_acc, a_roll, sd in itertools.product(*grid.values()):
        params = dict(k_acc=k_acc, a_roll=a_roll, servo_delay=sd)
        e = np.array([replay(params, d, i0) for d, i0 in segs])
        results.append((float(np.median(e[:, 14])), float(np.median(e[:, -1])), params))
    results.sort(key=lambda r: r[1])
    print("best (median error at 0.5 s / 1.0 s):")
    for e05, e10, p in results[:6]:
        print(f"  {e05 * 1000:5.1f} / {e10 * 1000:5.1f} mm  {p}")
    worst = results[-1]
    print(f"worst: {worst[0] * 1000:.1f} / {worst[1] * 1000:.1f} mm {worst[2]}")
    base = [r for r in results if r[2] == dict(k_acc=0.09, a_roll=0.025, servo_delay=4)]
    print("(errors after fitting a constant local slope per snippet)")
    print("zero-motion baseline (predict 'stays put'):",
          f"{np.median([np.hypot(*(d[1][i0 + 14] - d[1][i0])) for d, i0 in segs]) * 1000:.1f} / "
          f"{np.median([np.hypot(*(d[1][i0 + HORIZON] - d[1][i0])) for d, i0 in segs]) * 1000:.1f} mm")


if __name__ == "__main__":
    main()
