#!/usr/bin/env python3
"""
Whole maze runs in the simulator (2026-10-02): the rig's control law (rl_hw/line_path.PathTracker
reference with the line-of-sight clamp, the ODIL TrackPolicy, the breakaway boost, the rate limit,
the mid-run jolt) driving the fitted maze world (maze_world.py: physics + walls + holes), with the
camera's noise and one-frame delay. A run ends when the ball's centre crosses a hole edge (fall),
reaches the end (finished) or makes no progress for a long time.

Purpose: (1) check the simulator against the real board -- does the simulated ball get about as
far and fall at about the same places as in phase3_logs/maze_runs.jsonl? (2) score a policy in
simulation before it touches the rig.

  ../.venv-rl/bin/python3 maze_eval.py <policy.npz> [n_runs=40] [--route maze/route.json] [--model runs/maze_world.pt]
"""
import json
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "rl_hw"))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
_gc = types.ModuleType("goal_circle")
_gc.red_mask = _gc.pixel_to_plate = None
sys.modules.setdefault("goal_circle", _gc)
from line_path import PathTracker                      # noqa: E402
from odil_track import TrackPolicy                     # noqa: E402
import maze_world as mw                                # noqa: E402
import world_model as wm                               # noqa: E402

DT = 1.0 / 29.0
MAZE_SPEED, LOS_TOL = 0.025, 0.003
BOOST_RATE, BOOST_MAX = 1.0, 0.6
POS_NOISE = 0.0004
STALL_S, MAX_S = 15.0, 400.0


def run_once(world, policy_path, route, rng, max_s=MAX_S, verbose=False):
    pol = TrackPolicy(policy_path)
    start = route[0] + rng.normal(0, 0.002, 2)
    tracker = PathTracker(route, False, start, keep_direction=True, v=MAZE_SPEED, los_tol=LOS_TOL)
    p = torch.tensor(start[None], dtype=torch.float32)
    v = torch.zeros(1, 2)
    tilt = torch.zeros(1, 2)
    ubuf = [torch.zeros(1, 2) for _ in range(8)]          # commanded tilt (deg), newest last
    applied = np.zeros(2)
    meas_prev = start.copy()
    seen_p, seen_v = start.copy(), np.zeros(2)            # what the controller acts on (1 frame late)
    pend_p, pend_v = start.copy(), np.zeros(2)
    boost, still_anchor, jolts, jolt_q = 0.0, (start.copy(), 0.0), 0, []
    n = len(route)
    max_idx, t_prog, t = 0, 0.0, 0.0
    offs = []
    while t < max_s:
        ref = tracker.update(seen_p, DT)
        if ref["finished"]:
            return {"result": "finished", "progress": 1.0, "t": t, "median_mm": 1000 * float(np.median(offs))}
        if jolt_q:                                        # a jolt in progress overrides the law
            a_cmd = jolt_q.pop(0)
        else:
            a_cmd = pol(ref, seen_p, seen_v, DT)
            to_ref = ref["p"] - seen_p
            d_ref = float(np.hypot(*to_ref))
            if np.hypot(*seen_v) < 0.01 and ref["lag"] > 0.005:
                boost = min(BOOST_MAX, boost + BOOST_RATE * DT)
            else:
                boost = max(0.0, boost - 2 * BOOST_RATE * DT)
            bst = to_ref / d_ref if d_ref > 0.003 else ref["t_hat"]
            a_cmd = np.clip(a_cmd + boost * bst, -0.8, 0.8)
            # mid-run jolt: moved < 3 mm in 4 s and the reference is > 4 mm away
            if np.hypot(*(seen_p - still_anchor[0])) > 0.003:
                still_anchor = (seen_p.copy(), t)
            elif t - still_anchor[1] > 4.0 and d_ref > 0.004 and jolts < 3:
                jolts += 1
                ang = np.radians((0.0, 50.0, -50.0)[(jolts - 1) % 3])
                d = to_ref / max(d_ref, 1e-6)
                d = np.array([d[0] * np.cos(ang) - d[1] * np.sin(ang), d[0] * np.sin(ang) + d[1] * np.cos(ang)])
                jolt_q = [-0.4 * d] * 9 + [1.0 * d] * 17
                still_anchor = (seen_p.copy(), t)
        applied = applied + np.clip(np.clip(a_cmd, -1, 1) - applied, -0.5, 0.5)       # the env's rate limit
        ubuf = ubuf[1:] + [torch.tensor(5.0 * applied[None], dtype=torch.float32)]
        with torch.no_grad():
            uh = torch.stack(ubuf[-5:-1], 1)
            p, v, tilt = world.step(p, v, tilt, uh, ubuf[-1 - world.delay])
            edge = float(world.hole_edge(p)[0])
        pos = p[0].numpy().astype(float)
        t += DT
        d_route = np.hypot(*(route - pos).T)
        idx = int(np.argmin(d_route))
        offs.append(float(d_route[idx]))
        if idx > max_idx + 2:
            max_idx, t_prog, jolts = idx, t, 0
        if edge < 0:
            k = int(torch.argmin(torch.cdist(p, world.hole_c) - world.hole_r[None]))
            return {"result": "fell", "hole": k, "progress": max_idx / (n - 1), "t": t,
                    "speed_mm_s": 1000 * float(v.norm()), "median_mm": 1000 * float(np.median(offs))}
        if t - t_prog > STALL_S * 4:                      # the rig retries forward; give it 4 stall periods
            return {"result": "stuck", "progress": max_idx / (n - 1), "t": t, "at_mm": (1000 * pos).round().tolist(),
                    "median_mm": 1000 * float(np.median(offs))}
        meas = pos + rng.normal(0, POS_NOISE, 2)          # camera: noisy, one frame late
        seen_p, seen_v = pend_p, pend_v
        pend_p, pend_v = meas, (meas - meas_prev) / DT
        meas_prev = meas
    return {"result": "timeout", "progress": max_idx / (n - 1), "t": t, "median_mm": 1000 * float(np.median(offs))}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {sys.argv[i][2:]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
    args = [a for a in args if a not in opt.values()]
    policy = args[0]
    n_runs = int(args[1]) if len(args) > 1 else 40
    route = np.array(json.load(open(opt.get("route", os.path.join(ROOT, "maze", "route.json"))))["route_m"])
    world = mw.MazeWorld()
    world.load_state_dict(torch.load(opt.get("model", os.path.join(HERE, "runs", "maze_world.pt"))))
    torch.set_num_threads(1)
    rng = np.random.default_rng(int(opt.get("seed", "0")))
    R = []
    for i in range(n_runs):
        r = run_once(world, policy, route, rng)
        R.append(r)
        print(i + 1, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    p = np.array([r["progress"] for r in R])
    from collections import Counter
    print(f"\n{n_runs} runs: mean progress {100 * p.mean():.0f} %, median {100 * np.median(p):.0f} %, "
          f"finished {sum(r['result'] == 'finished' for r in R)}, past 50 %: {(p > 0.5).sum()}")
    print("results:", dict(Counter(r["result"] for r in R)))
    print("falls by hole:", Counter(r["hole"] for r in R if r["result"] == "fell").most_common(8))
    json.dump(R, open(os.path.join(HERE, "runs", "maze_eval_last.json"), "w"))


if __name__ == "__main__":
    main()
