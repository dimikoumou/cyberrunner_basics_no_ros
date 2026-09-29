#!/usr/bin/env python3
"""
Fall diagnosis for any maze (2026-09-28): scans the rig's logs for balls lost next to a hole
of the current maze (maze/route.json) and describes the last second before each fall --
speed, how directly the ball headed for the hole, how far it was off the route, how long it
had been resting (a resting ball at a hole's lip that then drops = a push/jolt problem, a
fast one = a braking/speed problem). Per hole: count and the dominant pattern.

  python3 maze_falls.py [log.csv ...]        (default: all pd_logs since the route capture)
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROUTE = os.path.join(ROOT, "maze", "route.json")
NEAR_HOLE_M = 0.012            # lost within this of a hole's edge = a fall into that hole
WINDOW_S = 1.0


def falls_in(fn, holes, route):
    rows = list(csv.DictReader(open(fn)))
    g = lambda q, k: float(q[k]) if q.get(k) not in (None, "", "nan") else np.nan
    t = np.array([g(q, "t") for q in rows])
    x, y = np.array([g(q, "xb") for q in rows]), np.array([g(q, "yb") for q in rows])
    found = np.array([g(q, "ball_found") for q in rows])
    status = [q.get("status", "") for q in rows]
    out = []
    for i in range(1, len(rows)):
        if not (found[i - 1] == 1 and found[i] == 0 and status[i - 1] in ("running", "in_circle")):
            continue
        p = np.array([x[i - 1], y[i - 1]])
        if not np.all(np.isfinite(p)):
            continue
        edge = [np.hypot(*(p - c)) - r for c, r in holes]
        k = int(np.argmin(edge))
        if edge[k] > NEAR_HOLE_M:
            continue
        # ball still lost 0.5 s later = a real fall, not a detection flicker
        j = np.searchsorted(t, t[i] + 0.5)
        if j < len(rows) and found[i:j + 1].max() == 1:
            continue
        w = (t >= t[i - 1] - WINDOW_S) & (t <= t[i - 1]) & (found == 1)
        P = np.column_stack([x[w], y[w]])
        tw = t[w]
        if len(P) < 5:
            continue
        v = np.diff(P, axis=0) / np.maximum(np.diff(tw), 1e-3)[:, None]
        speed = float(np.median(np.hypot(*v[-8:].T)))
        c = holes[k][0]
        head = P[-1] - P[max(0, len(P) - 8)]
        to_h = c - P[max(0, len(P) - 8)]
        cosang = float(head @ to_h / max(np.hypot(*head) * np.hypot(*to_h), 1e-9)) if np.hypot(*head) > 1e-3 else 0.0
        moved = float(np.hypot(*(P[-1] - P[0])))
        off = float(np.min(np.hypot(*(route - P[-1]).T)))
        prog = float(np.argmin(np.hypot(*(route - P[-1]).T)) / (len(route) - 1))
        out.append({"log": os.path.basename(fn), "t": float(t[i - 1]), "hole": k, "progress": prog,
                    "speed_mm_s": 1000 * speed, "moved_last_1s_mm": 1000 * moved,
                    "heading_to_hole_cos": cosang, "off_route_mm": 1000 * off,
                    "kind": ("resting at the lip" if moved < 0.004 else
                             "fast into it" if speed > 0.03 else "drifted in")})
    return out


def main():
    r = json.load(open(ROUTE))
    holes = [(np.array(h["center"]), float(h["radius"])) for h in r["holes"]]
    route = np.array(r["route_m"])
    t_route = os.path.getmtime(ROUTE)
    files = sys.argv[1:] or [f for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "pd_logs", "pd_*.csv")))
                             if os.path.getmtime(f) > t_route]
    F = []
    for fn in files:
        F += falls_in(fn, holes, route)
    by = defaultdict(list)
    for f in F:
        by[f["hole"]].append(f)
    print(f"{len(F)} falls in {len(files)} log(s)\n")
    print(f"{'hole':>4} {'n':>3} {'at %':>5} {'speed mm/s':>10} {'resting':>8} {'fast':>5} {'drift':>6} {'off route mm':>12}")
    for k, L in sorted(by.items(), key=lambda kv: -len(kv[1])):
        kinds = [f["kind"] for f in L]
        print(f"{k:>4} {len(L):>3} {100 * np.median([f['progress'] for f in L]):>5.0f} "
              f"{np.median([f['speed_mm_s'] for f in L]):>10.0f} {kinds.count('resting at the lip'):>8} "
              f"{kinds.count('fast into it'):>5} {kinds.count('drifted in'):>6} "
              f"{np.median([f['off_route_mm'] for f in L]):>12.1f}")
    with open(os.path.join(ROOT, "phase3_logs", "maze_falls.json"), "w") as fh:
        json.dump(F, fh, indent=1)


if __name__ == "__main__":
    main()
