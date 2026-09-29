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
import time
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


def falls_from_runs(files, holes, route):
    """falls anchored on the run log (phase3_logs/maze_runs.jsonl): every run that ended in a real
    hole -> the last frames the ball was seen before that moment, in the log covering it"""
    runs = [json.loads(l) for l in open(os.path.join(ROOT, "phase3_logs", "maze_runs.jsonl"))]
    runs = [r for r in runs if r.get("real") and "real hole" in r.get("result", "")]
    logs = []
    for fn in files:
        rows = list(csv.DictReader(open(fn)))
        if rows:
            # log times are seconds since the log started; its start is in the name (local time)
            t0 = time.mktime(time.strptime(os.path.basename(fn)[3:18], "%Y%m%d_%H%M%S"))
            for q in rows:
                q["t"] = str(t0 + float(q["t"]))
            logs.append((float(rows[0]["t"]), float(rows[-1]["t"]), fn, rows))
    out = []
    g = lambda q, k: float(q[k]) if q.get(k) not in (None, "", "nan") else np.nan
    for run in runs:
        L = [x for x in logs if x[0] <= run["t"] <= x[1] + 5]
        if not L:
            continue
        rows = L[0][3]
        t = np.array([g(q, "t") for q in rows])
        found = np.array([g(q, "ball_found") for q in rows])
        i = int(np.searchsorted(t, run["t"]))
        j = i - 1
        while j > 0 and found[j] != 1:          # last frame the ball was seen
            j -= 1
        w = np.arange(max(0, j - 60), j + 1)
        w = w[found[w] == 1]
        P = np.column_stack([[g(rows[k], "xb") for k in w], [g(rows[k], "yb") for k in w]])
        tw = t[w]
        if len(P) < 5 or not np.all(np.isfinite(P[-1])):
            continue
        p = P[-1]
        edge = [np.hypot(*(p - c)) - r for c, r in holes]
        k = int(np.argmin(edge))
        v = np.diff(P, axis=0) / np.maximum(np.diff(tw), 1e-3)[:, None]
        speed = float(np.median(np.hypot(*v[-8:].T)))
        moved = float(np.hypot(*(P[-1] - P[max(0, len(P) - 30)])))
        off = float(np.min(np.hypot(*(route - p).T)))
        out.append({"t": run["t"], "run": run.get("run"), "route": run.get("route", "route"), "hole": k,
                    "edge_mm": 1000 * edge[k], "progress": run["progress"], "speed_mm_s": 1000 * speed,
                    "moved_last_1s_mm": 1000 * moved, "off_route_mm": 1000 * off,
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
    F = falls_from_runs(files, holes, route)
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
