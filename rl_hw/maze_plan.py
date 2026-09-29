#!/usr/bin/env python3
"""
Safe route planner for any maze (2026-09-28): from the holes and walls extracted from a photo
of the board (maze_route.py -> maze/route.json), plan the path from start to goal that keeps
the ball's centre as far from the holes as the corridors allow. Walls are boundaries, not
dangers (the ball may touch and use them); holes are the only thing that ends a run.

Grid search (1 mm cells, 8-neighbour Dijkstra) over the board where the ball's centre can be
(walls grown by the ball radius). The cost of a step grows sharply near a hole's edge, so the
path takes the far side of a corridor next to a hole and cuts no corners past one. The result
is smoothed, resampled at 1 mm and written in the same format as the printed route, so the
controller, the practice and the learning loop use it unchanged.

  python3 maze_plan.py [--start x,y] [--goal x,y] [--out maze/route_safe.json]
  (start/goal in mm on the plate; default: the ends of the printed route)
"""
import argparse
import heapq
import json
import os

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CELL = 0.001                       # m
BALL_R = 0.0040                    # ball centre keeps this from a wall (the ball is ~12.7 mm; a
                                   # little under its radius: extracted walls are slightly fat)
HOLE_SCALE = 0.005                 # m: how fast the hole cost falls off with distance to the edge
HOLE_W = 8.0                       # a step right at a hole's edge costs 1 + HOLE_W times more
WALL_W = 4.0                       # soft cost for a target the ball can't reach (closer than REACH_R)
REACH_R = 0.0062                   # m: ball radius -- a centre closer to a wall than this is unreachable
SMOOTH_PASSES = 25                 # moving-average passes (a wiggly reference makes the ball wiggle)
HOLE_FORBID = 0.0                  # centre may not come closer than this to a hole's edge


def plan(route_json, start=None, goal=None, corridor=None):
    r = json.load(open(route_json))
    holes = [(np.array(h["center"], float), float(h["radius"])) for h in r["holes"]]
    walls = [np.array(w, float) for w in r["walls_m"]]
    printed = np.array(r["route_m"], float)
    start = printed[0] if start is None else np.asarray(start, float)
    goal = printed[-1] if goal is None else np.asarray(goal, float)
    grid_fn = os.path.join(os.path.dirname(route_json), "walls_grid.npz")
    if os.path.exists(grid_fn):
        # dense wall map from the photo (maze_walls.py): bars + the board edge
        gz = np.load(grid_fn)
        wall, lo = gz["occ"].astype(np.uint8), gz["lo"].astype(float)
        if "holes_extra" in gz.files:
            holes = holes + [(np.array(h[:2]), float(h[2])) for h in gz["holes_extra"]]
        ny, nx = wall.shape
    else:
        allp = np.vstack(walls + [printed])
        lo = allp.min(0) - 0.01
        hi = allp.max(0) + 0.01
        nx, ny = int(np.ceil((hi[0] - lo[0]) / CELL)), int(np.ceil((hi[1] - lo[1]) / CELL))
        wall = np.zeros((ny, nx), np.uint8)
        for w in walls:
            pts = np.array([[(q[0] - lo[0]) / CELL, (q[1] - lo[1]) / CELL] for q in w], np.int32)
            cv2.fillPoly(wall, [pts], 1)
    to_ij = lambda p: (int(round((p[0] - lo[0]) / CELL)), int(round((p[1] - lo[1]) / CELL)))
    free_d = cv2.distanceTransform((1 - wall).astype(np.uint8), cv2.DIST_L2, 5) * CELL
    blocked = free_d < BALL_R
    # holes -> distance of every cell to the nearest hole EDGE
    X, Y = np.meshgrid(lo[0] + np.arange(nx) * CELL, lo[1] + np.arange(ny) * CELL)
    d_hole = np.full((ny, nx), 1.0)
    for c, rad in holes:
        d_hole = np.minimum(d_hole, np.hypot(X - c[0], Y - c[1]) - rad)
    blocked |= d_hole < HOLE_FORBID
    if corridor:
        # along the printed route: stay within `corridor` of it (the game's path, but on its
        # safest side) -- without this the planner takes the board's physical shortcuts
        line = np.zeros((ny, nx), np.uint8)
        pts = np.array([[(q[0] - lo[0]) / CELL, (q[1] - lo[1]) / CELL] for q in printed], np.int32)
        cv2.polylines(line, [pts], False, 1, 1)
        d_line = cv2.distanceTransform((1 - line).astype(np.uint8), cv2.DIST_L2, 5) * CELL
        blocked |= d_line > corridor
    cost = 1.0 + HOLE_W * np.exp(-np.maximum(d_hole, 0.0) / HOLE_SCALE)
    # the target must be where the ball's centre CAN be: prefer >= REACH_R from a wall (the ball
    # radius; the first safe route ran 2.8 mm from walls and the ball, pushed off it, fell early)
    cost += WALL_W * np.exp(-np.maximum(free_d - BALL_R, 0.0) / 0.0015) * (free_d < REACH_R)
    si, sj = to_ij(start)[::-1], to_ij(goal)[::-1]
    for (i, j) in (si, sj):                       # start/goal must be free: snap to the nearest free cell
        if blocked[i, j]:
            fi, fj = np.nonzero(~blocked)
            k = np.argmin((fi - i) ** 2 + (fj - j) ** 2)
            if (i, j) == si:
                si = (int(fi[k]), int(fj[k]))
            else:
                sj = (int(fi[k]), int(fj[k]))
    # Dijkstra
    dist = np.full((ny, nx), np.inf)
    prev = -np.ones((ny, nx, 2), np.int32)
    dist[si] = 0.0
    pq = [(0.0, si)]
    nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5)]
    while pq:
        d, (i, j) = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        if (i, j) == sj:
            break
        for di, dj, L in nb:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and not blocked[a, b]:
                nd = d + L * 0.5 * (cost[i, j] + cost[a, b])
                if nd < dist[a, b]:
                    dist[a, b] = nd
                    prev[a, b] = (i, j)
                    heapq.heappush(pq, (nd, (a, b)))
    if not np.isfinite(dist[sj]):
        raise RuntimeError("no path from start to goal -- walls grown too far, or start/goal outside the maze")
    path = [sj]
    while path[-1] != si:
        path.append(tuple(prev[path[-1]]))
    path = path[::-1]
    P = np.array([[lo[0] + j * CELL, lo[1] + i * CELL] for i, j in path])
    # smooth (moving average, ends fixed), never into a blocked cell, then resample at 1 mm
    for _ in range(SMOOTH_PASSES):
        Q = P.copy()
        Q[2:-2] = (P[:-4] + P[1:-3] + P[2:-2] + P[3:-1] + P[4:]) / 5
        # a smoothed point may not cut closer to a wall than the point it replaces
        ok = np.array([not blocked[to_ij(q)[1], to_ij(q)[0]]
                       and free_d[to_ij(q)[1], to_ij(q)[0]] >= min(REACH_R, free_d[to_ij(p)[1], to_ij(p)[0]]) - 0.0005
                       for q, p in zip(Q, P)])
        P = np.where(ok[:, None], Q, P)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    n = int(s[-1] / 0.001) + 1
    route = np.column_stack([np.interp(np.linspace(0, s[-1], n), s, P[:, k]) for k in (0, 1)])

    def clearance(R):
        return [float(1000 * (np.min(np.hypot(*(R - c).T)) - rad)) for c, rad in holes]
    out = dict(r)
    out.update(route_m=route.tolist(), length_m=float(s[-1]), route_to_hole_edge_mm=clearance(route),
               planned={"ball_r_mm": 1000 * BALL_R, "hole_scale_mm": 1000 * HOLE_SCALE, "hole_w": HOLE_W,
                        "start": start.tolist(), "goal": goal.tolist()})
    return out, clearance(printed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default=os.path.join(ROOT, "maze", "route.json"))
    ap.add_argument("--start")
    ap.add_argument("--goal")
    ap.add_argument("--corridor", type=float, default=None,
                    help="mm: stay within this of the printed route (its safest side); omit = free planning")
    ap.add_argument("--out", default=os.path.join(ROOT, "maze", "route_safe.json"))
    a = ap.parse_args()
    mm = lambda s: np.array([float(v) / 1000 for v in s.split(",")]) if s else None
    out, printed_clear = plan(a.route, mm(a.start), mm(a.goal), a.corridor / 1000 if a.corridor else None)
    json.dump(out, open(a.out, "w"))
    pc, sc = np.array(printed_clear), np.array(out["route_to_hole_edge_mm"])
    print(f"planned route {out['length_m']:.2f} m -> {a.out}")
    print(f"closest approach to a hole edge: printed {pc.min():.1f} mm, planned {sc.min():.1f} mm")
    worst = np.argsort(pc)[:8]
    print("tightest holes of the printed route (hole: printed -> planned mm):",
          ", ".join(f"{k}: {pc[k]:.1f}->{sc[k]:.1f}" for k in worst))


if __name__ == "__main__":
    main()
