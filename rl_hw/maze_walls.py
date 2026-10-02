#!/usr/bin/env python3
"""
Wall map for any maze board (2026-09-28): a dense occupancy grid of the walls in plate
coordinates (1 mm cells), from the full-resolution capture (maze_capture.py). Replaces the
wall outlines of maze_route.py for planning: those missed most bars (the glossy black bars'
white highlight split them in a dark-pixel threshold) and had no outer boundary, so a planner
walked round the outside of the maze.

Walls = dark pixels (black bars) with the highlight gaps closed along the bar, minus the hole
discs, minus thin grey print (the route line, numbers); plus the board's own edge as a wall.
Output: maze/walls_grid.npz (occupancy, origin, cell) and maze/walls_check.png.

  python3 maze_walls.py [maze_dir]
"""
import json
import os
import sys

import cv2
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maze")
CELL = 0.001
DARK = 60                 # bars are near-black; bar shadows and the printed line are lighter


def main():
    full = cv2.imread(os.path.join(D, "maze_full.png"))
    g = np.load(os.path.join(D, "pixel_to_plate_grid.npz"))["grid"]        # (180, 320, 2), every 6 full-res px
    cap = json.load(open(os.path.join(D, "capture.json")))
    route = json.load(open(os.path.join(D, "route.json")))
    H, W = full.shape[:2]
    S = W / 640.0
    corners = np.array(cap["inner_corners_row_col"])[:, ::-1] * S
    ctr = corners.mean(axis=0)
    quad = (ctr + (corners - ctr) * 0.985).astype(np.int32)
    board = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(board, cv2.convexHull(quad), 1)
    gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(full, cv2.COLOR_BGR2HSV)
    dark = ((gray < DARK) & (board > 0)).astype(np.uint8)
    # glossy bars: close the highlight stripe (a few px wide, along the bar)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    # thin print (line, numbers) goes; bars (~20 px thick) stay
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    # hole discs are dark too: remove them (a hole is not a wall)
    for h in route["holes"]:
        x, y = h["px"][0], h["px"][1]
        if x < 700:                                 # stored in 640-wide pipeline pixels
            x, y = x * S, y * S
        r_px = h["radius"] * 1000 * 3.9 * 1.25      # ~3.9 px per mm at full res, +25 %
        cv2.circle(dark, (int(x), int(y)), int(r_px), 0, -1)
    # round dark blobs are holes the hole list missed (the board has 60, maze_route found 41):
    # they become holes, not walls (the route passes a few mm from them legitimately)
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = np.zeros(n, bool)
    extra_holes = []
    for k in range(1, n):
        w_, h_, a_ = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT], st[k, cv2.CC_STAT_AREA]
        r_eq = np.sqrt(a_ / np.pi)
        roundish = 0.75 < w_ / max(h_, 1) < 1.33 and a_ > 0.6 * w_ * h_ and 18 < r_eq < 48
        if roundish:
            extra_holes.append((cen[k][0], cen[k][1], r_eq))
            continue
        keep[k] = a_ > 400 and max(w_, h_) > 45
    wall_px = keep[lab]
    # the board edge is a wall: a band just inside the quad
    edge = np.zeros((H, W), np.uint8)
    cv2.polylines(edge, [cv2.convexHull(quad)], True, 1, 6)
    wall_px |= (edge > 0) & (board > 0)
    # pixels -> plate metres (bilinear on the lookup grid), rasterised into 1 mm cells
    rr, cc = np.nonzero(wall_px[::2, ::2])
    rr, cc = rr * 2.0, cc * 2.0
    gr, gc = rr / S / 2.0, cc / S / 2.0
    r0 = np.clip(np.floor(gr).astype(int), 0, g.shape[0] - 2)
    c0 = np.clip(np.floor(gc).astype(int), 0, g.shape[1] - 2)
    fr, fc = (gr - r0)[:, None], (gc - c0)[:, None]
    P = (g[r0, c0] * (1 - fr) * (1 - fc) + g[r0 + 1, c0] * fr * (1 - fc) + g[r0, c0 + 1] * (1 - fr) * fc
         + g[r0 + 1, c0 + 1] * fr * fc)
    P = P[np.all(np.isfinite(P), axis=1)]
    lo = np.array([-0.16, -0.14])
    nx, ny = int(0.32 / CELL), int(0.28 / CELL)
    occ = np.zeros((ny, nx), np.uint8)
    ij = np.floor((P - lo) / CELL).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    occ[ij[ok, 1], ij[ok, 0]] = 1
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # everything outside the board edge is blocked too
    outside = np.ones((ny, nx), np.uint8)
    qm = np.array([[(p[0] - lo[0]) / CELL, (p[1] - lo[1]) / CELL] for p in
                   [P[np.argmin(P @ v)] for v in ([1, 1], [-1, 1], [-1, -1], [1, -1])]], np.int32)
    cv2.fillConvexPoly(outside, cv2.convexHull(qm), 0)
    occ |= outside
    # the extra holes in plate metres (centre + radius from an edge point)
    def px2plate(x, y):
        gr, gc = y / S / 2.0, x / S / 2.0
        r0_, c0_ = int(np.floor(gr)), int(np.floor(gc))
        fr_, fc_ = gr - r0_, gc - c0_
        return (g[r0_, c0_] * (1 - fr_) * (1 - fc_) + g[r0_ + 1, c0_] * fr_ * (1 - fc_)
                + g[r0_, c0_ + 1] * (1 - fr_) * fc_ + g[r0_ + 1, c0_ + 1] * fr_ * fc_)
    holes_extra = []
    for x, y, r in extra_holes:
        c, e = px2plate(x, y), px2plate(x + r, y)
        if np.all(np.isfinite(c)) and np.all(np.isfinite(e)):
            holes_extra.append([float(c[0]), float(c[1]), float(np.hypot(*(e - c)))])
    # signed distance to the nearest wall (m, > 0 in free space), lightly smoothed: the maze
    # simulator (rl_sim/maze_world.py) needs it and its environment has no OpenCV / SciPy
    d_out = cv2.distanceTransform((1 - occ).astype(np.uint8), cv2.DIST_L2, 5) * CELL
    d_in = cv2.distanceTransform(occ.astype(np.uint8), cv2.DIST_L2, 5) * CELL
    sdf = cv2.GaussianBlur((d_out - d_in).astype(np.float32), (0, 0), 1.0)
    np.savez(os.path.join(D, "walls_grid.npz"), occ=occ, lo=lo, cell=CELL, holes_extra=np.array(holes_extra), sdf=sdf)
    print(f"{len(holes_extra)} extra holes found (not in route.json)")
    chk = full.copy()
    chk[wall_px] = (0.4 * chk[wall_px] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
    cv2.imwrite(os.path.join(D, "walls_check.png"), chk)
    print(f"wall map {ny}x{nx} cells, {occ.mean() * 100:.0f} % blocked -> {D}/walls_grid.npz")


if __name__ == "__main__":
    main()
