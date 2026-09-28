#!/usr/bin/env python3
"""
Extract the maze's printed route and its holes from the full-resolution capture
(maze_capture.py) -> plate metres, for practising the route on white paper (2026-09-27).

Route: the thin (~4 px at 1920x1080) dark-grey line. Dark pixels minus everything thick
(walls, holes: morphological opening) leaves the line and the printed numbers; the largest
connected piece is the route; Zhang-Suen skeleton + longest path gives it in order; it is
oriented to start at the start arrow (top middle). Holes: large dark round blobs.
Outputs: maze/route.json (route + holes in plate metres) and maze/route_check.png.

  python3 maze_route.py [maze_dir]
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from line_path import skeletonize, longest_path  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maze")
full = cv2.imread(os.path.join(D, "maze_full.png"))
g = np.load(os.path.join(D, "pixel_to_plate_grid.npz"))
cap = json.load(open(os.path.join(D, "capture.json")))
H, W = full.shape[:2]
S = W / 640.0                                   # full-res -> pipeline scale (3)

# the board: inside the 4 inner plate markers (pipeline coords -> full res), shrunk 3 %
corners = np.array(cap["inner_corners_row_col"])[:, ::-1] * S          # (x, y) full res
ctr = corners.mean(axis=0)
quad = (ctr + (corners - ctr) * 0.97).astype(np.int32)
board = np.zeros((H, W), np.uint8)
cv2.fillConvexPoly(board, cv2.convexHull(quad), 1)

gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
dark = ((gray < 170) & (board > 0)).astype(np.uint8)
thick = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
thick = cv2.dilate(thick, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
thin = dark & (1 - thick)
thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
order = np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1
route_mask = (lab == order[0]).astype(np.uint8)
print("route component: %d px (next largest: %s)" % (st[order[0], cv2.CC_STAT_AREA],
                                                      list(st[order[1:4], cv2.CC_STAT_AREA])))
path_px, _ = longest_path(skeletonize(route_mask), closed=False)
path_px = np.array(path_px, dtype=float)                                 # (row, col) full res

# start at the start arrow: the end nearer the board's top middle
top_mid = np.array([quad[:, 1].min(), quad[:, 0].mean()])
if np.hypot(*(path_px[-1] - top_mid)) < np.hypot(*(path_px[0] - top_mid)):
    path_px = path_px[::-1]


def to_plate(rc_full):
    """full-res (row, col) -> plate metres via the capture's lookup grid (bilinear)"""
    r, c = rc_full[0] / S / 2.0, rc_full[1] / S / 2.0                     # grid index (every 2 px)
    r0, c0 = int(np.floor(r)), int(np.floor(c))
    fr, fc = r - r0, c - c0
    G = g["grid"]
    q = (G[r0, c0] * (1 - fr) * (1 - fc) + G[r0 + 1, c0] * fr * (1 - fc) + G[r0, c0 + 1] * (1 - fr) * fc
         + G[r0 + 1, c0 + 1] * fr * fc)
    return q


route = np.array([to_plate(p) for p in path_px[::3]])
route = route[np.all(np.isfinite(route), axis=1)]
seg = np.hypot(*np.diff(route, axis=0).T)
L = float(seg.sum())

# holes: round discs, black or grey (a hole you can see into shows the lit floor below). Hough
# circles on the blurred image, kept if the disc is clearly darker than the board around it and
# not a wall (walls are long; their ends are rounded but the disc test fails on them)
blur = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=38, param1=90, param2=24,
                           minRadius=17, maxRadius=30)
holes = []
bg = cv2.medianBlur(gray, 61)
for x, y, r in (circles[0] if circles is not None else []):
    xi, yi, ri = int(x), int(y), int(r)
    if board[yi, xi] == 0:
        continue
    disc = np.zeros_like(gray)
    cv2.circle(disc, (xi, yi), max(3, int(0.7 * ri)), 1, -1)
    inner = float(np.median(gray[disc > 0]))
    ring = np.zeros_like(gray)
    cv2.circle(ring, (xi, yi), int(1.5 * ri), 1, 4)
    around = float(np.median(gray[ring > 0]))
    if inner > 0.72 * around:                     # not clearly darker than its surroundings
        continue
    if np.mean(dark[disc > 0]) < 0.6:               # mostly not dark inside -> not a hole
        continue
    c = to_plate((y, x))
    e = to_plate((y, x + r))
    if np.all(np.isfinite(c)) and np.all(np.isfinite(e)):
        holes.append({"center": [float(c[0]), float(c[1])], "radius": float(np.hypot(*(e - c))),
                      "px": [float(x), float(y)]})
dmin = [float(np.min(np.hypot(*(route - np.array(h["center"])).T))) for h in holes]
print(f"route: {len(route)} points, {L * 100:.0f} cm long; {len(holes)} holes; "
      f"closest approach route -> hole edge: {min(d - h['radius'] for d, h in zip(dmin, holes)) * 1000:.1f} mm")
json.dump({"route_m": route.tolist(), "length_m": L, "holes": holes,
           "route_to_hole_edge_mm": [(d - h["radius"]) * 1000 for d, h in zip(dmin, holes)]},
          open(os.path.join(D, "route.json"), "w"), indent=1)

chk = full.copy()
cv2.polylines(chk, [path_px[:, ::-1].astype(np.int32)], False, (0, 200, 0), 3, cv2.LINE_AA)
cv2.circle(chk, (int(path_px[0, 1]), int(path_px[0, 0])), 14, (0, 0, 255), 3)          # start: red
cv2.circle(chk, (int(path_px[-1, 1]), int(path_px[-1, 0])), 14, (255, 0, 0), 3)        # end: blue
for h in holes:
    cv2.circle(chk, (int(h["px"][0]), int(h["px"][1])), 8, (0, 140, 255), -1)
cv2.imwrite(os.path.join(D, "route_check.png"), chk)
print("wrote", os.path.join(D, "route.json"), "and route_check.png")
