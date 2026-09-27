"""
Holes in the paper (2026-09-27): find them, keep targets and moves away from them,
and know when the ball fell into one (-> the elevator reloads it).

A hole shows as a small, round, very dark blob on the white paper (the plate's own
hole underneath; V ~0 against the paper's ~200). The ball is excluded by position.
Plate frame and helpers are the same as for the red goal (goal_circle.py).
"""
import json
import os

import cv2
import numpy as np

from goal_circle import pixel_to_plate

DARK_V = 70                   # HSV value below this = hole (paper ~200, pen lines/grid > 100)
MIN_AREA_PX, MAX_AREA_PX = 40, 2500   # 640x360 frame; the plate hole is ~190 px
MIN_FILL = 0.55               # area / bounding-box area (a disc is ~0.79)
BALL_EXCLUDE_PX = 14
HOLE_MARGIN_M = 0.008         # keep-out clearance beyond the hole's edge (the ball falls once its centre is over the hole)
HOLES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "holes.json"))


def detect_holes(env, frame):
    """-> list of {"center": (x, y) m, "radius": m, "px": (col, row, r_px)} on the paper."""
    m = env.pipeline.measurements
    valid = m.mask[:, :, 0] > 0
    corners = m.detector.corners
    if corners is None:
        return []
    pts = np.asarray(corners, dtype=np.float64)[:, ::-1]
    ctr = pts.mean(axis=0)
    quad = (ctr + (pts - ctr) * 0.96).astype(np.int32)
    paper = np.zeros(valid.shape, np.uint8)
    cv2.fillConvexPoly(paper, cv2.convexHull(quad), 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = ((hsv[:, :, 2] < DARK_V) & valid & (paper > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ball_px = getattr(m.detector, "ball_pos", None)
    n, _, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    holes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (MIN_AREA_PX <= area <= MAX_AREA_PX) or area < MIN_FILL * w * h or max(w, h) > 2.2 * min(w, h):
            continue
        col, row = cents[i]
        if ball_px is not None and np.all(np.isfinite(ball_px)) and np.hypot(row - ball_px[0], col - ball_px[1]) < BALL_EXCLUDE_PX:
            continue
        r_px = float(np.sqrt(area / np.pi))
        c = pixel_to_plate(env, row, col)
        e = pixel_to_plate(env, row, col + r_px)
        if c is None or e is None:
            continue
        holes.append({"center": (float(c[0]), float(c[1])), "radius": float(np.hypot(*(e - c))),
                      "px": (float(col), float(row), r_px)})
    return holes


def detect_holes_stable(env, n_frames=8):
    """holes seen in most of n fresh frames, positions averaged (the ball rolling over
    a hole, or one bad frame, must not add or drop one)"""
    seen = []
    for _ in range(n_frames):
        env._read_state()
        f = getattr(env, "_last_frame", None)
        if f is not None:
            seen.append(detect_holes(env, f))
    if not seen:
        return []
    groups = []
    for hs in seen:
        for h in hs:
            for g in groups:
                if np.hypot(*(np.subtract(g[0]["center"], h["center"]))) < 0.01:
                    g.append(h)
                    break
            else:
                groups.append([h])
    out = []
    for g in groups:
        if len(g) >= max(2, len(seen) // 2):
            out.append({"center": tuple(np.mean([h["center"] for h in g], axis=0)),
                        "radius": float(np.median([h["radius"] for h in g])),
                        "px": tuple(np.mean([h["px"] for h in g], axis=0))})
    return out


def save_holes(holes, path=HOLES_PATH):
    try:
        with open(path, "w") as f:
            json.dump(holes, f)
    except OSError:
        pass


def load_holes(path=HOLES_PATH):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def near_hole(holes, xy, extra=0.0):
    """the hole whose keep-out zone (+extra) contains xy, else None"""
    for h in holes:
        if np.hypot(xy[0] - h["center"][0], xy[1] - h["center"][1]) < h["radius"] + HOLE_MARGIN_M + extra:
            return h
    return None


def detour(holes, start, goal, ring=1.35, step_deg=45.0):
    """Via-points so the run start -> goal clears every keep-out zone (radius R):
    for a hole whose zone the straight segment cuts, go round it the shorter way on
    an arc of radius ring*R, points <= step_deg apart (each chord then stays
    ring*R*cos(step/2) > R from the hole's centre)."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    vias = []
    for h in holes:
        c = np.asarray(h["center"], float)
        R = h["radius"] + HOLE_MARGIN_M
        d = goal - start
        L2 = float(np.dot(d, d))
        if L2 < 1e-12:
            continue
        t = float(np.clip(np.dot(c - start, d) / L2, 0.0, 1.0))
        if np.hypot(*(start + t * d - c)) >= R:
            continue
        th_s = np.arctan2(*(start - c)[::-1])
        th_g = np.arctan2(*(goal - c)[::-1])
        dth = (th_g - th_s + np.pi) % (2 * np.pi) - np.pi
        if abs(abs(dth) - np.pi) < 1e-3:
            dth = np.pi
        n = max(1, int(np.ceil(abs(np.degrees(dth)) / step_deg)))
        rho = ring * R
        # skip arc points that sit right at the start/goal side (they'd double back)
        pts = [c + rho * np.array([np.cos(th_s + dth * k / n), np.sin(th_s + dth * k / n)]) for k in range(n + 1)]
        if np.hypot(*(start - c)) >= rho * 0.95:
            pts = pts[1:]
        if np.hypot(*(goal - c)) >= rho * 0.95 and pts:
            pts = pts[:-1]
        vias += [(t, p) for p in pts]
    return [v for _, v in vias]


def segment_hits_hole(holes, a, b):
    """True if the segment a-b passes through a keep-out zone"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L2 = float(np.dot(d, d))
    for h in holes:
        c = np.asarray(h["center"], float)
        t = 0.0 if L2 < 1e-12 else float(np.clip(np.dot(c - a, d) / L2, 0.0, 1.0))
        if np.hypot(*(a + t * d - c)) < h["radius"] + HOLE_MARGIN_M:
            return True
    return False
