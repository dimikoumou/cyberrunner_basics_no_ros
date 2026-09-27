"""
Red line on the sheet -> ordered path for the ball to follow (2026-09-26).

detect_line(env, frame): finds the red stroke on the paper, thins it to a 1-pixel
skeleton, takes the longest route through it (so small spurs / pen blobs are
ignored), converts it to plate coordinates and resamples it every SPACING_M.
Works for open lines and closed loops.

LineFollower: "carrot" guidance -- the target is a point LOOKAHEAD_M ahead of the
ball's closest point on the path; progress only moves forward (the ball can't
jump back along the route), and if the ball strays far off the line it is first
brought back to the nearest point.
"""
from collections import deque

import cv2
import numpy as np

from goal_circle import red_mask, pixel_to_plate

SPACING_M = 0.005
LOOKAHEAD_M = 0.025
REJOIN_M = 0.04
MIN_LENGTH_M = 0.05


def skeletonize(mask):
    """Zhang-Suen thinning (vectorised): a connected 1-pixel skeleton of a 0/1 mask.
    (A plain morphological skeleton left gaps/stubs on thick curved strokes, which
    broke closed loops into short open pieces.)"""
    img = np.pad((mask > 0).astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = img
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(n.astype(np.int32) for n in nb)
            seq = nb + [p2]
            A = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.int32) for k in range(8))
            c = P[1:-1, 1:-1] == 1
            if step == 0:
                rm = c & (B >= 2) & (B <= 6) & (A == 1) & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                rm = c & (B >= 2) & (B <= 6) & (A == 1) & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
    return img[1:-1, 1:-1]


def _neighbours(p, pts):
    r, c = p
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr or dc) and (r + dr, c + dc) in pts:
                yield (r + dr, c + dc)


def _bfs(start, pts):
    prev, dist, q = {start: None}, {start: 0}, deque([start])
    while q:
        p = q.popleft()
        for n in _neighbours(p, pts):
            if n not in dist:
                dist[n], prev[n] = dist[p] + 1, p
                q.append(n)
    far = max(dist, key=dist.get)
    return far, prev


def longest_path(skel, closed=None):
    """Ordered (row, col) pixels of the longest route through a skeleton, and
    whether the stroke is a closed loop (given, or: no end points)."""
    pts = set(zip(*np.nonzero(skel)))
    if not pts:
        return [], False
    ends = [p for p in pts if sum(1 for _ in _neighbours(p, pts)) == 1]
    if closed is None:
        closed = len(ends) == 0
    if closed:
        # open the loop at one pixel (with its 8 neighbours) so the longest-path
        # search runs once around it; stubs off the loop are left out by the search
        cut = max(pts, key=lambda p: sum(1 for _ in _neighbours(p, pts)) == 2)
        pts = pts - {cut} - set(_neighbours(cut, pts))
    start = ends[0] if ends else next(iter(pts))
    a, _ = _bfs(start, pts)           # farthest from an arbitrary point...
    b, prev = _bfs(a, pts)            # ...then farthest from that: the diameter
    path, p = [], b
    while p is not None:
        path.append(p)
        p = prev[p]
    return path, closed


def image_path(mask):
    """Largest red stroke in a mask -> (ordered pixel path (row, col), closed)."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return [], False
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    stroke = (labels == biggest).astype(np.uint8)
    # closed loop <=> the stroke encloses a hole (robust to little stubs on the line)
    contours, hier = cv2.findContours(stroke, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    closed = hier is not None and any(h[3] >= 0 and cv2.contourArea(c) > 200
                                      for c, h in zip(contours, hier[0]))
    return longest_path(skeletonize(stroke), closed)


def resample(path_m, spacing):
    seg = np.hypot(*np.diff(path_m, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return path_m
    t = np.arange(0.0, s[-1], spacing)
    return np.column_stack([np.interp(t, s, path_m[:, 0]), np.interp(t, s, path_m[:, 1])])


def detect_line(env, frame):
    """-> {"path": (N,2) plate metres, "path_px": (N,2) (x=col, y=row), "closed": bool,
    "length": m} or None if there is no plausible red line."""
    mask = red_mask(env, frame)
    if mask is None:
        return None
    pix, closed = image_path(mask)
    if len(pix) < 10:
        return None
    pix = pix[::2]  # every 2nd skeleton pixel is plenty before resampling
    pts, pts_px = [], []
    for r, c in pix:
        xy = pixel_to_plate(env, r, c)
        if xy is not None:
            pts.append(xy)
            pts_px.append((c, r))
    if len(pts) < 5:
        return None
    path = resample(np.array(pts), SPACING_M)
    length = float(np.sum(np.hypot(*np.diff(path, axis=0).T)))
    if length < MIN_LENGTH_M:
        return None
    return {"path": path, "path_px": np.array(pts_px, dtype=float), "closed": closed, "length": length}


class LineFollower:
    def __init__(self, path, closed, ball_xy):
        self.path = np.asarray(path)
        self.closed = closed
        # open line: start from the end nearer to the ball
        if not closed and np.hypot(*(self.path[-1] - ball_xy)) < np.hypot(*(self.path[0] - ball_xy)):
            self.path = self.path[::-1]
        self.n = len(self.path)
        self.i = int(np.argmin(np.hypot(*(self.path - np.asarray(ball_xy)).T))) if closed else 0
        self.look = max(1, int(round(LOOKAHEAD_M / SPACING_M)))

    def target(self, ball_xy):
        """-> (target xy, distance from the line (m), finished)"""
        ball_xy = np.asarray(ball_xy)
        d_all = np.hypot(*(self.path - ball_xy).T)
        off = float(d_all.min())
        if off > REJOIN_M:  # far off the line: go back to the nearest point first
            self.i = int(np.argmin(d_all)) if self.closed else max(self.i, int(np.argmin(d_all)))
            return self.path[self.i], off, False
        # progress only forward, within a window ahead of the current index
        win = [(self.i + k) % self.n if self.closed else min(self.i + k, self.n - 1) for k in range(0, 3 * self.look)]
        j = win[int(np.argmin(d_all[win]))]
        self.i = j
        if self.closed:
            return self.path[(j + self.look) % self.n], off, False
        k = min(j + self.look, self.n - 1)
        return self.path[k], off, k == self.n - 1 and np.hypot(*(self.path[-1] - ball_xy)) < 0.01
