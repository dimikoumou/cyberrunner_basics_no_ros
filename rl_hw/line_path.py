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


V_LINE, A_LINE = 0.04, 0.08    # m/s, m/s^2 along the route
LAG_MAX = 0.025                # reference waits if the ball lags more than this (m)


A_LAT = 0.03      # m/s^2: lateral (centripetal) acceleration allowed in bends (~0.3 deg of tilt)
V_CORNER_MIN = 0.006   # m/s: never plan slower than this (a stopped reference just waits)


class PathTracker:
    """Smooth route following (2026-09-26): instead of chasing a point ahead (the
    ball crept along in stick-slip hops at ~5 mm/s), a reference point moves along
    the route with a speed profile -- ramp up to V_LINE, and on an open route brake
    to a stop at the end -- and the controller tracks it with the planned
    acceleration fed forward (tangential + centripetal on curves) plus rolling
    friction. If the ball falls behind by more than LAG_MAX the reference waits."""

    def __init__(self, path, closed, ball_xy, v=V_LINE, a=A_LINE, keep_direction=False, ref_offset=None,
                 speed_scale=None, los_tol=None):
        """los_tol: (maze) keep the reference in straight-line sight of the ball -- the
        chord ball -> reference may leave the route by at most los_tol (m), so the target
        never sits past a corner, behind a wall the ball would be pushed into.
        keep_direction: follow the path in the given order (drawn paths start at the
        first clicked point); otherwise an open line starts at the end nearer the ball."""
        path = np.asarray(path, dtype=float)
        ball_xy = np.asarray(ball_xy, dtype=float)
        if (not closed and not keep_direction
                and np.hypot(*(path[-1] - ball_xy)) < np.hypot(*(path[0] - ball_xy))):
            path = path[::-1]
        self.closed, self.v_max, self.a_max = closed, v, a
        self.los_tol = los_tol
        pts = np.vstack([path, path[:1]]) if closed else path
        seg = np.hypot(*np.diff(pts, axis=0).T)
        self.S = np.concatenate([[0.0], np.cumsum(seg)])
        self.pts, self.L = pts, float(self.S[-1])
        # tangents and curvature vectors (dT/ds), lightly smoothed
        d = np.gradient(pts, self.S, axis=0)
        T = d / np.maximum(np.hypot(*d.T), 1e-9)[:, None]
        k = np.gradient(T, self.S, axis=0)
        ker = np.ones(5) / 5
        self.T = T
        self.K = np.column_stack([np.convolve(k[:, 0], ker, "same"), np.convolve(k[:, 1], ker, "same")])
        self.s = self._project(ball_xy, None) if closed else 0.0
        self.v = 0.0
        # corner-aware speed profile (2026-09-27, for maze tracing): in a bend of curvature
        # kappa the speed is limited to sqrt(A_LAT / kappa) (a sharp polyline corner -> near
        # stop), and a backward pass brakes early enough (a_max) for every bend ahead
        kap = np.hypot(*self.K.T)
        vp = np.minimum(self.v_max, np.sqrt(A_LAT / np.maximum(kap, 1e-6)))
        if speed_scale is not None:
            # learned slow zones (maze practice): per route point factor <= 1
            vp = vp * np.asarray(speed_scale, dtype=float)[:len(vp)]
        vp = np.maximum(vp, V_CORNER_MIN)
        ds = np.diff(self.S)
        for i in range(len(vp) - 2, -1, -1):
            vp[i] = min(vp[i], np.sqrt(vp[i + 1] ** 2 + 2 * self.a_max * ds[i]))
        if closed:                      # the same once more across the wrap-around
            vp[-1] = min(vp[-1], vp[0])
            for i in range(len(vp) - 2, -1, -1):
                vp[i] = min(vp[i], np.sqrt(vp[i + 1] ** 2 + 2 * self.a_max * ds[i]))
        self.v_prof = vp
        self.offs = []                  # distance ball <-> line while following (accuracy report)
        # iterative learning control (maze practice): the REFERENCE is the route shifted by a
        # learned per-point offset; distances are still measured to the true route
        self.ref_pts = pts if ref_offset is None else pts + np.asarray(ref_offset, dtype=float)[:len(pts)]

    def _interp(self, arr, s):
        s = s % self.L if self.closed else min(max(s, 0.0), self.L)
        return np.array([np.interp(s, self.S, arr[:, 0]), np.interp(s, self.S, arr[:, 1])])

    def point(self, s):
        return self._interp(self.ref_pts, s)

    def true_point(self, s):
        return self._interp(self.pts, s)

    def _project(self, xy, near_s, window=0.04):
        d = np.hypot(*(self.pts - xy).T)
        if near_s is not None:
            ds = np.abs(self.S - (near_s % self.L if self.closed else near_s))
            if self.closed:
                ds = np.minimum(ds, self.L - ds)
            d = np.where(ds <= window, d, np.inf)
        return float(self.S[int(np.argmin(d))])

    def _visible_s(self, xy, s_from, s_to):
        """furthest route position in [s_from, s_to] whose chord from xy stays within
        los_tol of the route points in between"""
        i0 = int(np.searchsorted(self.S, s_from))
        i1 = int(np.searchsorted(self.S, s_to))
        best = s_from
        for j in range(i0 + 1, min(i1, len(self.pts) - 1) + 1):
            c = self.pts[j] - xy
            n = max(np.hypot(*c), 1e-9)
            q = self.pts[i0:j] - xy
            t = np.clip((q @ c) / n ** 2, 0.0, 1.0)
            dev = np.hypot(*(q - t[:, None] * c).T)
            if dev.max() > self.los_tol:
                break
            best = float(self.S[j])
        return best

    def off_line(self, xy):
        return float(np.min(np.hypot(*(self.pts - np.asarray(xy)).T)))

    def update(self, ball_xy, dt):
        """-> dict(p, v, a, t_hat, lag, off, finished) for this control step."""
        s_ball = self._project(np.asarray(ball_xy), self.s)
        lag = self.s - s_ball
        if self.closed:
            lag = (lag + self.L / 2) % self.L - self.L / 2
        v_des = min(self.v_max, float(np.interp(self.s % self.L if self.closed else self.s, self.S, self.v_prof)))
        if not self.closed:
            v_des = min(v_des, float(np.sqrt(2 * self.a_max * max(self.L - self.s, 0.0))))
        if lag > LAG_MAX:
            v_des = 0.0  # wait for the ball
        dv = float(np.clip(v_des - self.v, -self.a_max * dt, self.a_max * dt))
        self.v = max(0.0, self.v + dv)
        self.s = self.s + self.v * dt
        if not self.closed:
            self.s = min(self.s, self.L)
        if self.los_tol and not self.closed and self.s > s_ball:
            s_vis = self._visible_s(np.asarray(ball_xy, dtype=float), s_ball, self.s)
            if self.s > s_vis:          # target would be behind a corner -> pull it back
                self.s = s_vis
                self.v = min(self.v, 0.5 * V_CORNER_MIN)
        t_hat = self._interp(self.T, self.s)
        t_hat = t_hat / max(np.hypot(*t_hat), 1e-9)
        a = (dv / max(dt, 1e-3)) * t_hat + self.v ** 2 * self._interp(self.K, self.s)
        finished = (not self.closed) and self.s >= self.L - 1e-4 and self.v < 1e-3
        off = self.off_line(ball_xy)
        self.offs.append(off)
        return {"p": self.point(self.s), "v": self.v * t_hat, "a": a, "t_hat": t_hat, "lag": lag,
                "off": off, "finished": finished}

    def accuracy(self):
        """distance ball <-> line while following, in mm: median / p90 / max"""
        o = np.array(self.offs) * 1000
        if len(o) == 0:
            return None
        return {"median_mm": float(np.median(o)), "p90_mm": float(np.percentile(o, 90)),
                "max_mm": float(np.max(o)), "n": int(len(o))}
