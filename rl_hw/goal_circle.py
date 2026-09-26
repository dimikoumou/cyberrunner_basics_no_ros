"""
Find the red goal disc (or outline circle) drawn on the plate's paper and return
it in the same plate frame (meters) that HardwarePlateEnv's xb/yb use.

Shared by measure_goal_circle.py (manual check) and pd_balance.py (automatic
goal at startup, so swapping the paper sheet needs no code edits).
"""
import numpy as np
import cv2

# Hue is the reliable separator: the pen's red sampled at hue ~175-177, the wooden
# frame (which a loose "R > G,B" test also matches) at ~16-19.
RED_MIN_HUE = 165
RED_MIN_SAT = 40
MIN_RED_PIXELS = 20
RADIUS_RANGE_M = (0.005, 0.09)   # plausible goal radius (small dots down to ~1 cm across are fine)
MAX_CENTER_SPREAD_M = 0.004      # per-frame centre estimates must agree this well
BALL_CLEAR_PX = 10               # ball must be this far outside the circle's edge (640x360 px) for a clean view
LAST_GOAL_PATH = __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", "last_goal.json"))


def _detect_once(env, frame):
    """One frame -> (center_M(2,), radius_m, (cx_px, cy_px, r_px)) or None."""
    measurements = env.pipeline.measurements
    plate_pose = measurements.plate_pose
    # Restrict to the valid plate area (excludes most of the frame; the mask still
    # reaches the outer wood, hence the hue test).
    valid = measurements.mask[:, :, 0] > 0
    # Only inside the paper: the quad spanned by the 4 inner plate markers (tracked
    # every frame), shrunk 4% toward its centre. The wooden frame edges produce
    # faint reddish speckles (hue >165) that out-sized small goal dots otherwise.
    corners = measurements.detector.corners
    if corners is not None:
        pts = np.asarray(corners, dtype=np.float64)[:, ::-1]  # (row, col) -> (x, y)
        ctr = pts.mean(axis=0)
        quad = (ctr + (pts - ctr) * 0.96).astype(np.int32)
        paper = np.zeros(valid.shape, np.uint8)
        cv2.fillConvexPoly(paper, cv2.convexHull(quad), 1)
        valid &= paper.astype(bool)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    reddish = ((hue > RED_MIN_HUE) & (sat > RED_MIN_SAT) & valid).astype(np.uint8)
    if reddish.sum() < MIN_RED_PIXELS:
        return None
    # A hand-drawn line has small gaps -> dilate to merge it into one component.
    dilated = cv2.dilate(reddish, np.ones((9, 9), np.uint8), iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if n < 2:
        return None
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    component = (labels == biggest) & (reddish > 0)
    # Fit the component's OUTER boundary: correct for a filled disc (fitting every
    # red pixel of a filled disc badly underestimates the radius) and, for an
    # outline, just its outer edge. The ball sitting on the disc only punches a
    # hole inside it, which RETR_EXTERNAL ignores.
    closed = cv2.morphologyEx(component.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2).astype(np.float64)
    seg = np.vstack([hull, hull[:1]])
    outer = np.vstack([np.linspace(seg[i], seg[i + 1], max(2, int(np.hypot(*(seg[i + 1] - seg[i]))) + 1))[:-1]
                       for i in range(len(hull))])
    xs, ys = outer[:, 0], outer[:, 1]
    # least-squares circle fit: (x-cx)^2 + (y-cy)^2 = r^2, linearized
    A = np.column_stack([2 * xs, 2 * ys, np.ones(len(xs))])
    (cx, cy, c), *_ = np.linalg.lstsq(A, xs**2 + ys**2, rcond=None)
    r_px = float(np.sqrt(c + cx**2 + cy**2))
    pts_undist = plate_pose.undistort_points(np.array([[cy, cx], [cy, cx + r_px]]))  # (row, col)
    center = measurements.ball_pos_backproject(pts_undist[0], plate_pose.K, plate_pose.T__C_M)
    edge = measurements.ball_pos_backproject(pts_undist[1], plate_pose.K, plate_pose.T__C_M)
    radius = float(np.hypot(edge[0] - center[0], edge[1] - center[1]))
    # A ball sitting on (or touching) a small goal hides part of it and biases the
    # fit (a 9.0 mm dot read as 5.9-7.4 mm, centre off by 3 mm): flag such views.
    occluded = False
    ball_px = getattr(env.pipeline.measurements.detector, "ball_pos", None)
    if ball_px is not None and np.all(np.isfinite(ball_px)):
        occluded = np.hypot(ball_px[1] - cx, ball_px[0] - cy) < r_px + BALL_CLEAR_PX
    return np.array(center[:2], dtype=float), radius, (float(cx), float(cy), r_px), occluded


def detect_on_frame(env, frame):
    """Single-frame check used while balancing (live goal tracking): `frame` must
    be the frame the env's current plate pose was estimated from (env._last_frame).
    Returns (center(2,), radius) or None if no plausible circle is visible (e.g.
    mid sheet-swap, hands in view)."""
    try:
        r = _detect_once(env, frame)
    except Exception:
        return None
    if r is None:
        return None
    center, radius, _, occluded = r
    if occluded:
        return None  # live tracking only trusts unobscured views
    if not (RADIUS_RANGE_M[0] <= radius <= RADIUS_RANGE_M[1]):
        return None
    if abs(center[0]) > env._x_half or abs(center[1]) > env._y_half:
        return None
    return center, radius


def detect_goal_circle(env, n_frames=9):
    """Median over several frames. Returns dict(center, radius, px, frame) or raises
    RuntimeError with a clear message if no plausible disc is found."""
    results, frame = [], None
    for _ in range(n_frames * 3):
        # Plate pose and red pixels must come from the SAME frame: if the plate
        # moves between two frames, back-projecting frame N+1's red disc with frame
        # N's pose is wrong (a 275 mm centre spread was seen that way at startup).
        frame = env._grab_frame()
        if frame is None:
            continue
        try:
            _, _, (alpha, beta), _, _ = env.pipeline.estimate(frame)
        except Exception:
            continue
        if max(abs(np.degrees(alpha)), abs(np.degrees(beta))) > 30.0:
            continue  # lost plate-marker tracking on this frame -> pose unusable
        r = _detect_once(env, frame)
        if r is not None:
            results.append(r)
        if len(results) >= n_frames:
            break
    if len(results) < max(3, n_frames // 2):
        raise RuntimeError(f"goal circle: only found red in {len(results)} frames -- is a red "
                           f"circle drawn on the sheet and visible to the camera?")
    clean = [r for r in results if not r[3]]
    source = "clean"
    if len(clean) >= 3:
        results = clean
    else:
        # Ball is sitting on the goal (typical when restarting where it balanced).
        # Reuse the last clean detection if the visible red lies inside it (same
        # sheet); otherwise take the occluded fit as provisional -- live tracking
        # refines it once the ball moves off.
        occ_c = np.median(np.array([r[0] for r in results]), axis=0)
        last = load_last_goal()
        if last is not None and np.hypot(*(occ_c - np.array(last["center"]))) < last["radius"] + 0.003:
            return {"center": tuple(last["center"]), "radius": last["radius"], "px": None, "frame": frame,
                    "n_frames": 0, "spread_m": 0.0, "source": "last_goal"}
        source = "occluded"
    centers = np.array([r[0] for r in results])
    radii = np.array([r[1] for r in results])
    center, radius = np.median(centers, axis=0), float(np.median(radii))
    spread = float(np.max(np.linalg.norm(centers - center, axis=1)))
    if spread > MAX_CENTER_SPREAD_M:
        raise RuntimeError(f"goal circle: per-frame centres disagree by {spread * 1000:.1f} mm "
                           f"(> {MAX_CENTER_SPREAD_M * 1000:.0f} mm) -- red detection is unstable")
    if not (RADIUS_RANGE_M[0] <= radius <= RADIUS_RANGE_M[1]):
        raise RuntimeError(f"goal circle: radius {radius * 1000:.1f} mm is outside the plausible "
                           f"{RADIUS_RANGE_M[0] * 1000:.0f}-{RADIUS_RANGE_M[1] * 1000:.0f} mm")
    if abs(center[0]) > env._x_half or abs(center[1]) > env._y_half:
        raise RuntimeError(f"goal circle: centre ({center[0]:.3f}, {center[1]:.3f}) is off the plate "
                           f"(+-{env._x_half:.3f}, +-{env._y_half:.3f})")
    px = tuple(np.median(np.array([r[2] for r in results]), axis=0))
    out = {"center": (float(center[0]), float(center[1])), "radius": radius,
           "px": px, "frame": frame, "n_frames": len(results), "spread_m": spread, "source": source}
    if source == "clean":
        save_last_goal(out["center"], radius)
    return out


def load_last_goal():
    try:
        with open(LAST_GOAL_PATH) as f:
            d = __import__("json").load(f)
        return {"center": (float(d["center"][0]), float(d["center"][1])), "radius": float(d["radius"])}
    except (OSError, ValueError, KeyError):
        return None


def save_last_goal(center, radius):
    try:
        with open(LAST_GOAL_PATH, "w") as f:
            __import__("json").dump({"center": [float(center[0]), float(center[1])], "radius": float(radius)}, f)
    except OSError:
        pass


def save_debug(result, path):
    if result.get("px") is None or result.get("frame") is None:
        return
    debug = result["frame"].copy()
    cx, cy, r = result["px"]
    cv2.circle(debug, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 1)
    cv2.drawMarker(debug, (int(round(cx)), int(round(cy))), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
    cv2.imwrite(path, debug)
