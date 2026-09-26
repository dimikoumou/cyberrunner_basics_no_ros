#!/usr/bin/env python3
"""
Detects a hand-drawn red circle on the plate's paper and reports its center/radius
in the same world frame (meters) that HardwarePlateEnv's xb/yb use, so it can be
passed directly as fixed_goal / goal_tolerance to train_sac_hw.py.

Restricts the red-pixel search to measurements.mask (the valid-plate-area mask
built during camera_localization) instead of a hand-picked crop -- this excludes
the wooden frame automatically, which is what a naive whole-frame color threshold
picks up (the frame's warm wood tones satisfy a loose "red" threshold too).

Usage:
    python3 measure_goal_circle.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from plate_env import HardwarePlateEnv  # noqa: E402


def main():
    env = HardwarePlateEnv()
    try:
        # a full pipeline pass populates plate_pose.T__C_M and measurements.mask
        env._read_state()
        measurements = env.pipeline.measurements
        plate_pose = measurements.plate_pose
        frame = env._grab_frame()

        # Hue is the reliable separator here, not raw RGB differences -- this pen's
        # dark red sampled at hue ~175-177, while the wooden frame (which a simple
        # "R noticeably bigger than G/B" test also matches) sampled at hue ~16-19.
        # Restricting to the mask alone isn't enough: it covers up to the outer
        # frame, not just the inner paper, so wood pixels are still in-bounds.
        valid = measurements.mask[:, :, 0] > 0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat = hsv[:, :, 0], hsv[:, :, 1]
        reddish = ((hue > 165) & (sat > 40) & valid).astype(np.uint8)
        if reddish.sum() < 20:
            raise RuntimeError(
                f"only found {int(reddish.sum())} red pixels inside the valid plate area -- "
                "is the circle actually drawn and the paper in view?"
            )

        # A hand-drawn line has small gaps, which split it into several disconnected
        # blobs -- fitting only the biggest raw blob systematically cuts off part of
        # the circle and biases the fit. Dilate first to bridge those gaps so the
        # whole outline merges into one component, then fit using the ORIGINAL
        # (non-dilated) pixels within that component, not the thickened ones.
        dilated = cv2.dilate(reddish, np.ones((9, 9), np.uint8), iterations=2)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
        biggest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        component_mask = (labels == biggest_label) & (reddish > 0)
        ys, xs = np.where(component_mask)

        # least-squares circle fit: (x-cx)^2 + (y-cy)^2 = r^2, linearized
        A = np.column_stack([2 * xs, 2 * ys, np.ones(len(xs))]).astype(np.float64)
        bvec = xs.astype(np.float64) ** 2 + ys.astype(np.float64) ** 2
        (cx, cy, c), *_ = np.linalg.lstsq(A, bvec, rcond=None)
        radius_px = float(np.sqrt(c + cx**2 + cy**2))
        print(f"detected circle (pixel, this frame's resolution): "
              f"center=({cx:.1f},{cy:.1f}) radius={radius_px:.1f}")

        pts_raw = np.array([[cy, cx], [cy, cx + radius_px]])  # (row, col) convention
        pts_undist = plate_pose.undistort_points(pts_raw)
        center_M = measurements.ball_pos_backproject(pts_undist[0], plate_pose.K, plate_pose.T__C_M)
        edge_M = measurements.ball_pos_backproject(pts_undist[1], plate_pose.K, plate_pose.T__C_M)
        radius_m = float(np.hypot(edge_M[0] - center_M[0], edge_M[1] - center_M[1]))

        print(f"\ncircle center (world meters): ({center_M[0]:.4f}, {center_M[1]:.4f})")
        print(f"circle radius (world meters): {radius_m:.4f}")
        print(f"plate half-extents for reference: x_half={env._x_half:.4f} y_half={env._y_half:.4f}")
        print(f"\nfor train_sac_hw.py:\n"
              f"  fixed_goal=({center_M[0]:.4f}, {center_M[1]:.4f}), goal_tolerance={radius_m:.4f}")

        debug = frame.copy()
        cv2.circle(debug, (int(cx), int(cy)), int(radius_px), (0, 255, 0), 1)
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goal_circle_debug.jpg")
        cv2.imwrite(out_path, debug)
        print(f"\ndebug overlay saved to {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
