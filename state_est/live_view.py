#!/usr/bin/env python3
"""
Continuous live viewer for the state estimation pipeline -- no motor control, no
per-frame keypress. Shows corners, the ball (or its search window when not found),
and the estimated plate angles, updating every frame, so you can see exactly what the
detector sees in real time.

Usage:
    python3 live_view.py
"""
import os

import cv2
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from divers import init_capture  # noqa: E402
from estimation_pipeline import EstimationPipeline  # noqa: E402


def annotate(frame, pipe, xb, yb, inputs):
    out = frame.copy()
    det = pipe.measurements.detector
    if det.corners is not None:
        for (r, c) in np.asarray(det.corners):
            if np.isfinite(r) and np.isfinite(c):
                cv2.circle(out, (int(c), int(r)), 4, (0, 255, 255), -1)
    if getattr(det, "is_ball_found", False) and det.ball_pos is not None:
        r, c = det.ball_pos
        if np.isfinite(r) and np.isfinite(c):
            cv2.circle(out, (int(c), int(r)), 8, (0, 0, 255), 2)
    else:
        # ball not found -- draw the fixed search window it's looking in, so you can
        # see whether the ball is even inside the region being searched
        ul, dr = det.default_coords_subimage_ball
        cv2.rectangle(out, (int(ul[1]), int(ul[0])), (int(dr[1]), int(dr[0])), (0, 165, 255), 2)
        cv2.putText(out, "ball search window (not found)", (int(ul[1]), int(ul[0]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    a_deg, b_deg = np.degrees(inputs[0]), np.degrees(inputs[1])
    cv2.putText(out, f"alpha={a_deg:+.1f}deg  beta={b_deg:+.1f}deg", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"ball maze=({xb:+.3f},{yb:+.3f})m  found={getattr(det, 'is_ball_found', False)}",
                (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main():
    cap, _, _ = init_capture("CAM", 0, None, None)
    if not cap.isOpened():
        raise SystemExit("could not open the camera")

    pipe = EstimationPipeline(fps=55, estimator="FiniteDiff", FiniteDiff_mean_steps=4,
                               print_measurements=False, show_image=False)
    exp_w = int(pipe.measurements.plate_pose.o.width)
    exp_h = int(pipe.measurements.plate_pose.o.height)
    print(f"live view at {exp_w}x{exp_h} -- press 'q' or ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame_r = cv2.resize(frame, (exp_w, exp_h))
        try:
            x_hat, P, inputs, xb, yb = pipe.estimate(frame_r)
            vis = annotate(frame_r, pipe, xb, yb, inputs)
        except Exception as e:
            vis = frame_r.copy()
            cv2.putText(vis, f"estimate() error: {e}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("live_view", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
