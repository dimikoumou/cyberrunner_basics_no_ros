"""
One-shot estimation check: grab a single live frame, run it through the (canonical)
EstimationPipeline, print the results, and save an annotated snapshot to demo_output/
so you can see exactly what was detected. Companion to live_view.py (continuous) --
use this when you just want a quick single check rather than a live window.

Usage:
    python3 test_estimation.py
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
        ul, dr = det.default_coords_subimage_ball
        cv2.rectangle(out, (int(ul[1]), int(ul[0])), (int(dr[1]), int(dr[0])), (0, 165, 255), 2)
        cv2.putText(out, "ball search window (not found)", (int(ul[1]), int(ul[0]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    a_deg, b_deg = np.degrees(inputs[0]), np.degrees(inputs[1])
    cv2.putText(out, f"alpha={a_deg:+.1f}deg  beta={b_deg:+.1f}deg", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"ball maze=({xb:+.3f},{yb:+.3f})m", (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main():
    cap, _, _ = init_capture("CAM", 0, None, None)
    if not cap.isOpened():
        raise SystemExit("could not open the camera")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read a frame from the camera")

    pipeline = EstimationPipeline(
        fps=55,
        estimator="FiniteDiff",
        FiniteDiff_mean_steps=4,  # 0 produces NaN velocities out of the estimator
        print_measurements=True,
        show_image=False,
    )
    exp_w = int(pipeline.measurements.plate_pose.o.width)
    exp_h = int(pipeline.measurements.plate_pose.o.height)
    frame_r = cv2.resize(frame, (exp_w, exp_h))

    x_hat, P, inputs, xb, yb = pipeline.estimate(frame_r)

    print("\nEstimation Results:")
    print(f"Ball position (x, y): ({xb:.3f}, {yb:.3f})")
    print(f"Ball found: {pipeline.measurements.detector.is_ball_found}")
    print(f"Corners found: {pipeline.measurements.detector.corners is not None}")
    print(f"Plate angles (alpha, beta): ({np.degrees(inputs[0]):.2f}, {np.degrees(inputs[1]):.2f}) degrees")
    print(f"State estimate (x_hat): {x_hat}")
    print(f"Covariance matrix (P):\n{P}")

    os.makedirs("demo_output", exist_ok=True)
    out_path = "demo_output/test_estimation.png"
    cv2.imwrite(out_path, annotate(frame_r, pipeline, xb, yb, inputs))
    print(f"\nannotated snapshot saved to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
