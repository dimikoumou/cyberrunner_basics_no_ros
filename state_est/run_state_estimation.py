"""
Clean, hardware-free entry point for the (no-ROS) CyberRunner state estimation.

Runs the estimation pipeline on a single image or a video file -- no camera and
no motors required -- and saves annotated frames (detected corners, ball, and
estimated plate angles) to demo_output/ as portfolio evidence.

Key robustness fix vs. the old scripts: instead of a hardcoded /3 downscale, we
resize every input to the EXACT resolution the OCamCalib model expects
(calibration_resolution / scale_factor). This is what makes detection work
regardless of whether the source is 720p, 1080p, or already downscaled.

Usage:
    python run_state_estimation.py                      # defaults to testimg_jan.png
    python run_state_estimation.py <image.png|video.avi>
"""
import os
import sys

# Make relative resource loads (markers.csv, calib_razer_data.txt) work no matter
# where this is launched from: run everything relative to this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLBACKEND", "Agg")  # headless: no GUI windows required

import cv2
import numpy as np
from estimation_pipeline import EstimationPipeline

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")


def build_pipeline():
    return EstimationPipeline(
        fps=55,
        estimator="FiniteDiff",
        FiniteDiff_mean_steps=4,   # NOT 0 -- 0 produces NaN velocities
        print_measurements=True,
        show_image=False,          # headless
        do_anim_3d=False,
        show_subimages_detector=False,
    )


def fit_to_model(frame, pipe):
    """Resize a frame to the model's expected (width, height)."""
    exp_h = int(pipe.measurements.plate_pose.o.height)
    exp_w = int(pipe.measurements.plate_pose.o.width)
    if frame.shape[:2] != (exp_h, exp_w):
        frame = cv2.resize(frame, (exp_w, exp_h))
    return frame


def annotate(frame, pipe, xb, yb, inputs):
    out = frame.copy()
    # detected corners (may be None on a failed frame)
    corners = pipe.measurements.detector.corners
    if corners is not None:
        for (r, c) in np.asarray(corners):
            if np.isfinite(r) and np.isfinite(c):
                cv2.circle(out, (int(c), int(r)), 4, (0, 255, 255), -1)
    # ball position in image space, if it was actually found
    det = pipe.measurements.detector
    if getattr(det, "is_ball_found", False) and det.ball_pos is not None:
        r, c = det.ball_pos
        if np.isfinite(r) and np.isfinite(c):
            cv2.circle(out, (int(c), int(r)), 6, (0, 0, 255), 2)
    a_deg, b_deg = np.degrees(inputs[0]), np.degrees(inputs[1])
    cv2.putText(out, f"alpha={a_deg:+.1f}deg  beta={b_deg:+.1f}deg",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"ball maze=({xb:+.3f},{yb:+.3f})m",
                (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def run_image(path):
    pipe = build_pipeline()
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    frame = fit_to_model(frame, pipe)
    x_hat, P, inputs, xb, yb = pipe.estimate(frame)
    os.makedirs("demo_output", exist_ok=True)
    out_path = os.path.join("demo_output", "annotated_" + os.path.basename(path))
    cv2.imwrite(out_path, annotate(frame, pipe, xb, yb, inputs))
    print(f"\n=> corners found : {pipe.measurements.detector.corners is not None}")
    print(f"=> plate angles  : ({np.degrees(inputs[0]):+.2f}, {np.degrees(inputs[1]):+.2f}) deg")
    print(f"=> ball maze pos : ({xb:+.4f}, {yb:+.4f}) m")
    print(f"=> annotated image saved: {os.path.abspath(out_path)}")


def run_video(path):
    pipe = build_pipeline()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    os.makedirs("demo_output", exist_ok=True)
    writer = None
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:          # clean EOF handling (old scripts crashed here)
            break
        frame = fit_to_model(frame, pipe)
        x_hat, P, inputs, xb, yb = pipe.estimate(frame)
        vis = annotate(frame, pipe, xb, yb, inputs)
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(os.path.join("demo_output", "annotated.avi"),
                                     cv2.VideoWriter_fourcc(*"XVID"), 30, (w, h))
        writer.write(vis)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    print(f"\n=> processed {n} frames -> demo_output/annotated.avi")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "testimg_jan.png"
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        run_video(path)
    elif ext in IMAGE_EXTS:
        run_image(path)
    else:
        raise SystemExit(f"Unsupported file type: {ext}")


if __name__ == "__main__":
    main()
