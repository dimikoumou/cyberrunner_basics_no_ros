#!/usr/bin/env python3
"""
Measure the 4 FIXED (outer-frame) corner markers once and store them in
state_est/fixed_corners_cache.json, so HardwarePlateEnv can localize the camera
from the stored positions instead of re-detecting them on every launch.

Why: those markers never move, but since the glass came out (2026-09-26) the two
left-side ones are intermittently half-hidden behind a strip of the tilting frame
(depends on the rig's motion history, not the commanded posture). A half-hidden
marker leaves ~5px of blue, detection falls back to a wrong blob, and the camera
pose comes out ~30deg off (rotation diagonal 0.86 instead of ~1). Measuring once,
across several postures, and keeping only the frames where all 4 are cleanly
visible and the pose is plausible avoids that entirely.

Only goal-position commands are sent to motors 1 and 3 (small moves around the
cached level position); no configuration registers are written.

Usage:
    python3 measure_fixed_markers.py
"""
import os
import sys
import json
import time

import numpy as np
import cv2

STATE_EST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "state_est"))
sys.path.insert(0, STATE_EST_DIR)
os.chdir(STATE_EST_DIR)

import dynamixel_sdk as dxl  # noqa: E402
from estimation_pipeline import EstimationPipeline  # noqa: E402
from divers import init_capture  # noqa: E402
import glob  # noqa: E402

CACHE_PATH = os.path.join(STATE_EST_DIR, "fixed_corners_cache.json")
LEVEL_PATH = os.path.abspath(os.path.join(STATE_EST_DIR, "..", "last_level_position.json"))
ADDR_TORQUE_ENABLE, ADDR_GOAL_POSITION, ADDR_POSITION_I_GAIN = 64, 116, 82
FRAMES_PER_POSTURE = 25
PLAUSIBLE_DIAG = 0.99  # every known-good frame this session gave >=0.998


def main():
    pipeline = EstimationPipeline(fps=55, estimator="FiniteDiff", FiniteDiff_mean_steps=4,
                                  print_measurements=False, show_image=False)
    m = pipeline.measurements
    det = m.detector_fixed_points
    h, w = int(m.plate_pose.o.height), int(m.plate_pose.o.width)

    try:
        with open(LEVEL_PATH) as f:
            lvl = json.load(f)
        m1, m3 = int(lvl["m1"]), int(lvl["m3"])
    except (OSError, KeyError, ValueError):
        m1, m3 = 2712, 3865
    postures = [(m1, m3), (m1 + 80, m3), (m1 - 80, m3), (m1, m3 + 120), (m1, m3 - 120), (m1, m3)]

    port = dxl.PortHandler(sorted(glob.glob("/dev/tty.usbserial-*"))[0])
    ph = dxl.PacketHandler(2.0)
    port.openPort()
    port.setBaudRate(1000000)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plate_env import _find_rig_camera
    cap, _, _ = init_capture("CAM", _find_rig_camera(), None, None)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        cap.read()

    good, n_total = [], 0
    try:
        for i in (1, 3):
            ph.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 1)
            ph.write2ByteTxRx(port, i, ADDR_POSITION_I_GAIN, 150)  # RAM, same as HardwarePlateEnv
        for a, b in postures:
            ph.write4ByteTxRx(port, 1, ADDR_GOAL_POSITION, a)
            ph.write4ByteTxRx(port, 3, ADDR_GOAL_POSITION, b)
            time.sleep(1.5)
            n_ok = 0
            for _ in range(FRAMES_PER_POSTURE):
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.resize(frame, (w, h))
                n_total += 1
                det.corners, det.corners_missing = None, True  # always search from defaults
                pts = det.detect_corners(frame)
                if det.corners_missing:
                    continue
                m.plate_pose.camera_localization(pts)
                if np.min(np.abs(np.diag(m.plate_pose.T__W_C[:3, :3]))) < PLAUSIBLE_DIAG:
                    continue
                good.append(pts.copy())
                n_ok += 1
            print(f"posture m1={a} m3={b}: {n_ok}/{FRAMES_PER_POSTURE} clean frames")
    finally:
        for i in (1, 3):
            ph.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 0)
        port.closePort()
        cap.release()

    if len(good) < 10:
        sys.exit(f"only {len(good)}/{n_total} clean frames -- not saving a cache")
    stacked = np.stack(good)
    med = np.median(stacked, axis=0)
    spread = np.max(np.linalg.norm(stacked - med[None], axis=2), axis=0)
    m.plate_pose.camera_localization(med.astype(np.float32))
    diag = np.diag(m.plate_pose.T__W_C[:3, :3])
    print(f"\n{len(good)}/{n_total} clean frames used")
    print("median fixed corners (row, col) at %dx%d:\n%s" % (w, h, np.round(med, 2)))
    print("max per-corner deviation from median (px):", np.round(spread, 2))
    print("rotation diagonal from median corners:", np.round(diag, 4))
    if np.min(np.abs(diag)) < PLAUSIBLE_DIAG:
        sys.exit("median pose is implausible -- not saving")
    with open(CACHE_PATH, "w") as f:
        json.dump({
            "resolution_wh": [w, h],
            "corners_row_col": med.tolist(),
            "n_frames": len(good),
            "max_dev_px": spread.tolist(),
            "measured": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"saved {CACHE_PATH}")


if __name__ == "__main__":
    main()
