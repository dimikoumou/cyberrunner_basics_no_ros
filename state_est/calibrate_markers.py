#!/usr/bin/env python3
"""
Click-based corner-marker calibration for state_est/markers.csv, against a LIVE
camera feed, with a zoomed-in nudge/confirm step per point.

Coordinates are saved at NATIVE camera resolution (1920x1080), matching what the
repo-root board_detection.py also does -- detection.py's Detector.__init__ divides
whatever it loads from markers.csv by 3 internally to match the pipeline's 640x360
working resolution, so markers.csv must hold full-resolution pixel coordinates, not
pre-scaled ones.

Why the nudge/confirm step: on this setup, the raw mouse-click coordinate reported by
OpenCV's window landed dozens of pixels away from the actual click location (a
window/backing-scale mismatch, common with OpenCV's Cocoa backend on macOS Retina
displays). Rather than chase that platform quirk, a rough click just places a
candidate point; you then nudge it pixel-by-pixel against a zoomed crop -- which is
computed directly from array indices I control, not from the window's mouse mapping --
until it's exactly centered on the marker, and confirm it. This is correct regardless
of whatever the click-to-pixel offset actually is.

Order: 4 outer/fixed corners (lower-left, lower-right, upper-right, upper-left), then
4 inner/moving plate corners in the same rotational order -- matching what
Measurements.__init__ expects (markers[:4] = fixed reference corners passed to
DetectorFixedPts, markers[4:] = the plate's own moving corners passed to Detector).

Controls:
    click       rough-place the candidate point for the current corner
    w/a/s/d     nudge the candidate up/left/down/right by the current step size
    1 / 2 / 3   set nudge step size to 1 / 5 / 20 pixels
    space/enter confirm the candidate, lock it in, advance to the next corner
    r           clear everything and restart
    ESC         abort without saving

Usage:
    python3 calibrate_markers.py
"""
import os

import cv2
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from divers import init_capture  # noqa: E402

CORNER_LABELS = [
    ("outer", "lower left"), ("outer", "lower right"),
    ("outer", "upper right"), ("outer", "upper left"),
    ("inner", "lower left"), ("inner", "lower right"),
    ("inner", "upper right"), ("inner", "upper left"),
]

ZOOM_HALF_SIZE = 40   # crop is (2*ZOOM_HALF_SIZE) square, in native-resolution pixels
ZOOM_SCALE = 8        # displayed zoom window is ZOOM_SCALE times the crop size
STEP_SIZES = {ord("1"): 1, ord("2"): 5, ord("3"): 20}

points = []          # confirmed (x, y) points, native resolution
candidate = None     # [x, y] currently being placed/nudged, or None
step_size = 5


def mouse_callback(event, x, y, flags, userdata):
    global candidate
    if event == cv2.EVENT_LBUTTONUP and len(points) < 8:
        candidate = [x, y]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_main(frame):
    out = frame.copy()
    h, w = out.shape[:2]
    for i, (x, y) in enumerate(points):
        cv2.drawMarker(out, (x, y), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 15, 2)
        cv2.putText(out, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if candidate is not None:
        cv2.drawMarker(out, tuple(candidate), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    if len(points) < 8:
        kind, pos = CORNER_LABELS[len(points)]
        msg = f"click near {kind}/{pos}  ({len(points)}/8)  step={step_size}px"
    else:
        msg = "8/8 confirmed -- press any key to save, 'r' to redo"
    cv2.putText(out, msg, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    if candidate is None and len(points) < 8:
        cv2.putText(out, "click roughly on the marker, then nudge with w/a/s/d",
                    (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_zoom(frame):
    h, w = frame.shape[:2]
    if candidate is None:
        canvas = np.zeros((ZOOM_HALF_SIZE * 2 * ZOOM_SCALE, ZOOM_HALF_SIZE * 2 * ZOOM_SCALE, 3), dtype=np.uint8)
        cv2.putText(canvas, "click a point", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return canvas
    cx, cy = candidate
    x0 = clamp(cx - ZOOM_HALF_SIZE, 0, w - 2 * ZOOM_HALF_SIZE)
    y0 = clamp(cy - ZOOM_HALF_SIZE, 0, h - 2 * ZOOM_HALF_SIZE)
    crop = frame[y0:y0 + 2 * ZOOM_HALF_SIZE, x0:x0 + 2 * ZOOM_HALF_SIZE]
    zoomed = cv2.resize(crop, (crop.shape[1] * ZOOM_SCALE, crop.shape[0] * ZOOM_SCALE),
                         interpolation=cv2.INTER_NEAREST)
    # candidate's position within the crop, scaled up -- this is where the crosshair goes
    zx = (cx - x0) * ZOOM_SCALE
    zy = (cy - y0) * ZOOM_SCALE
    cv2.drawMarker(zoomed, (zx, zy), (0, 255, 255), cv2.MARKER_CROSS, 30, 1)
    cv2.putText(zoomed, "w/a/s/d nudge, 1/2/3 step, space/enter confirm",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return zoomed


def main():
    global candidate, step_size

    cap, _, _ = init_capture("CAM", 0, None, None)
    if not cap.isOpened():
        raise SystemExit("could not open the camera")

    print("Calibrating at native camera resolution (markers.csv values get divided by 3 "
          "internally by detection.py).")
    print("Click roughly on each marker, then nudge the candidate (w/a/s/d) using the zoomed "
          "preview until it's exactly centered, then space/enter to confirm.")
    print("1/2/3 change nudge step size (1/5/20 px). 'r' resets. ESC aborts.\n")
    print(f"Click near the {CORNER_LABELS[0][0]} ({CORNER_LABELS[0][1]}) corner...")

    cv2.namedWindow("calibrate_markers (live)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrate_markers (live)", mouse_callback)
    cv2.namedWindow("zoom")

    last_frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("warning: dropped frame, retrying...")
            continue
        last_frame = frame
        h, w = frame.shape[:2]
        cv2.imshow("calibrate_markers (live)", draw_main(frame))
        cv2.imshow("zoom", draw_zoom(frame))

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            print("aborted, nothing saved")
            cap.release()
            cv2.destroyAllWindows()
            return

        if key == ord("r"):
            points.clear()
            candidate = None
            print(f"cleared. Click near the {CORNER_LABELS[0][0]} ({CORNER_LABELS[0][1]}) corner...")
            continue

        if key in STEP_SIZES:
            step_size = STEP_SIZES[key]
            continue

        if candidate is not None:
            if key == ord("w"):
                candidate[1] = clamp(candidate[1] - step_size, 0, h - 1)
            elif key == ord("s"):
                candidate[1] = clamp(candidate[1] + step_size, 0, h - 1)
            elif key == ord("a"):
                candidate[0] = clamp(candidate[0] - step_size, 0, w - 1)
            elif key == ord("d"):
                candidate[0] = clamp(candidate[0] + step_size, 0, w - 1)
            elif key in (13, 32):  # Enter or Space
                points.append(tuple(candidate))
                candidate = None
                if len(points) < 8:
                    kind, pos = CORNER_LABELS[len(points)]
                    print(f"confirmed. Click near the {kind} ({pos}) corner...")
                else:
                    print("All 8 corners confirmed, saving...")

        if len(points) >= 8 and candidate is None and key != 255:
            break

    cap.release()
    cv2.destroyAllWindows()

    with open("markers.csv", "w") as f:
        f.write("corner_type,position,x,y\n")
        for (kind, pos), (x, y) in zip(CORNER_LABELS, points):
            f.write(f"{kind},{pos},{x},{y}\n")
    print("saved state_est/markers.csv")
    print("re-run run_state_estimation.py (or the live check) to confirm detection now works.")


if __name__ == "__main__":
    main()
