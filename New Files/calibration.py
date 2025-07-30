# Save this as calibration.py

import cv2
import numpy as np
import os

def calibrate_camera():
    """
    Captures images of a checkerboard and computes the standard OpenCV
    camera matrix (mtx) and distortion coefficients (dist).
    Saves the results to 'calibration_data.npz'.
    """
    # --- Configuration ---
    CHECKERBOARD_DIMS = (10, 7)  # Inner corners: (width, height)
    SQUARE_SIZE_METERS = 0.024   # The size of a square on your checkerboard
    NUM_IMAGES_TO_CAPTURE = 20
    CAMERA_INDEX = 1 # IMPORTANT: Adjust if your camera has a different index (e.g., 0, 2)
    
    # --- Setup ---
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((CHECKERBOARD_DIMS[0] * CHECKERBOARD_DIMS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_DIMS[0], 0:CHECKERBOARD_DIMS[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_METERS
    objpoints = []
    imgpoints = []
    
    # --- Image Capture Loop ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"🚨 Error: Could not open camera with index {CAMERA_INDEX}.")
        return

    # Use the highest resolution your camera supports for best results
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    images_captured = 0
    print(f"📷 Starting calibration. Please show the {CHECKERBOARD_DIMS} checkerboard from various angles.")
    
    while images_captured < NUM_IMAGES_TO_CAPTURE:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_DIMS, None)

        if found:
            print(f"✅ Found board! Capturing image {images_captured + 1}/{NUM_IMAGES_TO_CAPTURE}...")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            images_captured += 1
            # Draw feedback on the frame and wait 2 seconds
            cv2.drawChessboardCorners(frame, CHECKERBOARD_DIMS, corners2, True)
            cv2.imshow('Calibration', frame)
            cv2.waitKey(2000)
        else:
             cv2.imshow('Calibration', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- Calibration Calculation ---
    if len(imgpoints) >= 10:
        print(f"\nCaptured {len(imgpoints)} images. Calibrating camera...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            print("✅ Camera calibrated successfully.")
            np.savez('calibration_data.npz', mtx=mtx, dist=dist)
            print("✅ Calibration data saved to 'calibration_data.npz'")
        else:
            print("🚨 Calibration failed.")
    else:
        print("🚨 Calibration cancelled or not enough images captured.")

if __name__ == "__main__":
    calibrate_camera()
