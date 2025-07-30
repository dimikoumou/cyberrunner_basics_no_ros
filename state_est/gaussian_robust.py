# NOTE:  cyberrunner_state_estimation/cyberrunner_state_estimation/core/gaussian_robust.py
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import time
from copy import deepcopy


def detect_gaussian(mask, j, q, th, show_sub, use_contour=True):
    """DEPRECATED: This function is not robust enough. Use detect_gaussian_robust instead."""
    # ... (rest of the old function, but we will no longer use it)
    if not use_contour:
        X = np.array(np.where(mask > 0)).T
        if X.shape[0] < 2:
            return np.array([0, 0]), False
        c = np.mean(X, axis=0)
        return c, True
    else:
        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        if len(contours) == 0:
            return (np.asarray(mask.shape) - 1.0) / 2.0, False
        contour = max(contours, key=cv.contourArea)
        M = cv.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return np.array([cy, cx]), True
        else:
            return (np.asarray(mask.shape) - 1.0) / 2.0, False

def detect_gaussian_robust(mask, j, q, th, show_sub, use_contour=True):
    """
    Robustly detects the centroid of a marker in a binary mask using advanced
    contour filtering based on area and circularity.
    """
    if not use_contour:
        # This part is kept for compatibility but is not recommended
        X = np.array(np.where(mask > 0)).T
        if X.shape[0] < 2:
            return np.array([0, 0]), False
        c = np.mean(X, axis=0)
        return c, True

    # 1. Pre-process the mask with morphological operations
    kernel_erosion = np.ones((2, 2), np.uint8)
    kernel_dilatation = np.ones((4, 4), np.uint8)
    mask_eroded = cv.erode(mask, kernel_erosion, iterations=1)
    mask_processed = cv.dilate(mask_eroded, kernel_dilatation, iterations=2)

    # 2. Find all potential contours
    contours = cv.findContours(mask_processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

    if len(contours) == 0:
        if show_sub:
            print(f"[Robust detection] MASK {j}: no contours found")
        return (np.asarray(mask.shape) - 1.0) / 2.0, False

    # 3. Filter contours based on area and circularity
    valid_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        
        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        # These thresholds are critical for robustness:
        # Area: filters out tiny noise and large spurious objects
        # Circularity: ensures we get dot-like markers, not lines or weird shapes
        if 50 < area < 1500 and circularity > 0.4:
            valid_contours.append(contour)

    if not valid_contours:
        if show_sub:
            print(f"[Robust detection] MASK {j}: no valid contours after filtering")
        return (np.asarray(mask.shape) - 1.0) / 2.0, False

    # 4. Select the best contour (largest valid one)
    best_contour = max(valid_contours, key=cv.contourArea)
    M = cv.moments(best_contour)

    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        c = np.array([cy, cx])
        blob_found = True
    else:
        c = (np.asarray(mask.shape) - 1.0) / 2.0
        blob_found = False

    if show_sub:
        debug_img = cv.cvtColor(mask, cv.COLOR_GRAY_BGR)
        cv.drawContours(debug_img, contours, -1, (255, 0, 0), 1) # All contours in blue
        cv.drawContours(debug_img, valid_contours, -1, (0, 255, 0), 1) # Valid in green
        if blob_found:
            cv.drawMarker(debug_img, (int(c[1]), int(c[0])), (0, 0, 255), cv.MARKER_TILTED_CROSS, 7, 2)
        cv.imshow(f"sub_robust_{j}", debug_img)

    return c, blob_found


if __name__ == "__main__":

    im = cv.imread("board.png")

    c = detect_gaussian(im, 0, 5, 0.5, True)
    print(c)
    cv.drawMarker(im, c, (0,0,255), cv.MARKER_TILTED_CROSS, 5, 1)

    cv.imshow("out", im)
    cv.waitKey(0)