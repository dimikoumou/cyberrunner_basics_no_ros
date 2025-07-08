# NOTE: cyberrunner_state_estimation/cyberrunner_state_estimation/core/masking.py
import numpy as np
import cv2 as cv

from typing import Tuple


def mask_hsv(img, color_params=None):
    # NOTE: useful fior debugging the corner detection
    # print(" IM MASKING THE IMAGE OF SHAPE: ", img.shape)
    imageHsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    minHsv = np.array([color_params[0][0], color_params[1][0], color_params[2][0]])
    maxHsv = np.array([color_params[0][1], color_params[1][1], color_params[2][1]])

    h, w = imageHsv.shape[:2]
    mask = cv.inRange(imageHsv, minHsv, maxHsv)
    # img_masked = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8) * 255
    return None, mask
