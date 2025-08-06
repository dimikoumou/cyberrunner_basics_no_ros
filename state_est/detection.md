Author: Dimitris Koumoutsakos 

This file serves as notes for the 'detection.py' file. 

The libraries needed are OpenCV(cv2) and NumPy, and the modules imported from local files are detect_guassian, detect_gaussian_robust, and mask hsv. This script uses the detect_gaussian and the mask_hsv function from their respective files. This script has one main class: Detector. Detector's job is to find the ball and the corners using gaussian-based detection (see gaussian_robust.md for explanation of gaussian-based detection) and HSV masking. The core components of the system are the following: 
 1. Corner Detection System: Identifies and tracks four colored markers that define the boundaries of the playing field
 2. Ball Tracking System: Locates and follows the blue ball within the defined area

Some key features of the system include the following:
- HSV-based Masking: Both the corners and ball are located using HSV thresholds with optional fallback ranges for robustness.

- Predictive Cropping: To improve performance and reduce false detections, it crops regions around expected positions instead of scanning the full frame.

- Robust Corner Detection: Each corner is detected with circularity and area filtering to isolate reliable blobs, or fallback methods using Gaussian blob detection.

- Ball Detection: The ball is tracked similarly, with checks to avoid false detections near corners.

The Detector class is the main class, with a subclass called DetectorFixedPts. The Detector Class is defined parameters including HSV ranges, quiality and threshold parameters for detection, cropping sizes, and initial ball position. It also has a frame processing function (process_frame()), which handles corner and ball detection for each new frame. The corner detection works by searching each corner area with optional recovery strategies if detection fails. The ball detection works by tracking the ball using HSV masking and blob detection. There are alsp visual indicators included used for debugging (draw_corners, draw_ball). 

The DetectorFixedPts, the subclass, overrides the corner detection method. It uses fixed HSV parameters to redetect a lost corner. 
