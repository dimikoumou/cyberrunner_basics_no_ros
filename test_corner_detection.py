import sys
sys.path.append('state_est')
import cv2
import numpy as np
from state_est.detection import Detector

# Load markers and create detector
markers = np.loadtxt("markers.csv", delimiter=",", skiprows=1, usecols=[2, 3])
print("Loaded markers:", markers[:4])  # Just outer corners

# Create detector with outer corners only (first 4)
detector = Detector(markers[4:], show_subimages=True)  # Note: markers[4:] for inner corners as per original code

# Capture frame
cap = cv2.VideoCapture(1)
ret, frame = cap.read()
if ret:
    print(f"Frame shape: {frame.shape}")
    
    # Try to detect corners
    try:
        corners = detector.detect_corners(frame)
        print("Detected corners:", corners)
    except Exception as e:
        print(f"Error during detection: {e}")
        
    # Show the cropped regions
    cv2.waitKey(0)
else:
    print("Failed to capture frame")
    
cap.release()
cv2.destroyAllWindows()
