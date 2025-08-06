import cv2
import numpy as np
import sys
sys.path.append('state_est')

# First, let's understand what's in markers.csv
print("=== MARKERS.CSV ANALYSIS ===")
markers = np.loadtxt("markers.csv", delimiter=",", skiprows=1, usecols=[2, 3])
print("Raw markers from CSV:")
for i, marker in enumerate(markers):
    labels = ["outer_ll", "outer_lr", "outer_ur", "outer_ul", "inner_ll", "inner_lr", "inner_ur", "inner_ul"]
    print(f"  {i}: {labels[i]} = ({marker[0]}, {marker[1]})")

print(f"\nOuter corners (0-3): {markers[:4]}")
print(f"Inner corners (4-7): {markers[4:]}")

# Test the coordinate transformation used by Detector
print("\n=== COORDINATE TRANSFORMATION DEBUG ===")
from detection import Detector

# Test with inner corners (what the main detector uses)
detector_inner = Detector(markers[4:])
print("Inner corners transformation:")
print(f"  Input markers[4:]: {markers[4:]}")
print(f"  After transformation: {detector_inner.default_coords_subimages_corners}")

# Test with outer corners (what fixed points detector uses)
try:
    from detection import DetectorFixedPts
    detector_outer = DetectorFixedPts(markers[:4])
    print("Outer corners transformation:")
    print(f"  Input markers[:4]: {markers[:4]}")
    # DetectorFixedPts might have different coordinate handling
except Exception as e:
    print(f"Couldn't test DetectorFixedPts: {e}")

# Capture a frame and test what we're actually cropping
print("\n=== ACTUAL CROPPING TEST ===")
cap = cv2.VideoCapture(1)
ret, frame = cap.read()
if ret:
    print(f"Frame shape: {frame.shape}")
    h, w = frame.shape[:2]
    
    # Test what the detector is actually cropping
    subimages, coords = detector_inner.get_default_subimages_corners(frame)
    
    # Save a full frame with crop rectangles drawn
    frame_copy = frame.copy()
    for i, coord_pair in enumerate(coords):
        ul, dr = coord_pair[0], coord_pair[1]
        # Draw rectangle on full frame
        cv2.rectangle(frame_copy, (ul[1], ul[0]), (dr[1], dr[0]), (0, 255, 0), 2)
        cv2.putText(frame_copy, f'C{i}', (ul[1], ul[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        print(f"Crop {i}: ul={ul}, dr={dr}, size={dr-ul}")
        
        # Save individual crops with better names
        cv2.imwrite(f'debug_crop_{i}_inner_corner.png', subimages[i])
    
    cv2.imwrite('debug_full_frame_with_crops.png', frame_copy)
    print("Saved debug_full_frame_with_crops.png - check if green rectangles are on corners!")
    
    # Also test manual coordinate access
    print("\n=== MANUAL COORDINATE TEST ===")
    print("Let's manually check if your calibrated coordinates make sense:")
    for i, marker in enumerate(markers[4:]):  # Inner corners
        x, y = int(marker[0]), int(marker[1])
        print(f"Inner corner {i}: trying to access pixel at ({x}, {y})")
        if 0 <= y < h and 0 <= x < w:
            # Crop a small region around this point
            crop_size = 34
            y1, y2 = max(0, y-crop_size//2), min(h, y+crop_size//2)
            x1, x2 = max(0, x-crop_size//2), min(w, x+crop_size//2)
            manual_crop = frame[y1:y2, x1:x2]
            cv2.imwrite(f'debug_manual_crop_{i}_at_{x}_{y}.png', manual_crop)
            print(f"  Saved manual crop: debug_manual_crop_{i}_at_{x}_{y}.png")
        else:
            print(f"  ERROR: Coordinates ({x}, {y}) are outside frame bounds!")
    
cap.release()

print("\n=== SUMMARY ===")
print("Check these files to debug:")
print("1. debug_full_frame_with_crops.png - Are green rectangles on corners?")
print("2. debug_manual_crop_*_at_*.png - Do these show the actual corners?")
print("3. debug_crop_*_inner_corner.png - What the detector is actually seeing")
