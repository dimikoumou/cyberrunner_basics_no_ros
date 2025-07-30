import cv2
import numpy as np
import sys
sys.path.append('state_est')

from detection import Detector
from masking import mask_hsv

def debug_corner_hsv():
    """Debug HSV parameters for corner detection"""
    
    # Load markers
    markers = np.loadtxt("markers.csv", delimiter=",", skiprows=1, usecols=[2, 3])
    print("Loaded markers:")
    for i, marker in enumerate(markers):
        print(f"  {i}: ({marker[0]}, {marker[1]})")
    
    # Create detector (using inner corners for detection)
    detector = Detector(markers[4:], show_subimages=True)
    
    # Capture frame
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to capture frame")
        return
    
    print(f"Frame shape: {frame.shape}")
    
    # Get corner subimages
    subimages, coords = detector.get_default_subimages_corners(frame)
    
    print(f"\nAnalyzing {len(subimages)} corner subimages...")
    
    # Test different HSV parameters
    hsv_params_list = [
        ((76, 138), (133, 255), (68, 255)),  # Default
        ((43, 140), (125, 255), (9, 255)),   # Alternative 1
        ((60, 120), (120, 255), (50, 255)),  # Alternative 2
        ((70, 150), (140, 255), (40, 255)),  # Alternative 3
    ]
    
    for param_idx, hsv_params in enumerate(hsv_params_list):
        print(f"\n--- Testing HSV parameters {param_idx + 1}: {hsv_params} ---")
        
        for i, subimg in enumerate(subimages):
            print(f"\nCorner {i}:")
            print(f"  Subimage shape: {subimg.shape}")
            print(f"  Crop coordinates: {coords[i]}")
            
            # Apply HSV masking
            try:
                sub_masked, mask = mask_hsv(subimg, hsv_params)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
                
                print(f"  Contours found: {len(contours)}")
                if len(contours) > 0:
                    areas = [cv2.contourArea(c) for c in contours]
                    print(f"  Contour areas: {areas}")
                    print(f"  Max area: {max(areas)}")
                
                # Save debug images
                cv2.imwrite(f'debug_corner_{i}_param_{param_idx}_original.png', subimg)
                cv2.imwrite(f'debug_corner_{i}_param_{param_idx}_mask.png', mask)
                
                # Show HSV analysis of center pixel
                center_y, center_x = subimg.shape[0]//2, subimg.shape[1]//2
                hsv_subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2HSV)
                center_hsv = hsv_subimg[center_y, center_x]
                print(f"  Center pixel HSV: {center_hsv}")
                
            except Exception as e:
                print(f"  Error processing corner {i}: {e}")
    
    # Create a visualization showing all corners
    frame_vis = frame.copy()
    for i, coord_pair in enumerate(coords):
        ul, dr = coord_pair[0], coord_pair[1]
        cv2.rectangle(frame_vis, (ul[1], ul[0]), (dr[1], dr[0]), (0, 255, 0), 2)
        cv2.putText(frame_vis, f'C{i}', (ul[1], ul[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Also mark the center of each marker
        marker_center = markers[4 + i]  # Inner corners
        cv2.circle(frame_vis, (int(marker_center[0]), int(marker_center[1])), 3, (255, 0, 0), -1)
    
    cv2.imwrite('debug_all_corners_visualization.png', frame_vis)
    print("\nSaved debug_all_corners_visualization.png")
    print("Check the debug images to analyze HSV masking results")
    
    # Create HSV analysis for the full frame
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Sample HSV values at marker locations
    print("\nHSV values at marker centers:")
    for i, marker in enumerate(markers[4:]):  # Inner corners
        x, y = int(marker[0]), int(marker[1])
        if 0 <= y < hsv_frame.shape[0] and 0 <= x < hsv_frame.shape[1]:
            hsv_val = hsv_frame[y, x]
            print(f"  Inner corner {i} at ({x}, {y}): HSV = {hsv_val}")
    
    print("\nRecommended next steps:")
    print("1. Check debug_corner_*_original.png files to see what the detector is looking at")
    print("2. Check debug_corner_*_mask.png files to see which HSV parameters work best")
    print("3. Look at the HSV values printed above to tune the HSV parameters")
    print("4. Use the HSV values to create better HSV ranges in detection.py")

if __name__ == "__main__":
    debug_corner_hsv()
