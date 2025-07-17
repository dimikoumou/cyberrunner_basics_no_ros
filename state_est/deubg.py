import cv2
import numpy as np
import csv
import sys # Used to exit gracefully if files are not found

def debug_corner_detection(frame, markers, corner_index=0):
    """
    Debug function to understand why corner detection is failing.
    This version includes explanations of the original errors.
    """
    # Get the marker position from the loaded data
    marker_pos = markers[4 + corner_index]
    print(f"\nMarker {corner_index} position: {marker_pos}")

    # Default crop size
    crop_size = 65 / 3

    # Center coordinates from CSV
    center_x = marker_pos[2]
    center_y = marker_pos[3]

    # Calculate crop bounds
    h, w = frame.shape[:2]
    y_start = max(0, int(center_y - crop_size / 2))
    y_end = min(h, int(center_y + crop_size / 2))
    x_start = max(0, int(center_x - crop_size / 2))
    x_end = min(w, int(center_x + crop_size / 2))

    print(f"Crop bounds (Y, X): [{y_start}:{y_end}, {x_start}:{x_end}]")

    # --- Explanation of Correction 2: Image Cropping ---
    print("\n[CORRECTION 2]: Fixing the image cropping logic.")
    print("  - The original code had a slicing error: `frame[x_range, y_range]`.")
    print("  - NumPy requires slicing in `[rows, columns]` format, which is `frame[y_range, x_range]`.")
    print("  - Slicing correctly now...")
    
    # Extract subimage using correct [row, column] indexing
    sub_im = frame[y_start:y_end, x_start:x_end]
    print(f"Subimage shape: {sub_im.shape}")
    
    if sub_im.size == 0:
        print("Error: Cropped subimage is empty. Check crop bounds and image dimensions.")
        return None, None

    # Convert to HSV
    hsv_sub = cv2.cvtColor(sub_im, cv2.COLOR_BGR2HSV)

    # ... (rest of the function is the same)
    
    # Apply default HSV mask
    hsv_params = ((43, 140), (125, 255), (9, 255))
    lower_hsv = np.array([hsv_params[0][0], hsv_params[1][0], hsv_params[2][0]])
    upper_hsv = np.array([hsv_params[0][1], hsv_params[1][1], hsv_params[2][1]])

    mask = cv2.inRange(hsv_sub, lower_hsv, upper_hsv)

    # Show images
    cv2.imshow('Original Subimage', sub_im)
    cv2.imshow('HSV Mask', mask)
    cv2.imshow('Masked Result', cv2.bitwise_and(sub_im, sub_im, mask=mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return sub_im, mask

# --- Main execution ---
# Provide the correct paths to your files
image_path = 'board.png'
csv_path = 'markers.csv'

# 1. Read the image
frame = cv2.imread(image_path)
if frame is None:
    print(f"Fatal Error: Could not read image from {image_path}. Exiting.")
    sys.exit()

# --- Explanation of Correction 1: File Reading ---
print("[CORRECTION 1]: Reading the CSV file properly.")
print("  - The original code passed the filename 'markers.csv' (a string) to the function.")
print("  - The function cannot work with a string; it needs the data from inside the file.")
print(f"  - Loading data from '{csv_path}' now...")

# 2. Read and parse the CSV data
markers_data = []
try:
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            markers_data.append([row[0], row[1], float(row[2]), float(row[3])])
    print(f"  - Success! Loaded {len(markers_data)} markers.")

except FileNotFoundError:
    print(f"Fatal Error: Could not find markers file at {csv_path}. Exiting.")
    sys.exit()

# 3. Call the function with the loaded data
if len(markers_data) > 4:
     debug_corner_detection(frame, markers_data, corner_index=0)
else:
    print("Error: Not enough markers in the CSV file to find an inner corner.")