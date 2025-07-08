#!/usr/bin/env python3

import cv2
import numpy as np

# Global state
img = None
outer_points = []
inner_points = []
clicked_index = 0  # how many corners clicked so far

# We'll label corners in the order:
#  1) lower left
#  2) lower right
#  3) upper right
#  4) upper left
# Then we repeat the same order for inner corners.
corner_labels = [
    "outer: lower left",
    "outer: lower right",
    "outer: upper right",
    "outer: upper left",
    "inner: lower left",
    "inner: lower right",
    "inner: upper right",
    "inner: upper left",
]

def mouse_callback(event, x, y, flags, param):
    global outer_points, inner_points, clicked_index, img

    if event == cv2.EVENT_LBUTTONUP:
        # Determine if we are still collecting outer corners or moving to inner corners
        if clicked_index < 4:
            # Outer corners
            outer_points.append((x, y))
        else:
            # Inner corners
            inner_points.append((x, y))

        # Draw a marker on the image so the user can see the click
        cv2.drawMarker(img, (x, y), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 15, 2)

        clicked_index += 1
        if clicked_index < 8:
            # Print next corner label
            print(f"Please click the {corner_labels[clicked_index]} corner...")
        else:
            # We have all 8 points
            print("All 8 corners selected. Saving to 'board_corners.csv'...")

            # Prepare data for CSV
            with open('markers.csv', 'w') as f:
                # Write header
                f.write("corner_type,position,x,y\n")
                
                # Write outer points
                for i, (x, y) in enumerate(outer_points):
                    position = corner_labels[i].split(': ')[1]
                    f.write(f"outer,{position},{x},{y}\n")
                
                # Write inner points
                for i, (x, y) in enumerate(inner_points):
                    position = corner_labels[i+4].split(': ')[1]
                    f.write(f"inner,{position},{x},{y}\n")
                
            print("Saved 'board_corners.csv' with corner type, position, and coordinates.")
            print("Press any key in the image window to exit.")

def main():
    global img, clicked_index

    # 1) Load the image
    img = cv2.imread("board.png")
    if img is None:
        print("Error: Could not load 'board.png'. Make sure the file exists.")
        return

    # 2) Set up the window and mouse callback
    cv2.namedWindow("Board Detection")
    cv2.setMouseCallback("Board Detection", mouse_callback)

    print(" the  loaded image is of the resolution: ", img.shape)
    print("Instructions:")
    print("  - We will collect 8 total points:")
    print("    4 'outer' corners (lower left, lower right, upper right, upper left)")
    print("    4 'inner' corners (lower left, lower right, upper right, upper left)")
    print("  - Click in the image window to set each corner in order.")
    print("  - Press ESC at any time to abort without saving.\n")

    print(f"Please click the {corner_labels[clicked_index]} corner...")

    while True:
        cv2.imshow("Board Detection", img)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Exiting without saving (ESC pressed).")
            break

        # If we already clicked 8 points, wait for a key press to exit
        if clicked_index >= 8:
            if key != 255:  # some key pressed
                break

    cv2.destroyAllWindows()
    
    
    # create a new image and display blue squares on top of saved points
    img = cv2.imread("board.png")
    for i, (x, y) in enumerate(outer_points):
        cv2.drawMarker(img, (x, y), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(img, f"outer {i}", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    for i, (x, y) in enumerate(inner_points):
        cv2.drawMarker(img, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(img, f"inner {i}", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("Board Detection with Markers", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Finished processing. Exiting.")

if __name__ == "__main__":
    main()