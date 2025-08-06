# debug_full_coordinates.py

## Description
This script provides a comprehensive debugging environment for marker coordinates. It loads markers from a CSV file, analyzes their structure, and tests the coordinate transformations used by different detector classes. Additionally, it captures a live camera frame, draws rectangles around the detected crop regions, and saves both the full frame and individual cropped images for visual inspection.

## When to use this
Use this script when you need a deep dive into how marker coordinates are being processed. It is particularly useful for visualizing the exact regions of the camera feed that are being cropped and for verifying that the coordinate transformations align with the physical markers on the board.

