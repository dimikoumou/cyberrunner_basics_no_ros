import numpy as np
import pandas as pd

# Load markers
markers = np.loadtxt("markers.csv", delimiter=",", skiprows=1, usecols=[2, 3])
print("Original markers from CSV:")
print(markers)

# Simulate the transformation in detection.py
corner_subimage_half_size = 17

print("\nAfter [:, ::-1] (swap x,y):")
swapped = markers[:, ::-1]
print(swapped)

print("\nAfter expand_dims and repeat:")
corners = np.repeat(np.expand_dims(swapped, axis=1), 2, axis=1)
print(corners)

print("\nAfter corner adjustments:")
corners[:, 0] -= corner_subimage_half_size
corners[:, 1] += corner_subimage_half_size
print(corners.astype(int))

print("\nOuter corners (first 4):")
print(corners[:4].astype(int))
