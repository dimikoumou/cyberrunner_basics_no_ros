import numpy as np

# Test the exact transformation used in detection.py
markers = np.array([[356, 609], [930, 616], [944, 135], [363, 119]])  # Inner corners
print("Input markers:", markers)

corner_subimage_half_size = 17

# Step 1: Reverse x,y coordinates
print("\nStep 1: markers[:, ::-1]")
swapped = markers[:, ::-1]
print("Swapped:", swapped)

# Step 2: expand_dims
print("\nStep 2: expand_dims")
expanded = np.expand_dims(swapped, axis=1)
print("Expanded shape:", expanded.shape)
print("Expanded:", expanded)

# Step 3: repeat
print("\nStep 3: repeat")
corners = np.repeat(expanded, 2, axis=1)
print("Repeated shape:", corners.shape)
print("Repeated:", corners)

# Step 4: corner adjustments
print("\nStep 4: corner adjustments")
corners[:, 0] -= corner_subimage_half_size
corners[:, 1] += corner_subimage_half_size
print("Final corners:", corners.astype(int))

# Compare with what we expect for corner 0
print("\n=== ANALYSIS ===")
print("Original marker 0: (356, 609)")
print("Expected: should crop around row=609, col=356")
print("Actual result:", corners[0].astype(int))
print("This crops from row", corners[0][0][0], "to row", corners[0][1][0])
print("This crops from col", corners[0][0][1], "to col", corners[0][1][1])
