import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_region(image_path, top_left, bottom_right, are_coords_for_original=True):
    """
    Load an image, crop the region between top_left and bottom_right coordinates,
    and display both the original image and the cropped region.
    
    Args:
        image_path (str): Path to the image file
        top_left (list): [x, y] coordinates of the top-left corner
        bottom_right (list): [x, y] coordinates of the bottom-right corner
        are_coords_for_original (bool): Whether the provided coordinates are for the original image
    """
    # Load the image
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        # Display original image size
        print(f"Original image shape: {img.shape}")
        h_orig, w_orig = img.shape[:2]
        
        # Initial coordinate check before scaling
        if are_coords_for_original:
            x1, y1 = top_left
            x2, y2 = bottom_right
            if x1 < 0 or x1 >= w_orig or y1 < 0 or y1 >= h_orig or x2 <= x1 or y2 <= y1 or x2 > w_orig or y2 > h_orig:
                print(f"Warning: Coordinates are invalid or outside image bounds.")
                print(f"Image dimensions: width={w_orig}, height={h_orig}")
                print(f"Provided coordinates: top_left={top_left}, bottom_right={bottom_right}")
                print("Attempting to adjust coordinates to fit within image bounds...")
        
        # Scale down the image by a factor of 3
        scale_factor = 3
        scaled_img = cv2.resize(img, (img.shape[1] // scale_factor, img.shape[0] // scale_factor))
        print(f"Scaled image shape: {scaled_img.shape}")
        h_scaled, w_scaled = scaled_img.shape[:2]
        
        # Convert from BGR to RGB for matplotlib display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scaled_img_rgb = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
        
        # Calculate coordinates based on whether they're for original or scaled image
        if are_coords_for_original:
            # Coordinates are for original image
            orig_x1, orig_y1 = top_left
            orig_x2, orig_y2 = bottom_right
            
            # Calculate scaled coordinates
            x1, y1 = orig_x1 // scale_factor, orig_y1 // scale_factor
            x2, y2 = orig_x2 // scale_factor, orig_y2 // scale_factor
        else:
            # Coordinates are for scaled image
            x1, y1 = top_left
            x2, y2 = bottom_right
            
            # Calculate original coordinates
            orig_x1, orig_y1 = x1 * scale_factor, y1 * scale_factor
            orig_x2, orig_y2 = x2 * scale_factor, y2 * scale_factor
        
        # Make sure coordinates are within bounds
        orig_x1_before = orig_x1
        orig_y1_before = orig_y1
        orig_x2_before = orig_x2
        orig_y2_before = orig_y2
        
        orig_x1 = max(0, min(orig_x1, w_orig-1))
        orig_y1 = max(0, min(orig_y1, h_orig-1))
        orig_x2 = max(orig_x1+1, min(orig_x2, w_orig))
        orig_y2 = max(orig_y1+1, min(orig_y2, h_orig))
        
        # Report if coordinates were adjusted
        if orig_x1 != orig_x1_before or orig_y1 != orig_y1_before or orig_x2 != orig_x2_before or orig_y2 != orig_y2_before:
            print("Coordinates were adjusted to fit within image bounds:")
            print(f"Original: ({orig_x1_before}, {orig_y1_before}) to ({orig_x2_before}, {orig_y2_before})")
            print(f"Adjusted: ({orig_x1}, {orig_y1}) to ({orig_x2}, {orig_y2})")
        
        # Recalculate scaled coordinates based on adjusted original coordinates
        x1 = orig_x1 // scale_factor
        y1 = orig_y1 // scale_factor
        x2 = min(w_scaled, (orig_x2 + scale_factor - 1) // scale_factor)
        y2 = min(h_scaled, (orig_y2 + scale_factor - 1) // scale_factor)
        
        # Ensure the second coordinate is greater than the first
        if x2 <= x1:
            x2 = min(x1 + 1, w_scaled)
        if y2 <= y1:
            y2 = min(y1 + 1, h_scaled)
        
        # Crop the image regions
        cropped_scaled = scaled_img_rgb[y1:y2, x1:x2]
        cropped_orig = img_rgb[orig_y1:orig_y2, orig_x1:orig_x2]
        
        # Display images with rectangles
        plt.figure(figsize=(15, 10))
        
        # Original full-sized image with rectangle
        plt.subplot(2, 2, 1)
        plt.imshow(img_rgb)
        plt.title('Original Image')
        plt.gca().add_patch(plt.Rectangle((orig_x1, orig_y1), orig_x2-orig_x1, orig_y2-orig_y1, 
                            edgecolor='red', facecolor='none', linewidth=2))
        
        # Scaled image with rectangle
        plt.subplot(2, 2, 2)
        plt.imshow(scaled_img_rgb)
        plt.title(f'Scaled Image (1/{scale_factor})')
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            edgecolor='red', facecolor='none', linewidth=2))
        
        # Cropped region from original image
        plt.subplot(2, 2, 3)
        plt.imshow(cropped_orig)
        plt.title('Cropped Region (Original)')
        
        # Cropped region from scaled image
        plt.subplot(2, 2, 4)
        plt.imshow(cropped_scaled)
        plt.title('Cropped Region (Scaled)')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Cropped region shape (original): {cropped_orig.shape}")
        print(f"Cropped region shape (scaled): {cropped_scaled.shape}")
        print(f"Original coordinates: ({orig_x1}, {orig_y1}) to ({orig_x2}, {orig_y2})")
        print(f"Scaled coordinates: ({x1}, {y1}) to ({x2}, {y2})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Define the image path and coordinates
    image_path = "undistorted_image.jpg"
    
    # Coordinates for the ROI in the original image (not scaled)
    top_left = [111, 1050]
    bottom_right = [135, 1074]
    
    # Display the cropped region
    # The default for are_coords_for_original is True, so we don't need to specify it
    display_region(image_path, top_left, bottom_right)
