import cv2
import numpy as np
import os
import json
from tqdm import tqdm

def adjust_brightness(image, factor):
    """
    Adjust the brightness of the image by a factor.
    factor > 1 increases brightness, factor < 1 decreases brightness.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def rotate_image(image, angle, center=None):
    """
    Rotate the image around center by the given angle.
    Returns the rotated image and the rotation matrix.
    """
    height, width = image.shape[:2]
    if center is None:
        center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image, rotation_matrix

def add_shading(image, direction="top", intensity=0.3):
    """
    Add shading to the image.
    direction can be "top", "bottom", "left", or "right".
    intensity controls the strength of the shading.
    """
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=np.float32)
    
    if direction == "top":
        for y in range(height):
            mask[y, :] = 1 - intensity * (1 - y / height)
    elif direction == "bottom":
        for y in range(height):
            mask[y, :] = 1 - intensity * y / height
    elif direction == "left":
        for x in range(width):
            mask[:, x] = 1 - intensity * (1 - x / width)
    elif direction == "right":
        for x in range(width):
            mask[:, x] = 1 - intensity * x / width
    
    # Apply mask to each channel
    result = image.copy().astype(np.float32)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * mask
    
    return np.clip(result, 0, 255).astype(np.uint8)

def transform_points(points, rotation_matrix=None):
    """
    Transform points using the given rotation matrix.
    If no rotation matrix is provided, return the original points.
    """
    if rotation_matrix is None:
        return points
    
    transformed_points = []
    for point in points:
        # Convert to homogeneous coordinates
        p = np.array([[point[0]], [point[1]], [1]])
        # Apply transformation
        p_transformed = np.dot(rotation_matrix, p)
        transformed_points.append((int(p_transformed[0][0]), int(p_transformed[1][0])))
    
    return transformed_points

def augment_dataset(dataset_dir):
    """
    Augment the dataset with various transformations.
    """
    # Load original dataset
    with open(os.path.join(dataset_dir, "markers_dataset.json"), 'r') as f:
        original_dataset = json.load(f)
    
    # Create directory for augmented dataset
    augmented_dir = os.path.join(dataset_dir, "augmented")
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)
        os.makedirs(os.path.join(augmented_dir, "images"))
    
    augmented_dataset = []
    img_idx = 0
    
    # Add original images to the augmented dataset
    for item in original_dataset:
        img_path = os.path.join(dataset_dir, "images", item["image"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Save original image to augmented directory
        aug_img_filename = f"img_{img_idx:05d}.png"
        aug_img_path = os.path.join(augmented_dir, "images", aug_img_filename)
        cv2.imwrite(aug_img_path, img)
        
        # Add to augmented dataset
        augmented_dataset.append({
            "image": aug_img_filename,
            "centers": item["centers"],
            "augmentation": "original"
        })
        img_idx += 1
    
    # Parameters for augmentations
    brightness_factors = [0.7, 0.85, 1.15, 1.3]  # Dimmer and brighter
    rotation_angles = [-10, -5, 5, 10]  # Slight rotations
    shading_directions = ["top", "bottom", "left", "right"]
    
    print(f"Creating augmented dataset from {len(original_dataset)} original images...")
    
    # Process each original image with augmentations
    for item in tqdm(original_dataset):
        img_path = os.path.join(dataset_dir, "images", item["image"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        centers = item["centers"]
        
        # 1. Brightness adjustment
        for factor in brightness_factors:
            aug_img = adjust_brightness(img, factor)
            aug_img_filename = f"img_{img_idx:05d}.png"
            aug_img_path = os.path.join(augmented_dir, "images", aug_img_filename)
            cv2.imwrite(aug_img_path, aug_img)
            
            augmented_dataset.append({
                "image": aug_img_filename,
                "centers": centers,
                "augmentation": f"brightness_{factor}"
            })
            img_idx += 1
        
        # 2. Rotation
        for angle in rotation_angles:
            aug_img, rotation_matrix = rotate_image(img, angle)
            aug_img_filename = f"img_{img_idx:05d}.png"
            aug_img_path = os.path.join(augmented_dir, "images", aug_img_filename)
            cv2.imwrite(aug_img_path, aug_img)
            
            # Transform marker positions
            transformed_centers = transform_points(centers, rotation_matrix)
            
            augmented_dataset.append({
                "image": aug_img_filename,
                "centers": transformed_centers,
                "augmentation": f"rotation_{angle}"
            })
            img_idx += 1
        
        # 3. Shading
        for direction in shading_directions:
            aug_img = add_shading(img, direction)
            aug_img_filename = f"img_{img_idx:05d}.png"
            aug_img_path = os.path.join(augmented_dir, "images", aug_img_filename)
            cv2.imwrite(aug_img_path, aug_img)
            
            augmented_dataset.append({
                "image": aug_img_filename,
                "centers": centers,
                "augmentation": f"shading_{direction}"
            })
            img_idx += 1
    
    # Save the augmented dataset as JSON
    with open(os.path.join(augmented_dir, "augmented_dataset.json"), 'w') as f:
        json.dump(augmented_dataset, f, indent=2)
    
    print(f"Created augmented dataset with {len(augmented_dataset)} images.")

if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    augment_dataset(dataset_dir)
