import cv2
import numpy as np
import os
import json

def detect_blue_markers(frame):
    """
    Detect blue markers in a frame and return their center coordinates.
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define blue color range in HSV
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    
    # Threshold the image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area to get the 8 largest blue regions
    min_area = 20  # Adjust as needed
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:8]
    
    # Get centers
    centers = []
    for contour in valid_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
    
    # If we don't find exactly 8 markers, return None
    if len(centers) != 8:
        return None
    
    # Sort centers by x-coordinate to ensure consistent ordering
    centers = sorted(centers, key=lambda x: x[0])
    
    return centers

def manual_annotation(frame):
    """
    Allow user to manually annotate blue markers when automatic detection fails.
    """
    centers = []
    clone = frame.copy()
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            centers.append((x, y))
            # Draw a circle at the clicked point with a number
            cv2.circle(clone, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(clone, str(len(centers)), (x+10, y+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update instruction overlay with current progress
            update_instruction_overlay()
            
            print(f"Point {len(centers)} added at ({x}, {y})")
    
    def update_instruction_overlay():
        # Create a semi-transparent overlay for instructions
        overlay = clone.copy()
        
        # Create a dark semi-transparent rectangle at the bottom
        h, w = overlay.shape[:2]
        cv2.rectangle(overlay, (0, h-120), (w, h), (0, 0, 0), -1)
        
        # Add instructions and progress text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f'Selected {len(centers)}/8 markers', (20, h-80), 
                    font, 0.8, (0, 255, 0), 2)
        
        if len(centers) < 8:
            cv2.putText(overlay, 'Click on the blue markers', (20, h-50), 
                        font, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, 'All markers selected! Press ESC to continue', (20, h-50), 
                        font, 0.8, (0, 255, 0), 2)
            
        cv2.putText(overlay, 'Press R to reset, ESC when done', (20, h-20), 
                    font, 0.8, (0, 255, 0), 2)
        
        # Apply the overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, clone, 1 - alpha, 0, clone)
        
        # Display the updated image
        cv2.imshow('Manual Annotation', clone)
    
    # Set up the window for manual annotation
    cv2.namedWindow('Manual Annotation')
    cv2.setMouseCallback('Manual Annotation', mouse_callback)
    
    # Initial instruction overlay
    update_instruction_overlay()
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # ESC key to finish annotation
        if key == 27:
            if len(centers) == 8:
                break
            else:
                print(f"You need to select exactly 8 markers. Currently selected: {len(centers)}")
                # Flash the instruction text to alert the user
                flash_clone = clone.copy()
                h, w = flash_clone.shape[:2]
                cv2.rectangle(flash_clone, (0, h-120), (w, h), (0, 0, 255), -1)
                cv2.putText(flash_clone, f'Please select exactly 8 markers! ({len(centers)}/8 selected)', 
                           (20, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow('Manual Annotation', flash_clone)
                cv2.waitKey(1000)  # Show alert for 1 second
                update_instruction_overlay()
        
        # R key to reset annotation
        if key == ord('r'):
            centers = []
            clone = frame.copy()
            update_instruction_overlay()
            print("Annotation reset. Please select 8 points.")
    
    cv2.destroyWindow('Manual Annotation')
    
    # Sort centers by x-coordinate to ensure consistent ordering
    centers = sorted(centers, key=lambda x: x[0])
    
    return centers

def process_video(video_path, output_dir):
    """
    Process video frame by frame and extract blue marker positions.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "images"))
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    dataset = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect blue markers automatically first
        centers = detect_blue_markers(frame)
        
        # If automatic detection fails, use manual annotation
        if centers is None:
            print(f"Frame {frame_idx}: Automatic detection failed. Switching to manual annotation.")
            centers = manual_annotation(frame)
            annotation_type = "manual"
        else:
            annotation_type = "automatic"
        
        # If we have valid centers (either automatic or manual)
        if centers and len(centers) == 8:
            # Save the frame as an image
            img_filename = f"frame_{frame_idx:03d}.png"
            img_path = os.path.join(output_dir, "images", img_filename)
            cv2.imwrite(img_path, frame)
            
            # Add to dataset
            dataset.append({
                "image": img_filename,
                "centers": centers,
                "annotation_type": annotation_type
            })
            
            # Draw centers on the frame for visualization
            display_frame = frame.copy()
            for cx, cy in centers:
                cv2.circle(display_frame, (cx, cy), 5, (0, 255, 0), -1)
            
            # Display the frame with detected markers
            cv2.imshow('Frame with detected markers', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"Frame {frame_idx}: No valid markers detected.")
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Save the dataset as JSON
    with open(os.path.join(output_dir, "markers_dataset.json"), 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Processed {frame_idx} frames and saved {len(dataset)} valid frames with 8 markers.")
    print(f"Automatic annotations: {sum(1 for item in dataset if item['annotation_type'] == 'automatic')}")
    print(f"Manual annotations: {sum(1 for item in dataset if item['annotation_type'] == 'manual')}")

if __name__ == "__main__":
    video_path = "state_est/example_vid.avi"
    # Ensure the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist at {video_path}")
        exit(1)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    process_video(video_path, output_dir)
