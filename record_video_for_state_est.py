import cv2 as cv
import time

def record_short_video():
    # Video parameters
    output_file = "example_vid.avi"
    duration = 5
    camera_idx = 0
    fps = 60
    
    # Initialize video capture
    cap = cv.VideoCapture(camera_idx)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get camera properties
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv.CAP_PROP_FPS)
    
    # Print resolution and FPS
    print(f"Recording video with resolution: {width}x{height}")
    print(f"Camera FPS: {actual_fps}")
    print(f"Recording FPS: {fps}")
    
    # Create video writer
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Create window for preview
    cv.namedWindow("Recording", cv.WINDOW_NORMAL)
    
    # Record for specified duration
    start_time = time.time()
    print(f"Recording {duration} second video...")
    frame_count = 0
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Save frame to video
        out.write(frame)
        frame_count += 1
        
        # Display the frame
        cv.imshow("Recording", frame)
        
        # Check for exit key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Calculate actual FPS achieved
    elapsed_time = time.time() - start_time
    achieved_fps = frame_count / elapsed_time
    print(f"Recorded {frame_count} frames in {elapsed_time:.2f} seconds")
    print(f"Achieved FPS: {achieved_fps:.2f}")
    
    # Release resources
    out.release()
    cap.release()
    cv.destroyAllWindows()
    
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    record_short_video()
