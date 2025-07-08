import cv2
import os

def count_frames_in_video(video_path):
    """
    Count the number of frames in a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        int: Number of frames in the video
    """
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: File does not exist at {video_path}")
        return -1
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return -1
    
    # Get frame count using the CAP_PROP_FRAME_COUNT property
    property_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get video FPS
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Reset video to beginning for manual counting
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Manually count frames (more reliable for certain formats)
    manual_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        manual_count += 1
    
    # Release the video capture object
    video.release()
    
    print(f"Video path: {video_path}")
    print(f"FPS: {fps}")
    print(f"OpenCV property reports: {property_frame_count} frames")
    print(f"Manual counting found: {manual_count} frames")
    
    return manual_count

if __name__ == "__main__":
    video_path = "state_est/example_vid.avi"
    frame_count = count_frames_in_video(video_path)
    
    if frame_count > 0:
        print(f"\nThe video contains {frame_count} frames")
    else:
        print("\nFailed to count frames in the video")
