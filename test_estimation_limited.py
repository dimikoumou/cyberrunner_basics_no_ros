import cv2
import sys
sys.path.append('state_est')

from state_est.board_estimation import EstimationPipeline

# Create the estimation pipeline first
pipeline = EstimationPipeline(
    fps=30,
    estimator="FiniteDiff",
    print_measurements=True,
    show_image=True
)

# Start camera feed
cap = cv2.VideoCapture(1)

frame_count = 0
max_frames = 5  # Process only 5 frames for testing

print("Starting estimation test - processing 5 frames...")

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    print(f"\n=== Processing frame {frame_count + 1}/{max_frames} ===")
    
    try:
        # Run state estimation on every frame
        x_hat, P, inputs, xb, yb = pipeline.estimate(frame)
        
        print(f"Ball position: ({xb:.3f}, {yb:.3f})")
        print(f"Plate angles: ({inputs[0]*180/3.14159:.2f}°, {inputs[1]*180/3.14159:.2f}°)")
        print(f"State estimate: {x_hat}")
        
    except Exception as e:
        print(f"Error during estimation: {e}")
        import traceback
        traceback.print_exc()
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("\nTest completed!")
