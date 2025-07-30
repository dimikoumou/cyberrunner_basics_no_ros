import cv2
from board_estimation import EstimationPipeline

# Create the estimation pipeline first
pipeline = EstimationPipeline(
    fps=30,
    estimator="FiniteDiff",
    print_measurements=True,
    show_image=True
)

# # Start camera feed
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Live Feed", frame)
    # Run state estimation on every frame
    x_hat, P, inputs, xb, yb = pipeline.estimate(frame)

    # Optionally, draw or overlay stuff onto the frame here
    # (But `pipeline` may already show stuff if show_image=True)

    # Show original frame if needed (optional)
    # 

    # # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cap.release()
cv2.destroyAllWindows()
