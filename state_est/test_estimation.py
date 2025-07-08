import cv2
from board_estimation import EstimationPipeline

# Load the test image
frame = cv2.imread('../marked_image.jpg')

# Create estimation pipeline
pipeline = EstimationPipeline(
    fps=30,
    estimator="FiniteDiff",
    print_measurements=True,  # This will print detailed measurements
    show_image=True  # This will show the detection visualization
)

# Run estimation
x_hat, P, inputs, xb, yb = pipeline.estimate(frame)

print("\nEstimation Results:")
print(f"Ball position (x, y): ({xb:.3f}, {yb:.3f})")
print(f"Plate angles (α, β): ({inputs[0]*180/3.14:.2f}, {inputs[1]*180/3.14:.2f}) degrees")
print(f"State estimate (x_hat): {x_hat}")
print(f"Covariance matrix (P):\n{P}")