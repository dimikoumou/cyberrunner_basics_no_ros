
import cv2
import numpy as np
from state_est.board_estimation import EstimationPipeline
# This is a placeholder for the actual motor control library
# You will need to replace this with the actual library you are using
class Motor:
    def __init__(self, port, baud_rate):
        self.port = port
        self.baud_rate = baud_rate
        self.motor_id = 2
        print(f"Motor initialized on port {self.port} with baud rate {self.baud_rate}")
    def move(self, motor_id, position):
        if motor_id == self.motor_id:
            print(f"Moving motor {motor_id} to position {position}")
    def stop(self, motor_id):
        if motor_id == self.motor_id:
            print(f"Stopping motor {motor_id}")
def main():
    # Initialize the camera
    cap = cv2.VideoCapture(1)
    # Initialize the estimation pipeline
    pipeline = EstimationPipeline(
        fps=30,
        estimator="FiniteDiff",
        print_measurements=True,
        show_image=True
    )
    # Initialize the motor
    # IMPORTANT: Replace with the actual port for your motor controller
    motor = Motor(port='/dev/tty.usbmodem14201', baud_rate=1000000)
    motor_id_to_control = 2
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        x_hat, P, inputs, xb, yb = pipeline.estimate(frame)
        # Check if the ball is detected (xb and yb are not NaN)
        if np.isnan(xb) or np.isnan(yb):
            # Ball not found, move the motor
            motor.move(motor_id_to_control, 100)  # Move to a predefined position
        else:
            # Ball found, stop the motor
            motor.stop(motor_id_to_control)
        # Display the frame
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

