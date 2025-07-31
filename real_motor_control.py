
import cv2
import numpy as np
from state_est.board_estimation import EstimationPipeline
from dynamixel_sdk import *

# --- Dynamixel Motor Configuration ---
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
DXL_ID = 2
BAUDRATE = 1000000
DEVICENAME = '/dev/tty.usbserial-FTA2U13I'  # Change this to your device name
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

class RealMotor:
    def __init__(self, port, baud_rate):
        self.portHandler = PortHandler(port)
        self.packetHandler = PacketHandler(2.0)
        if not self.portHandler.openPort():
            print("Failed to open the port")
            exit()
        if not self.portHandler.setBaudRate(baud_rate):
            print("Failed to change the baudrate")
            exit()
        self.enable_torque(True)

    def enable_torque(self, enable):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 1 if enable else 0
        )
        if dxl_comm_result != 0 or dxl_error != 0:
            print("Failed to enable/disable torque")

    def move(self, position):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, DXL_ID, ADDR_GOAL_POSITION, position
        )
        if dxl_comm_result != 0 or dxl_error != 0:
            print("Failed to move the motor")

    def stop(self):
        # Forcing the motor to hold its current position
        present_position, _, _ = self.packetHandler.read4ByteTxRx(
            self.portHandler, DXL_ID, ADDR_PRESENT_POSITION
        )
        self.move(present_position)

    def close(self):
        self.enable_torque(False)
        self.portHandler.closePort()

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
    motor = RealMotor(port=DEVICENAME, baud_rate=BAUDRATE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            x_hat, P, inputs, xb, yb = pipeline.estimate(frame)

            # Check if the ball is detected (xb and yb are not NaN)
            if np.isnan(xb) or np.isnan(yb):
                # Ball not found, move the motor
                motor.move(1000)  # Move to a predefined position
            else:
                # Ball found, stop the motor by holding the current position
                motor.stop()

            # Display the frame
            cv2.imshow("Live Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Clean up resources
        motor.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

