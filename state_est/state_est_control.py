# adapted from: brio_state_estimation/brio_state_estimation_nonros.py

import cv2 as cv
import numpy as np
import time

from estimation_pipeline import EstimationPipeline
from  divers import init_capture
import matplotlib.pyplot as plt
import os
import sys

import dynamixel_sdk as dxl
# Add Dynamixel SDK library path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dynamixel_sdk'))

# Dynamixel motor settings
DXL_PORT = '/dev/tty.usbserial-FT79212K'     # Update this with your port
BAUDRATE = 1000000              # Default baudrate for Dynamixel
DXL_IDS = [1, 2]                # Motor IDs for two motors

# Motor control parameters
ADDR_TORQUE_ENABLE = 64          # Torque enable address
ADDR_GOAL_POSITION = 116         # Goal position address
PROTOCOL_VERSION = 2.0           # Dynamixel protocol version
ADDR_PRESENT_POSITION = 132      # Address for Present Position

# Step sizes for motor control with keys
STEP_SIZE_MOTOR1 = 100 
STEP_SIZE_MOTOR2 = 100 


def mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN: 
        ball_pos = np.array([x,y])[::-1]
        estimation_pipeline.measurements.detector.reset(ball_pos)

def init_windows(): 
    winUndist = cv.namedWindow("ori", cv.WINDOW_NORMAL)
    cv.moveWindow("ori", 50,50)
    cv.resizeWindow("ori", 640, 400)
    
def display_instructions():
    print(""" Motor Control Instructions:
        Use the following keys to control the motors:
        w: Increase Motor 1 position by 300 steps
        s: Decrease Motor 1 position by 300 steps
        a: Increase Motor 2 position by 600 steps
        d: Decrease Motor 2 position by 600 steps
        Space: Continue with the current motors positions
        q: Quit the program
        """)
    
    
# MOTOR CONTROL FUNCTIONS


def set_position(portHandler, packetHandler, dxl_id, position):
    """Set the position of a motor."""
    # Ensure position is within safe limits
    position = max(0, min(4095, position))  # Dynamixel position range is 0-4095 for 360-degree rotation
    
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
        portHandler, dxl_id, ADDR_GOAL_POSITION, position)
        
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print(f"Motor {dxl_id} communication error: {packetHandler.getTxRxResult(dxl_comm_result)}")
        return False
    elif dxl_error != 0:
        print(f"Motor {dxl_id} error: {packetHandler.getRxPacketError(dxl_error)}")
        return False
    
    print(f"Motor {dxl_id} target position set to {position}")
    return True

def move_motor_to_position(portHandler, packetHandler, dxl_id, target_position):
    """Move a motor to the target position and wait until it reaches the position."""
    if not set_position(portHandler, packetHandler, dxl_id, target_position):
        print(f"Failed to move motor {dxl_id} to position {target_position}")
        return False

    command_timeout = 5 # max nr of commands to achieve the target position
    command_count = 0
    
    while True:
        current_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
            portHandler, dxl_id, ADDR_PRESENT_POSITION)
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print(f"Communication error on motor {dxl_id}: {packetHandler.getTxRxResult(dxl_comm_result)}")
            return False
        elif dxl_error != 0:
            print(f"Dynamixel error on motor {dxl_id}: {packetHandler.getRxPacketError(dxl_error)}")
            return False

        print(f"Motor {dxl_id} current position: {current_position}")
        if abs(current_position - target_position) < 50:  # Tolerance for position NOTE: 50 is arbitrary, adjust as needed
            print(f"Motor {dxl_id} reached target position: {target_position}")
            break

        command_count += 1
        if command_count >= command_timeout:
            print(f"Motor {dxl_id} did not reach target position within timeout.")
            break

    return True

def init_dynamixel():
    portHandler = dxl.PortHandler(DXL_PORT)
    packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)
    
    # Open port
    if not portHandler.openPort():
        print("Failed to open the port")
        return None, None
    
    # Set port baudrate
    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to change the baudrate")
        return None, None
    
    return portHandler, packetHandler

# Function to read multiple motor positions
def read_motor_positions(portHandler, packetHandler, dxl_ids):
    positions = {}
    for dxl_id in dxl_ids:
        position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
            portHandler, dxl_id, ADDR_PRESENT_POSITION)
        
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print(f"Communication error on motor {dxl_id}: {packetHandler.getTxRxResult(dxl_comm_result)}")
            positions[dxl_id] = None
        elif dxl_error != 0:
            print(f"Dynamixel error on motor {dxl_id}: {packetHandler.getRxPacketError(dxl_error)}")
            positions[dxl_id] = None
        else:
            positions[dxl_id] = position
    
    return positions

#####

if __name__ == "__main__":
     
    ### PARAMS ### gitignore    
    devices = ["CAM", "VIDEO", "IMAGE"]
    DEVICE = devices[0] # "CAM", "VIDEO", "IMAGE"
    IDX_CAM = 0
    VIDEO_PATH = "example_vid.avi" # who is timflueckiger haha

    INIT_VIDEO_FRAME_IDX = 0 # edge case : 1250
    PRINT_MEASUREMENTS = True
    estimation_pipeline = EstimationPipeline(fps = 60, # $$ to change !
                                             estimator="FiniteDiff", 
                                             print_measurements=True, 
                                             show_image=1, 
                                             do_anim_3d=0,
                                             viewpoint="top", # 'top', 'side', 'topandside'
                                             show_subimages_detector=True)
     ##############

    display_instructions()

    cap, width, height = init_capture(DEVICE, IDX_CAM, VIDEO_PATH, INIT_VIDEO_FRAME_IDX)
    print("width, height: ", width, height, " FPS: ", cap.get(cv.CAP_PROP_FPS))
    init_windows()
    cv.setMouseCallback("ori", mouse_click)
    pause = False
    tframe = 0

    # load the markers from markers.csv
    Markers = np.loadtxt("markers.csv", delimiter=",", skiprows=1, usecols=[2, 3])
    
    # scale down the markers by 1/3
    markers = Markers / 3

    portHandler, packetHandler = init_dynamixel()
    if not portHandler or not packetHandler:
        print("Failed to initialize Dynamixel motors. Exiting.")
        exit()

    # Enable torque for all motors
    for dxl_id in DXL_IDS:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
            portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print(f"Failed to enable torque for motor {dxl_id}: {packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Error enabling torque for motor {dxl_id}: {packetHandler.getRxPacketError(dxl_error)}")
        else:
            print(f"Torque enabled for motor {dxl_id}")

    # Read and set the current motor positions
    motor_positions = read_motor_positions(portHandler, packetHandler, DXL_IDS)
    print("Initial motor positions:", motor_positions)

    processed_frames = 0
    while True: 
        
        if not pause:
            ret, frame = cap.read() 
            timenew = time.time() 
            tframe = timenew
            
        processed_frames += 1
        frame = cv.resize(frame, (int(width/3), int(height/3))) 
        print("\n\n\n using frame of shape: ", frame.shape)
        

        
        # Process the frame
        b, g, r = np.mean(np.mean(frame, axis=0), axis=0)
        if g > 100 and b < 40 and r < 40:
            print("SKIP THIS FRAME")
            continue
        
        x_hat, P, inputs, xb, yb = estimation_pipeline.estimate(frame)
        alpha, beta = inputs 
        alpha = np.rad2deg(alpha)
        beta = np.rad2deg(beta)
        
        print("#"*5, "frame number: ", processed_frames, "#"*5)
        print("x_hat:", x_hat)
        print("estimated P:", P)
        print("estimated plate angles: ", alpha, beta)
        
        # display the frame
                # Create an overlay for displaying information
        overlay = frame.copy()
        y_pos = 20

        # Display motor positions
        motor1_position = motor_positions[1]
        motor2_position = motor_positions[2]
        cv.putText(overlay, f"Motor 1 Pos: {motor1_position}", 
                   (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        y_pos += 30
        cv.putText(overlay, f"Motor 2 Pos: {motor2_position}", 
                   (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        y_pos += 30

        # Display the estimated inputs as an overlay
        if inputs is not None:
            alpha, beta = inputs 
            alpha = np.rad2deg(alpha)
            beta = np.rad2deg(beta)
            cv.putText(overlay, f"Alpha: {alpha:.2f}", (10, y_pos), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
            cv.putText(overlay, f"Beta: {beta:.2f}", (10, y_pos), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30

        # Blend the overlay with the original frame
        blended = cv.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Display the blended frame
        cv.imshow("board", blended)

        # Wait for user input and display the pressed key
        key = cv.waitKey(0) & 0xFF  # Use waitKey(0) to ensure synchronization
        print(f"Key pressed: {chr(key)}")
        if key == ord('w'):
            print(f"Increasing Motor 1 position by {STEP_SIZE_MOTOR1} steps")
            motor_positions[1] += STEP_SIZE_MOTOR1
            if not move_motor_to_position(portHandler, packetHandler, 1, motor_positions[1]):
                print("Error moving Motor 1")
        elif key == ord('s'):
            print(f"Decreasing Motor 1 position by {STEP_SIZE_MOTOR1} steps")
            motor_positions[1] -= STEP_SIZE_MOTOR1
            if not move_motor_to_position(portHandler, packetHandler, 1, motor_positions[1]):
                print("Error moving Motor 1")
        elif key == ord('a'):
            print(f"Increasing Motor 2 position by {STEP_SIZE_MOTOR2} steps")
            motor_positions[2] += STEP_SIZE_MOTOR2
            if not move_motor_to_position(portHandler, packetHandler, 2, motor_positions[2]):
                print("Error moving Motor 2")
        elif key == ord('d'):
            print(f"Decreasing Motor 2 position by {STEP_SIZE_MOTOR2} steps")
            motor_positions[2] -= STEP_SIZE_MOTOR2
            if not move_motor_to_position(portHandler, packetHandler, 2, motor_positions[2]):
                print("Error moving Motor 2")
        elif key == ord(' '):  # Space key
            print("Continuing with the current motor positions")
        elif key == ord('q'):
            print("Quitting the program")
            break

    # Disable torque and close port
    for dxl_id in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    portHandler.closePort()

    cap.release()
    cv.destroyAllWindows()