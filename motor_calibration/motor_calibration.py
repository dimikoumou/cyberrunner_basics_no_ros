#!/usr/bin/env python3

import os
import json
import time
import sys
import termios
import tty
import matplotlib.pyplot as plt
from dynamixel_sdk import *

# Control table address for Dynamixel motors
ADDR_OPERATING_MODE = 11          # Operating mode
ADDR_TORQUE_ENABLE  = 64          # Torque enable
ADDR_GOAL_POSITION  = 116         # Goal position
ADDR_PRESENT_POSITION = 132       # Present position
ADDR_CURRENT_LIMIT = 38           # Current limit address
ADDR_GOAL_CURRENT = 102           # Goal current address
ADDR_PRESENT_CURRENT = 126        # Present current address (for reading current)
ADDR_DRIVE_MODE = 10              # Drive mode address

# Data Byte Length
LEN_GOAL_POSITION = 4
LEN_PRESENT_POSITION = 4
LEN_CURRENT_LIMIT = 2
LEN_GOAL_CURRENT = 2
LEN_PRESENT_CURRENT = 2

# Protocol version
PROTOCOL_VERSION = 2.0

# Default setting
DXL1_ID = 1                       # Dynamixel#1 ID
DXL2_ID = 2                       # Dynamixel#2 ID
BAUDRATE = 1000000
DEVICENAME = '/dev/tty.usbserial-FT79212K'   # Default device name (update this for your system)

# Position control mode
POS_CONTROL_MODE = 3

# Step sizes - much larger for motor 2 due to friction
STEP_SIZE_MOTOR1 = 100
STEP_SIZE_MOTOR2 = 400  # Increased step size for motor 2

# Step sizes for limit detection (smaller for precision)
LIMIT_STEP_MOTOR1 = 50
LIMIT_STEP_MOTOR2 = 100

# Current limits (0-1941 for XM series or 0-1193 for X series, check your model)
CURRENT_LIMIT_MOTOR1 = 800        # Default current limit for motor 1
CURRENT_LIMIT_MOTOR2 = 1800       # Increased current limit for motor 2 (almost max)
GOAL_CURRENT_MOTOR1 = 800         # Strong current for motor 1
GOAL_CURRENT_MOTOR2 = 1800        # Strong current for motor 2

# Current threshold for limit detection (adjust based on testing)
CURRENT_THRESHOLD_MOTOR1 = 200    # Threshold to detect limits for motor 1
CURRENT_THRESHOLD_MOTOR2 = 400    # Threshold to detect limits for motor 2

# Define position limits for safety
DXL_MINIMUM_POSITION_VALUE = 0      # Minimum limit
DXL_MAXIMUM_POSITION_VALUE = 4000   # Maximum limit

def getch():
    """Get a single character from the user"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def verify_motor_connection(portHandler, packetHandler, dxl_id):
    """Verify that we can communicate with the motor"""
    print(f"Verifying connection to motor ID {dxl_id}...")
    
    # Check if we can read from the motor
    dxl_model_number, dxl_comm_result, dxl_error = packetHandler.ping(portHandler, dxl_id)
    
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Failed to ping motor {dxl_id}: {packetHandler.getTxRxResult(dxl_comm_result)}")
        return False
    elif dxl_error != 0:
        print(f"Error from motor {dxl_id}: {packetHandler.getRxPacketError(dxl_error)}")
        return False
    else:
        print(f"Motor {dxl_id} responding with model number: {dxl_model_number}")
        return True

def setup_dynamixel():
    """Setup connection to Dynamixel motors"""
    # Initialize PortHandler
    portHandler = PortHandler(DEVICENAME)

    # Initialize PacketHandler
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # Open port
    try:
        portHandler.openPort()
        print("Succeeded to open the port")
    except:
        print("Failed to open the port")
        print("Make sure the device name is correct and the device is connected")
        quit()

    # Set port baudrate
    try:
        portHandler.setBaudRate(BAUDRATE)
        print("Succeeded to change the baudrate")
    except:
        print("Failed to change the baudrate")
        quit()

    # Explicitly verify each motor connection
    motor1_ok = verify_motor_connection(portHandler, packetHandler, DXL1_ID)
    motor2_ok = verify_motor_connection(portHandler, packetHandler, DXL2_ID)
    
    if not motor1_ok or not motor2_ok:
        print("WARNING: One or more motors not responding!")
        print("Motor 1 status:", "OK" if motor1_ok else "FAILED")
        print("Motor 2 status:", "OK" if motor2_ok else "FAILED")
        if not motor1_ok and not motor2_ok:
            print("Both motors failed to respond. Exiting.")
            quit()
        user_input = input("Continue with available motors? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting.")
            quit()

    # Special setup for each motor with appropriate current limits
    if motor1_ok:
        print("Initializing Motor 1...")
        # Disable torque to change settings
        packetHandler.write1ByteTxRx(portHandler, DXL1_ID, ADDR_TORQUE_ENABLE, 0)
        
        # Set operating mode to position control
        packetHandler.write1ByteTxRx(portHandler, DXL1_ID, ADDR_OPERATING_MODE, POS_CONTROL_MODE)
        
        # Set current limit for motor 1 (if applicable for your model)
        try:
            packetHandler.write2ByteTxRx(portHandler, DXL1_ID, ADDR_CURRENT_LIMIT, CURRENT_LIMIT_MOTOR1)
            print(f"Motor 1 current limit set to {CURRENT_LIMIT_MOTOR1}")
        except:
            print("Could not set current limit for Motor 1 - may not be supported by this model")
        
        # Enable torque
        packetHandler.write1ByteTxRx(portHandler, DXL1_ID, ADDR_TORQUE_ENABLE, 1)
        print("Motor 1 initialized in position control mode")
    
    if motor2_ok:
        print("Initializing Motor 2 with HIGHER POWER settings...")
        # Disable torque to change settings
        packetHandler.write1ByteTxRx(portHandler, DXL2_ID, ADDR_TORQUE_ENABLE, 0)
        
        # Set operating mode to position control
        packetHandler.write1ByteTxRx(portHandler, DXL2_ID, ADDR_OPERATING_MODE, POS_CONTROL_MODE)
        
        # Set drive mode - try to disable current protection if possible
        try:
            drive_mode, _, _ = packetHandler.read1ByteTxRx(portHandler, DXL2_ID, ADDR_DRIVE_MODE)
            # Set bit 0 to 1 for reverse direction if needed
            # Set bit 2 to 1 to disable acceleration profile if it helps with friction
            packetHandler.write1ByteTxRx(portHandler, DXL2_ID, ADDR_DRIVE_MODE, drive_mode | 0x04)
            print(f"Motor 2 drive mode updated to bypass acceleration profile")
        except:
            print("Could not modify drive mode for Motor 2")
        
        # Set MAXIMUM current limit for motor 2 to overcome friction
        try:
            packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_CURRENT_LIMIT, CURRENT_LIMIT_MOTOR2)
            print(f"Motor 2 current limit set to MAXIMUM {CURRENT_LIMIT_MOTOR2}")
        except:
            print("Could not set current limit for Motor 2 - may not be supported by this model")
            
        # Try setting goal current if applicable
        try:
            packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_GOAL_CURRENT, GOAL_CURRENT_MOTOR2)
            print(f"Motor 2 goal current set to MAXIMUM {GOAL_CURRENT_MOTOR2}")
        except:
            print("Could not set goal current for Motor 2 - may not be supported by this model")
        
        # Enable torque
        packetHandler.write1ByteTxRx(portHandler, DXL2_ID, ADDR_TORQUE_ENABLE, 1)
        print("Motor 2 initialized with maximum power settings")

    return portHandler, packetHandler

def read_position(portHandler, packetHandler, dxl_id):
    """Read current position of a motor"""
    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
        portHandler, dxl_id, ADDR_PRESENT_POSITION)
        
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        return None
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
        return None
        
    return dxl_present_position

def read_current(portHandler, packetHandler, dxl_id):
    """Read present current of a motor"""
    dxl_present_current, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(
        portHandler, dxl_id, ADDR_PRESENT_CURRENT)
        
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        return None
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
        return None
    
    # Convert from raw value to signed (handle 2's complement)
    if dxl_present_current > 32767:
        dxl_present_current -= 65536
    
    return abs(dxl_present_current)  # Return absolute value for threshold detection

def set_position(portHandler, packetHandler, dxl_id, position):
    """Set the position of a motor"""
    # Ensure position is within safe limits
    if position < DXL_MINIMUM_POSITION_VALUE:
        position = DXL_MINIMUM_POSITION_VALUE
    elif position > DXL_MAXIMUM_POSITION_VALUE:
        position = DXL_MAXIMUM_POSITION_VALUE
    
    print(f"Setting motor {dxl_id} to position {position}")
    
    # For motor 2, special handling to overcome friction
    if dxl_id == DXL2_ID:
        # Force re-enable torque with maximum power
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
        
        # Set max current for this move
        try:
            packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_GOAL_CURRENT, GOAL_CURRENT_MOTOR2)
        except:
            pass  # Ignore if not supported
    
    # Try sending the position command
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
        portHandler, dxl_id, ADDR_GOAL_POSITION, position)
        
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        return False
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
        return False
    
    # Small delay to allow the motor to start moving
    time.sleep(0.1)
    return True

def detect_motor_limits(portHandler, packetHandler, dxl_id, center_position):
    """Detect min and max positions for a motor based on current spikes"""
    print(f"\nDetecting position limits for motor {dxl_id}...")
    
    # For storing position and current data for plotting
    up_positions = []
    up_currents = []
    down_positions = []
    down_currents = []
    
    # Set step size and threshold based on motor ID
    step_size = LIMIT_STEP_MOTOR1 if dxl_id == DXL1_ID else LIMIT_STEP_MOTOR2
    current_threshold = CURRENT_THRESHOLD_MOTOR1 if dxl_id == DXL1_ID else CURRENT_THRESHOLD_MOTOR2
    
    # Start at center position
    current_pos = center_position
    set_position(portHandler, packetHandler, dxl_id, current_pos)
    time.sleep(1.0)  # Wait for motor to reach position
    
    # Find maximum position (moving up)
    print(f"Finding maximum position for motor {dxl_id} (moving up)...")
    max_position = None
    base_current = read_current(portHandler, packetHandler, dxl_id)
    if base_current is None:
        print(f"Could not read current from motor {dxl_id}")
        return None, None, {}
    
    print(f"Base current: {base_current}")
    
    # Record starting point
    up_positions.append(current_pos)
    up_currents.append(base_current)
    
    while True:
        # Move up by step size
        current_pos += step_size
        if current_pos > DXL_MAXIMUM_POSITION_VALUE:
            print("Reached software position limit")
            max_position = DXL_MAXIMUM_POSITION_VALUE
            break
            
        set_position(portHandler, packetHandler, dxl_id, current_pos)
        time.sleep(0.5)  # Wait for motor to move
        
        # Read current
        current = read_current(portHandler, packetHandler, dxl_id)
        if current is None:
            continue
        
        # Store position and current for plotting
        up_positions.append(current_pos)
        up_currents.append(current)
            
        print(f"Position: {current_pos}, Current: {current}")
        
        # Check if current exceeds threshold
        if current > (base_current + current_threshold):
            print(f"Current spike detected at position {current_pos}")
            max_position = current_pos - step_size  # Use position before spike
            break
    
    # Return to center
    print(f"Moving motor {dxl_id} back to center...")
    set_position(portHandler, packetHandler, dxl_id, center_position)
    time.sleep(1.0)
    
    # Find minimum position (moving down)
    print(f"Finding minimum position for motor {dxl_id} (moving down)...")
    min_position = None
    current_pos = center_position
    base_current = read_current(portHandler, packetHandler, dxl_id)
    
    # Record starting point for downward movement
    down_positions.append(current_pos)
    down_currents.append(base_current)
    
    while True:
        # Move down by step size
        current_pos -= step_size
        if current_pos < DXL_MINIMUM_POSITION_VALUE:
            print("Reached software position limit")
            min_position = DXL_MINIMUM_POSITION_VALUE
            break
            
        set_position(portHandler, packetHandler, dxl_id, current_pos)
        time.sleep(0.5)  # Wait for motor to move
        
        # Read current
        current = read_current(portHandler, packetHandler, dxl_id)
        if current is None:
            continue
        
        # Store position and current for plotting
        down_positions.append(current_pos)
        down_currents.append(current)
            
        print(f"Position: {current_pos}, Current: {current}")
        
        # Check if current exceeds threshold
        if current > (base_current + current_threshold):
            print(f"Current spike detected at position {current_pos}")
            min_position = current_pos + step_size  # Use position before spike
            break
    
    # Return to center
    print(f"Moving motor {dxl_id} back to center...")
    set_position(portHandler, packetHandler, dxl_id, center_position)
    
    print(f"Motor {dxl_id} limits detected - Min: {min_position}, Max: {max_position}")
    
    # Return collected data for plotting
    measurements = {
        "up_positions": up_positions,
        "up_currents": up_currents,
        "down_positions": down_positions,
        "down_currents": down_currents
    }
    
    return min_position, max_position, measurements

def generate_current_plots(motor1_data, motor2_data):
    """Generate plots showing current vs position for both motors"""
    plt.figure(figsize=(12, 10))
    
    # Plot for Motor 1
    plt.subplot(2, 1, 1)
    plt.plot(motor1_data["up_positions"], motor1_data["up_currents"], 'b-', label='Moving Up')
    plt.plot(motor1_data["down_positions"], motor1_data["down_currents"], 'r-', label='Moving Down')
    plt.title('Motor 1: Current vs Position')
    plt.xlabel('Position')
    plt.ylabel('Current')
    plt.legend()
    plt.grid(True)
    
    # Plot for Motor 2
    plt.subplot(2, 1, 2)
    plt.plot(motor2_data["up_positions"], motor2_data["up_currents"], 'b-', label='Moving Up')
    plt.plot(motor2_data["down_positions"], motor2_data["down_currents"], 'r-', label='Moving Down')
    plt.title('Motor 2: Current vs Position')
    plt.xlabel('Position')
    plt.ylabel('Current')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("motor_current_vs_position.png")
    print("Current vs Position plot saved as 'motor_current_vs_position.png'")
    
    # Show the plot
    plt.show()

def main():
    portHandler, packetHandler = setup_dynamixel()
    
    # Read initial positions
    motor1_position = read_position(portHandler, packetHandler, DXL1_ID)
    motor2_position = read_position(portHandler, packetHandler, DXL2_ID)
    
    if motor1_position is None or motor2_position is None:
        print("Error reading initial motor positions. Exiting.")
        return

    print("\nMotor Calibration Tool")
    print("----------------------")
    print("Please center the board as accurately as possible using the keyboard:")
    print(f"w: Motor 1 + ({STEP_SIZE_MOTOR1} steps)     s: Motor 1 - ({STEP_SIZE_MOTOR1} steps)")
    print(f"a: Motor 2 + ({STEP_SIZE_MOTOR2} steps)     d: Motor 2 - ({STEP_SIZE_MOTOR2} steps)")
    print("Note: Motor 2 requires larger movements due to friction")
    print("Press Enter when the calibration is acceptable.")
    print("\nCurrent position:")
    print(f"Motor 1: {motor1_position}   Motor 2: {motor2_position}")

    while True:
        key = getch()
        print(f"\nKey pressed: {key}")  # Debug output

        if key == '\r': # Enter key
            break
        elif key == 'w':
            print("Moving Motor 1 up")
            motor1_position += STEP_SIZE_MOTOR1
            if set_position(portHandler, packetHandler, DXL1_ID, motor1_position):
                print("Motor 1 moving up...")
        elif key == 's':
            print("Moving Motor 1 down")
            motor1_position -= STEP_SIZE_MOTOR1
            if set_position(portHandler, packetHandler, DXL1_ID, motor1_position):
                print("Motor 1 moving down...")
        elif key == 'a':
            print("Moving Motor 2 up with HIGH POWER")
            motor2_position += STEP_SIZE_MOTOR2
            if set_position(portHandler, packetHandler, DXL2_ID, motor2_position):
                print("Motor 2 moving up with maximum power...")
                time.sleep(0.8)
        elif key == 'd':
            print("Moving Motor 2 down with HIGH POWER")
            motor2_position -= STEP_SIZE_MOTOR2
            if set_position(portHandler, packetHandler, DXL2_ID, motor2_position):
                print("Motor 2 moving down with maximum power...")
                time.sleep(0.8)
        else:
            print(f"Unrecognized key: {key}")
            continue
        
        time.sleep(0.3)
        
        print("Reading current positions...")
        current_pos1 = read_position(portHandler, packetHandler, DXL1_ID)
        current_pos2 = read_position(portHandler, packetHandler, DXL2_ID)
        
        if current_pos1 is not None:
            motor1_position = current_pos1
        if current_pos2 is not None:
            motor2_position = current_pos2
            
        print(f"Motor 1: {motor1_position}   Motor 2: {motor2_position}")

    motor1_center = read_position(portHandler, packetHandler, DXL1_ID)
    motor2_center = read_position(portHandler, packetHandler, DXL2_ID)

    print("\n\nManual centering completed!")
    print(f"Center position - Motor 1: {motor1_center}   Motor 2: {motor2_center}")
    
    print("\nStarting automatic limit detection...")
    print("This will move both motors to find their limits. Please wait...")
    
    motor1_min, motor1_max, motor1_data = detect_motor_limits(portHandler, packetHandler, DXL1_ID, motor1_center)
    motor2_min, motor2_max, motor2_data = detect_motor_limits(portHandler, packetHandler, DXL2_ID, motor2_center)
    
    print("\nReturning to center positions...")
    set_position(portHandler, packetHandler, DXL1_ID, motor1_center)
    set_position(portHandler, packetHandler, DXL2_ID, motor2_center)
    
    calibration_data = {
        "motor1_id": DXL1_ID,
        "motor1_center_position": motor1_center,
        "motor1_min_position": motor1_min,
        "motor1_max_position": motor1_max,
        "motor2_id": DXL2_ID,
        "motor2_center_position": motor2_center,
        "motor2_min_position": motor2_min,
        "motor2_max_position": motor2_max
    }
    
    calibration_file = "motor_calibration.json"
    with open(calibration_file, "w") as f:
        json.dump(calibration_data, f, indent=4)
    
    print(f"\nCalibration data saved to {calibration_file}")
    print("Summary:")
    print(f"Motor 1 - Center: {motor1_center}, Min: {motor1_min}, Max: {motor1_max}")
    print(f"Motor 2 - Center: {motor2_center}, Min: {motor2_min}, Max: {motor2_max}")
    
    # Generate plots from collected data
    print("\nGenerating current vs position plots...")
    generate_current_plots(motor1_data, motor2_data)
    
    for dxl_id in [DXL1_ID, DXL2_ID]:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    
    portHandler.closePort()

if __name__ == "__main__":
    main()
