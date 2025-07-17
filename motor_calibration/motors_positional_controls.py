"""
dynamixel_controller.py

This module provides functions for initializing and controlling Dynamixel motors using both current control
and position control modes. It supports a rocking motion by alternating between target positions.

Before using, ensure that the control table addresses, device name, baud rate, and motor IDs match your setup.
"""

from dynamixel_sdk import *  # Import Dynamixel SDK
import time

# Control table addresses (verify these with your motor's documentation)
ADDR_TORQUE_ENABLE = 64       # Address for torque enable
ADDR_OPERATING_MODE = 11      # Address for operating mode
ADDR_GOAL_CURRENT = 102       # Address for goal current (for current control mode)
ADDR_GOAL_POSITION = 116      # Address for goal position (for position control mode)
ADDR_PRESENT_POSITION = 132   # Address for present position

# Operating modes
CURRENT_CONTROL_MODE = 0
POSITION_CONTROL_MODE = 3

# Protocol version for Dynamixel communication (commonly 2.0)
PROTOCOL_VERSION = 2.0

# Motor IDs (set these to your motor IDs) # should be set to 1 and 2 in the dynamixel app.
DXL0_ID = 1
DXL1_ID = 3

# Communication settings (example for macOS; change as needed)
DEVICENAME = '/dev/tty.usbserial-FTA7NMFT'  # Replace with your device name
BAUDRATE = 1000000  # 1M bps

# Torque control flags
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# Position limits for rocking motion
MIN_POSITION = 900  # Reduced range for more gentle movement
MAX_POSITION = 2700 # Reduced range for more gentle movement
CENTER_POSITION = (MIN_POSITION + MAX_POSITION) // 2  # Center position for initialization


def convert_to_unsigned_16(value):
    """
    Convert a signed 16-bit integer into its unsigned 16-bit (two's complement) equivalent.

    This is useful for converting negative current values to the format expected by Dynamixel registers.

    Parameters:
        value (int): The signed 16-bit integer value.

    Returns:
        int: The corresponding unsigned 16-bit value.
    """
    if value < 0:
        return (1 << 16) + value
    return value


def setup_motors(operating_mode=POSITION_CONTROL_MODE):
    """
    Initialize the communication port and packet handler for the Dynamixel motors,
    and enable the torque for both motors.

    Parameters:
        operating_mode (int): The operating mode to set for the motors (default: position control)

    Returns:
        tuple: (portHandler, packetHandler) if setup is successful; otherwise, (None, None).
    """
    # Create a PortHandler for serial communication.
    portHandler = PortHandler(DEVICENAME)
    # Create a PacketHandler to manage protocol-specific operations.
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # Open the serial port.
    if not portHandler.openPort():
        print("Failed to open the port")
        return None, None

    # Set the baud rate for communication.
    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to change the baudrate")
        return None, None
        
    # Set operating mode for both motors
    for dxl_id in [DXL0_ID, DXL1_ID]:
        # Disable torque to change operating mode
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_DISABLE
        )
        
        # Set operating mode
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_OPERATING_MODE,
            operating_mode
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Motor {dxl_id}: Communication error setting mode: {packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Motor {dxl_id}: Rx error setting mode: {packetHandler.getRxPacketError(dxl_error)}")
        else:
            print(f"Motor {dxl_id} operating mode set to {operating_mode}")

    # Enable torque on both motors.
    for dxl_id in [DXL0_ID, DXL1_ID]:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_ENABLE
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Motor {dxl_id}: Communication error: {packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Motor {dxl_id}: Rx error: {packetHandler.getRxPacketError(dxl_error)}")
        else:
            print(f"Motor {dxl_id} torque enabled")

    return portHandler, packetHandler


def set_motor_current(packetHandler, portHandler, motor_id, current):
    """
    Set the goal current for a specified motor.

    The function converts the signed current value into an unsigned 16-bit value,
    which is then written to the motor's goal current register.

    Parameters:
        packetHandler: The PacketHandler instance.
        portHandler: The PortHandler instance.
        motor_id (int): The ID of the motor.
        current (int): The desired current value (can be negative or positive).
    """
    # Convert the signed current value to an unsigned 16-bit integer.
    current_unsigned = convert_to_unsigned_16(current)

    # Write the goal current to the motor's register.
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
        portHandler,
        motor_id,
        ADDR_GOAL_CURRENT,
        current_unsigned
    )
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Motor {motor_id}: Communication error: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"Motor {motor_id}: Rx error: {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"Motor {motor_id} set to current {current}")


def set_motor_position(packetHandler, portHandler, motor_id, position):
    """
    Set the goal position for a specified motor.

    Parameters:
        packetHandler: The PacketHandler instance.
        portHandler: The PortHandler instance.
        motor_id (int): The ID of the motor.
        position (int): The desired position value.
    """
    # Write the goal position to the motor's register
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
        portHandler,
        motor_id,
        ADDR_GOAL_POSITION,
        position
    )
    
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Motor {motor_id}: Communication error: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"Motor {motor_id}: Rx error: {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"Motor {motor_id} set to position {position}")


def move_to_position_smoothly(packetHandler, portHandler, motor_id, target_position, current_position=None, steps=10, step_delay=0.05):
    """
    Move a motor to a target position in small increments for gentle movement.
    
    Parameters:
        packetHandler: The PacketHandler instance.
        portHandler: The PortHandler instance.
        motor_id (int): The ID of the motor.
        target_position (int): The final desired position.
        current_position (int, optional): The starting position. If None, reads from the motor.
        steps (int): Number of intermediate steps to use.
        step_delay (float): Delay between each intermediate step.
    """
    # If current position is not provided, read it from the motor
    if current_position is None:
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
            portHandler, motor_id, ADDR_PRESENT_POSITION)
        
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Failed to read position from motor {motor_id}")
            return
        current_position = dxl_present_position
    
    # Calculate position increment
    position_diff = target_position - current_position
    increment = position_diff / steps
    
    # Move in small increments
    for step in range(1, steps + 1):
        intermediate_position = int(current_position + (increment * step))
        set_motor_position(packetHandler, portHandler, motor_id, intermediate_position)
        time.sleep(step_delay)
    
    # Ensure we reach the exact target position
    set_motor_position(packetHandler, portHandler, motor_id, target_position)


def disable_motors(packetHandler, portHandler):
    """
    Disable torque for both motors.

    First, this function commands a zero current to help clear any hardware faults,
    then disables torque and closes the communication port.

    Parameters:
        packetHandler: The PacketHandler instance.
        portHandler: The PortHandler instance.
    """
    # Clear the current command for each motor.
    for dxl_id in (DXL1_ID, DXL0_ID):
        print(f"Clearing current for Motor {dxl_id}...")
        set_motor_current(packetHandler, portHandler, dxl_id, 0)
    time.sleep(0.1)  # Allow a short delay for the motors to settle

    # Disable torque for each motor.
    for dxl_id in (DXL1_ID, DXL0_ID):
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_DISABLE
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(
                f"Motor {dxl_id}: Communication error while disabling torque: {packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            error_msg = packetHandler.getRxPacketError(dxl_error)
            # If a hardware error is reported, log it and continue.
            if "Hardware error occurred" in error_msg:
                print(f"Motor {dxl_id} reported hardware error when disabling torque, but command sent.")
            else:
                print(f"Motor {dxl_id}: Rx error while disabling torque: {error_msg}")
        else:
            print(f"Motor {dxl_id} torque disabled")

    # Close the communication port.
    portHandler.closePort()


def run_current_control_test(cycles=5, current_value=100, delay=0.50):
    """
    Run a current control test that alternates motor current between a positive and a negative value.

    This function initializes the motors, runs the specified number of test cycles, and then disables the motors.

    Parameters:
        cycles (int): Number of forward/backward cycles to execute.
        current_value (int): The magnitude of current to command (both positive and negative).
        delay (float): Time delay (in seconds) between switching directions.
    """
    portHandler, packetHandler = setup_motors(CURRENT_CONTROL_MODE)
    if portHandler is None or packetHandler is None:
        return

    print("Starting forward/backward current control test.")
    for i in range(cycles):
        print(f"Cycle {i + 1}/{cycles}: Moving motors forward")
        set_motor_current(packetHandler, portHandler, DXL0_ID, current_value)
        set_motor_current(packetHandler, portHandler, DXL1_ID, current_value)
        time.sleep(delay)

        print(f"Cycle {i + 1}/{cycles}: Moving motors backward")
        set_motor_current(packetHandler, portHandler, DXL0_ID, -current_value)
        set_motor_current(packetHandler, portHandler, DXL1_ID, -current_value)
        time.sleep(delay)

    print("Cycles complete. Disabling motors.")
    disable_motors(packetHandler, portHandler)


def run_position_control_test(cycles=5, delay=1.0):
    """
    Run a position control test that creates a rocking motion by alternating
    between two positions.

    Parameters:
        cycles (int): Number of rocking cycles to execute.
        delay (float): Time delay (in seconds) between position changes.
    """
    portHandler, packetHandler = setup_motors(POSITION_CONTROL_MODE)
    if portHandler is None or packetHandler is None:
        return
    
    print("Starting rocking motion position control test.")
    for i in range(cycles):
        print(f"Cycle {i + 1}/{cycles}: Moving motors to forward position")
        set_motor_position(packetHandler, portHandler, DXL0_ID, MAX_POSITION)
        set_motor_position(packetHandler, portHandler, DXL1_ID, MAX_POSITION)
        time.sleep(delay)
        
        print(f"Cycle {i + 1}/{cycles}: Moving motors to backward position")
        set_motor_position(packetHandler, portHandler, DXL0_ID, MIN_POSITION)
        set_motor_position(packetHandler, portHandler, DXL1_ID, MIN_POSITION)
        time.sleep(delay)
    
    print("Cycles complete. Disabling motors.")
    disable_motors(packetHandler, portHandler)


def run_gentle_rocking_test(cycles=5, delay=1.5, steps=15):
    """
    Run a position control test that creates a gentle rocking motion by
    smoothly transitioning between positions.

    Parameters:
        cycles (int): Number of rocking cycles to execute.
        delay (float): Time delay (in seconds) between major position changes.
        steps (int): Number of intermediate steps for smooth movement.
    """
    portHandler, packetHandler = setup_motors(POSITION_CONTROL_MODE)
    if portHandler is None or packetHandler is None:
        return
    
    # First, move both motors to center position to start
    print("Initializing motors to center position...")
    set_motor_position(packetHandler, portHandler, DXL0_ID, CENTER_POSITION)
    set_motor_position(packetHandler, portHandler, DXL1_ID, CENTER_POSITION)
    time.sleep(1.0)
    
    print("Starting gentle rocking motion test.")
    for i in range(cycles):
        print(f"Cycle {i + 1}/{cycles}: Smoothly moving to forward position")
        move_to_position_smoothly(packetHandler, portHandler, DXL0_ID, MAX_POSITION, CENTER_POSITION, steps)
        move_to_position_smoothly(packetHandler, portHandler, DXL1_ID, MAX_POSITION, CENTER_POSITION, steps)
        time.sleep(delay)
        
        print(f"Cycle {i + 1}/{cycles}: Smoothly moving to backward position")
        move_to_position_smoothly(packetHandler, portHandler, DXL0_ID, MIN_POSITION, MAX_POSITION, steps)
        move_to_position_smoothly(packetHandler, portHandler, DXL1_ID, MIN_POSITION, MAX_POSITION, steps)
        time.sleep(delay)
        
        # Return to center position for smoother overall motion
        if i < cycles - 1:  # Don't center on the last cycle
            print(f"Cycle {i + 1}/{cycles}: Returning to center")
            move_to_position_smoothly(packetHandler, portHandler, DXL0_ID, CENTER_POSITION, MIN_POSITION, steps)
            move_to_position_smoothly(packetHandler, portHandler, DXL1_ID, CENTER_POSITION, MIN_POSITION, steps)
            time.sleep(delay/2)
    
    print("Cycles complete. Disabling motors.")
    disable_motors(packetHandler, portHandler)


# Allow the module to be run as a standalone script for testing.
if __name__ == "__main__":
    # Run the gentle rocking motion test
    # run_gentle_rocking_test()
    # Uncomment any of these lines to run other tests
    run_position_control_test()
    # run_current_control_test()