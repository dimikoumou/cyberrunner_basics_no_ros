"""
dynamixel_controller.py

This module provides functions for initializing and controlling Dynamixel motors using current control mode.
It abstracts the communication setup, current control commands, and motor disabling functionality so that
they can be easily imported and used in other scripts.

Before using, ensure that the control table addresses, device name, baud rate, and motor IDs match your setup.
"""

from dynamixel_sdk import *  # Import Dynamixel SDK
import time

# Control table addresses (verify these with your motor's documentation)
ADDR_TORQUE_ENABLE = 64  # Address for torque enable
ADDR_GOAL_CURRENT = 102  # Address for goal current (for current control mode)

# Protocol version for Dynamixel communication (commonly 2.0)
PROTOCOL_VERSION = 2.0

# Motor IDs (set these to your motor IDs) # should be set to 1 and 2 in the dynamixel app.
DXL0_ID = 1
DXL1_ID = 2

# Communication settings (example for macOS; change as needed)
DEVICENAME = '/dev/tty.usbserial-FT79212K'  # Replace with your device name
BAUDRATE = 1000000  # 1M bps

# Torque control flags
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


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


def setup_motors():
    """
    Initialize the communication port and packet handler for the Dynamixel motors,
    and enable the torque for both motors.

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

    # Enable torque on both motors.
    for dxl_id in [DXL1_ID, DXL0_ID]:
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


def run_current_control_test(cycles=5, current_value=200, delay=0.50):
    """
    Run a current control test that alternates motor current between a positive and a negative value.

    This function initializes the motors, runs the specified number of test cycles, and then disables the motors.

    Parameters:
        cycles (int): Number of forward/backward cycles to execute.
        current_value (int): The magnitude of current to command (both positive and negative).
        delay (float): Time delay (in seconds) between switching directions.
    """
    portHandler, packetHandler = setup_motors()
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


# Allow the module to be run as a standalone script for testing.
if __name__ == "__main__":
    run_current_control_test()