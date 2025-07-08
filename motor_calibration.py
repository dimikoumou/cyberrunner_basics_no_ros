#!/usr/bin/env python3
import json
import os
import dynamixel_sdk as dxl  # Uses Dynamixel SDK library

# Dynamixel motor IDs
DXL0_ID = 1
DXL1_ID = 2


# Communication settings (example for macOS; change as needed)
DEVICENAME = '/dev/tty.usbserial-FT79212K'  # Replace with your device name
BAUDRATE = 1000000  # 1M bps


PROTOCOL_VERSION = 2.0  # Default for most modern Dynamixel motors

# Control table addresses (depends on your motor model - these are for X-series)
ADDR_PRESENT_POSITION = 132  # Address of Present Position in the Control Table

# Initialize handlers for Dynamixel communication
portHandler = dxl.PortHandler(DEVICENAME)
packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

def init_dynamixel():
    """Initialize communication with Dynamixel motors"""
    # Open port
    try:
        if not portHandler.openPort():
            print("Failed to open the port")
            return False
            
        # Set port baudrate
        if not portHandler.setBaudRate(BAUDRATE):
            print("Failed to change the baudrate")
            return False
            
        print("Successfully connected to Dynamixel motors")
        return True
    except Exception as e:
        print(f"Error initializing Dynamixel motors: {e}")
        return False

def close_dynamixel():
    """Close communication with Dynamixel motors"""
    portHandler.closePort()
    print("Closed connection to Dynamixel motors")

def read_motor_positions():
    """
    Read the current position of the motors using Dynamixel SDK.
    
    Returns:
        tuple: (motor1_position, motor2_position)
    """
    dxl_error = 0
    
    # Read position of first motor
    motor1_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
        portHandler, DXL0_ID, ADDR_PRESENT_POSITION)
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print(f"Communication error when reading motor 1: {packetHandler.getTxRxResult(dxl_comm_result)}")
        return 0, 0
    elif dxl_error != 0:
        print(f"Error in motor 1: {packetHandler.getRxPacketError(dxl_error)}")
        return 0, 0
        
    # Read position of second motor
    motor2_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
        portHandler, DXL1_ID, ADDR_PRESENT_POSITION)
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print(f"Communication error when reading motor 2: {packetHandler.getTxRxResult(dxl_comm_result)}")
        return 0, 0
    elif dxl_error != 0:
        print(f"Error in motor 2: {packetHandler.getRxPacketError(dxl_error)}")
        return 0, 0
    
    return motor1_position, motor2_position

def main():
    print("=" * 50)
    print("MOTOR CALIBRATION UTILITY")
    print("=" * 50)
    
    # Initialize Dynamixel communication
    if not init_dynamixel():
        print("Failed to initialize Dynamixel motors. Exiting...")
        return
    
    try:
        # Minimum position calibration
        print("\nStep 1: Set to MINIMUM position")
        print("Please push the board maximally to the top-left corner")
        input("Press Enter when the board is in position... ")
        
        min_motor1, min_motor2 = read_motor_positions()
        print(f"Recorded minimum positions: Motor 1 = {min_motor1}, Motor 2 = {min_motor2}")
        
        # Maximum position calibration
        print("\nStep 2: Set to MAXIMUM position")
        print("Please push the board maximally to the bottom-left corner")
        input("Press Enter when the board is in position... ")
        
        max_motor1, max_motor2 = read_motor_positions()
        print(f"Recorded maximum positions: Motor 1 = {max_motor1}, Motor 2 = {max_motor2}")
        
        # Save calibration data
        calibration_data = {
            "MIN_MOTOR1": min_motor1,
            "MIN_MOTOR2": min_motor2,
            "MAX_MOTOR1": max_motor1,
            "MAX_MOTOR2": max_motor2
        }
        
        file_path = os.path.join(os.path.dirname(__file__), "motor_minmax_positions.json")
        with open(file_path, 'w') as f:
            json.dump(calibration_data, f, indent=4)
        
        print(f"\nCalibration complete! Data saved to: {file_path}")
        
    finally:
        # Always close the connection to the motors
        close_dynamixel()

if __name__ == "__main__":
    main()
