# CyberRunner Robotics Basics

This repository contains tools for computer vision-based state estimation and motor control for a ball balancing platform project.

## Prerequisites

- Python 3.6 or newer
- Dependencies listed in `requirements.txt`:
  - numpy
  - opencv-python
  - apriltag
  - matplotlib
  - scipy
  - dynamixel_sdk

## Installation

1. Clone this repository to your local machine
2. Install the required dependencies:

-> pip install -r requirements.txt

## Step-by-Step Guide

### 1. Camera Calibration

The first step is to calibrate your camera:

-> python camera_calibration_realtime.py


This script will:
- Capture multiple images of a checkerboard pattern from different angles
- Calculate the camera matrix and distortion coefficients
- Save the calibration data to `calibration_data.npz`

You will need a checkerboard pattern with 28×17 inner corners. Follow the on-screen instructions to complete the calibration process.

### 2. Capture Frame

Once the camera is calibrated, capture an image of the ball balancing board:

-> python capture_frame.py


This script will:
- Open your camera
- Capture a frame
- Save it as `board.png`

### 3. Marker Calibration

Calibrate the corner markers on the ball balancing board for the current camera resolution:

-> python marker_calibration.py

This script will:
- Capture a live frame from your camera at full resolution (1920x1080)
- Guide you through clicking on the 8 corner points (4 outer and 4 inner corners)
- Save the coordinates to `markers.csv` with proper resolution handling
- Include key press debouncing for better user experience

Alternatively, you can use the simplified interface:

-> python board_detection.py

This will automatically launch the marker calibration tool.

Follow the on-screen instructions:
- Click on each corner in the specified order
- Press 'r' to reset if you make a mistake
- Press 's' to save once all 8 corners are selected
- Press 'q' to quit

### 4. Running State Estimation

You can run state estimation in two ways:

#### Option A: Using test_estimation.py

For a quick test of the estimation system:

-> python test_estimation.py


This script will:
- Load the captured `board.png` image
- Initialize the state estimation pipeline
- Display visual feedback with the detected board and ball
- Print estimation results (ball position, plate angles, state estimates)

#### Option B: Building Your Own Estimation Pipeline

To integrate state estimation in your own code:

```python
import cv2
from state_est.board_estimation import EstimationPipeline

# Initialize the pipeline
pipeline = EstimationPipeline(
    fps=30,                    # Camera frames per second
    estimator="FiniteDiff",    # Estimation method
    print_measurements=True,   # Print detailed measurements
    show_image=True            # Show detection visualization
)

# Capture a frame or load from file
frame = cv2.imread('board.png')  # or capture from camera

# Run estimation
x_hat, P, inputs, xb, yb = pipeline.estimate(frame)

# Access state information
# - x_hat: State estimate [x, y, vx, vy]
# - P: Covariance matrix
# - inputs: Plate angles [alpha, beta] in radians
# - xb, yb: Ball position in maze coordinates
```


### 5. Motor Control Integration
To control Dynamixel motors based on the estimated state:

```python

from motors_control import setup_motors, set_motor_current, disable_motors

# Setup motor communication
portHandler, packetHandler = setup_motors()

# Set motor currents proportional to desired tilt angles
set_motor_current(packetHandler, portHandler, 0, int(500 * inputs[0]))
set_motor_current(packetHandler, portHandler, 1, int(500 * inputs[1]))

# When finished, disable motors
disable_motors(packetHandler, portHandler)
```



Troubleshooting
Camera not found: Check your camera connection and adjust the camera_index parameter in scripts.
Calibration issues: Ensure your checkerboard pattern is fully visible and try to cover different angles.
Board detection errors: Make sure lighting is adequate and the board corners are clearly visible.
Motor communication errors: Verify port name in motors_control.py matches your system and check connections.
Project Structure
capture_frame.py: Camera frame capture utilities
camera_calibration_realtime.py: Camera calibration tools
board_detection.py: User-guided marker detection
motors_control.py: Dynamixel motor control functions
state_est/: State estimation modules
board_estimation.py: Main estimation pipeline
detection.py: Marker and ball detection
estimator.py: State estimation algorithms
measurements.py: Processing raw measurements
plate_pose.py: 3D pose estimation of the board
