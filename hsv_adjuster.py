import cv2
import numpy as np

def nothing(x):
    pass

# Create a window
cv2.namedWindow('Trackbars')

# Create trackbars for color change
cv2.createTrackbar('H_min', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('H_max', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('S_min', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('S_max', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V_min', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('V_max', 'Trackbars', 255, 255, nothing)

# Start camera feed
cap = cv2.VideoCapture(0)

print("\nAdjust the sliders to isolate the corner color.")
print("Press 'q' to quit and print the final HSV values.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of trackbars
    h_min = cv2.getTrackbarPos('H_min', 'Trackbars')
    h_max = cv2.getTrackbarPos('H_max', 'Trackbars')
    s_min = cv2.getTrackbarPos('S_min', 'Trackbars')
    s_max = cv2.getTrackbarPos('S_max', 'Trackbars')
    v_min = cv2.getTrackbarPos('V_min', 'Trackbars')
    v_max = cv2.getTrackbarPos('V_max', 'Trackbars')

    # Set lower and upper HSV limits
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    # Create a mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Optional: show result
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show the frames
    cv2.imshow('Original Feed', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print the final values before exiting
print("\n-t------------------------------------------")
print(f"Final HSV values:")
print(f"hsv_params = (({h_min}, {h_max}), ({s_min}, {s_max}), ({v_min}, {v_max}))")
print("-------------------------------------------")

cap.release()
cv2.destroyAllWindows()