# goal of this script is to load one captured image and detect the boundary of the red cicle in the undistorted and scaled down scale 
# if verbose is true we display the taken image and mark the detected circle
# then we return the (x,y) coordinates of the circle boundary

import cv2
import numpy as np

def get_circle_boundary(image_path, config_txt, verbose=False):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Could not load image.")
    # Optionally load calibration parameters from config_txt (placeholder)
    
    # Convert to HSV and create masks for red color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("No red circle found.")
    # Choose the largest contour
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    
    if radius < 1:
        raise Exception("Detected circle is too small.")
    
    # If verbose, display the detected circle on the image
    if verbose:
        output = img.copy()
        center = (int(x), int(y))
        radius_int = int(radius)
        cv2.circle(output, center, radius_int, (0, 255, 0), 2)
        cv2.circle(output, center, 2, (0, 0, 255), 3)
        cv2.imshow("Detected Circle", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Return the circle parameters : (x,y) of the center and the radius
    return (int(x), int(y)), radius_int

# Example usage
if __name__ == "__main__":
    image_path = "board.png"
    config_txt = "calib_razer_data.txt"
    verbose = True
    try:
        circle_parameters = get_circle_boundary(image_path, config_txt, verbose)
        print("Circle parameters (center and radius):", circle_parameters)
    except Exception as e:
        print("Error:", e)
