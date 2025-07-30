import cv2
import numpy as np
import sys
import time

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to capture corner coordinates"""
    if event == cv2.EVENT_LBUTTONDOWN:
        corners_list, img = param
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        corners_list.append([x, y])
        print(f"Corner {len(corners_list)}: ({x}, {y})")
        cv2.imshow('Calibration', img)

def calibrate_markers():
    """Interactive marker calibration tool"""
    # Capture frame
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to capture frame")
        return
    
    print(f"Frame shape: {frame.shape}")
    h, w = frame.shape[:2]
    
    corners = []
    img_display = frame.copy()
    
    cv2.namedWindow('Calibration', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Calibration', mouse_callback, (corners, img_display))
    
    print("\n=== MARKER CALIBRATION ===")
    print("Click on the corners in this order:")
    print("1-4: OUTER corners (large blue dots): lower_left, lower_right, upper_right, upper_left")
    print("5-8: INNER corners (small blue dots inside maze): lower_left, lower_right, upper_right, upper_left")
    print("Press 'r' to reset, 's' to save, 'q' to quit")
    
    cv2.imshow('Calibration', img_display)
    
    last_key_time = 0
    
    while True:
        key = cv2.waitKey(30) & 0xFF
        current_time = time.time()
        
        # Debounce key presses
        if key != 255 and current_time - last_key_time < 0.3:
            continue
            
        if key == ord('q'):
            print("Quitting...")
            last_key_time = current_time
            break
        elif key == ord('r'):
            print("Resetting corners...")
            corners.clear()
            img_display = frame.copy()
            cv2.imshow('Calibration', img_display)
            last_key_time = current_time
        elif key == ord('s'):
            last_key_time = current_time
            if len(corners) == 8:
                print("Saving markers...")
                labels = [
                    "outer,lower left", "outer,lower right", "outer,upper right", "outer,upper left",
                    "inner,lower left", "inner,lower right", "inner,upper right", "inner,upper left"
                ]
                
                print("\n=== SAVING MARKERS ===")
                with open('markers.csv', 'w') as f:
                    f.write("corner_type,position,x,y\n")
                    for i, (corner, label) in enumerate(zip(corners, labels)):
                        corner_type, position = label.split(',')
                        f.write(f"{corner_type},{position},{corner[0]},{corner[1]}\n")
                        print(f"  {i+1}: {label} = ({corner[0]}, {corner[1]})")
                
                print("Saved to markers.csv")
                
                # Clear display and show success message
                img_display.fill(0)
                cv2.putText(img_display, "Success!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow('Calibration', img_display)
                cv2.waitKey(2000) # Show for 2 seconds
                break
            else:
                print(f"Need {8 - len(corners)} more corners before saving!")
        
        # Show current status
        if len(corners) > 0:
            img_temp = img_display.copy()
            status = f"Corners: {len(corners)}/8 - Press 's' to save, 'r' to reset, 'q' to quit"
            cv2.putText(img_temp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Calibration', img_temp)
        else:
            img_temp = img_display.copy()
            status = "Click 8 corners - Press 'r' to reset, 'q' to quit"
            cv2.putText(img_temp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Calibration', img_temp)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate_markers()

