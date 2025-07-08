# Author: Aswin Ramachandran
# https://markhedleyjones.com/projects/calibration-checkerboard-collection

import cv2
import numpy as np
import time
import pygame
import os
from datetime import datetime
import json

def save_the_cal_in_csv(calib_engine):
    """saves the calibration in a csv file. 
    the csv file will have the following structure:
    #polynomial coefficients for the DIRECT mapping function (ocam_model.ss in MATLAB). These are used by cam2world

    [list the coefficients here]

    #polynomial coefficients for the inverse mapping function (ocam_model.invpol in MATLAB). These are used by world2cam

    [list the coefficients here]

    #center: "row" and "column", starting from 0 (C convention)

    [xc, yc]

    #affine parameters "c", "d", "e"

    [c, d, e]

    #image size: "height" and "width"

    [height, width]


    Args:
        calib_engine: calibration engine object from pyOCamCalib
    """
    
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    outputs = {"date": dt_string,
                   "camera_name": calib_engine.cam_name,
                   "valid": calib_engine.valid_pattern,
                   "taylor_coefficient": calib_engine.taylor_coefficient.tolist(),
                   "distortion_center": calib_engine.distortion_center,
                   "stretch_matrix": calib_engine.stretch_matrix.tolist(),
                   "inverse_poly": calib_engine.inverse_poly.tolist(),
                   "extrinsics_t": [e.tolist() for e in calib_engine.extrinsics_t],
                   "img_path": calib_engine.images_path,
                   "rms_overall": calib_engine.rms_overall,
                   "rms_mean_list": calib_engine.rms_mean_list,
                   "rms_std_list": calib_engine.rms_std_list
                   }
    # Save the this in a jscon file.
    with open(f'calibration_results/output.json', 'w') as f:
            json.dump(outputs, f, indent=4)
    
    # Save the calibration data in a CSV file
    with open('calibration_results/calibration_data.csv', 'w') as f:
        f.write("#polynomial coefficients for the DIRECT mapping function (ocam_model.ss in MATLAB). These are used by cam2world\n")
        f.write(" ".join(map(str, calib_engine.taylor_coefficient)) + "\n\n")

        # Write the polynomial coefficients for the inverse mapping function
        f.write("#polynomial coefficients for the inverse mapping function (ocam_model.invpol in MATLAB). These are used by world2cam\n")
        f.write(" ".join(map(str, calib_engine.inverse_poly)) + "\n\n")

        # Write the distortion center
        f.write("#center: row and column, starting from 0 (C convention)\n")
        f.write(" ".join(map(str, calib_engine.distortion_center)) + "\n\n")

        # Write the stretch matrix
        f.write("#affine parameters c, d, e\n")
        f.write(" ".join(map(str, calib_engine.stretch_matrix)) + "\n\n") # NOTE: here I need to make sure i am using them in correct order(whats e whats d)

        # Write the image size
        f.write("#image size: height and width\n")
        f.write(f"{img_height},{img_width}\n")
        
    print("Calibration data saved in 'calibration_results/calibration_data.csv'")
    


def play_gong_for_one_second(filename='sounds/bleep.wav'):
    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the gong sound
    pygame.mixer.music.load(filename=filename)

    # Play the gong sound
    pygame.mixer.music.play()

    # Wait for 1 second
    time.sleep(1.0)

    # Stop the sound after 1 second
    pygame.mixer.music.stop()

    # Wait for another 2 second
    time.sleep(1.0)

def calibrate_camera():
    # Define the dimensions of checkerboard (number of inner corners matters )
    CHECKERBOARD = (28, 17)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    # Multiply by square size (25mm = 0.025m)
    objp *= 0.024

    # Capture images from camera
    cap = cv2.VideoCapture(0)  # incdex 0 should point to the razer camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    global img_width, img_height
    img_width = 1920
    img_height = 1080

    # Set camera properties for See3CAM_24CUG
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)



    images_captured = 0
    max_images = 20  # Number of images to capture for calibration
    
    # check if there are 20 images in the calib_images folder
    # check if the folder calib_images exists
    if not os.path.exists('calib_images'):
        os.makedirs('calib_images')
    if len([name for name in os.listdir('calib_images') if os.path.isfile(os.path.join('calib_images', name))]) >= max_images:
        images_captured = max_images
        print("There are enough calib images available")
        
        # have to loop through them and detect cornesrs, prepare the objpoints and imgpoints
        for i in range(max_images):
            frame = cv2.imread(f'calib_images/image{i}.jpg')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                    cv2.CALIB_CB_FAST_CHECK +
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
    
    else:
        print(f"Please show the {CHECKERBOARD} checkerboard pattern from different angles.")
        print(f"Capturing {max_images} images for calibration...")

        while images_captured < max_images:
            ret, frame = cap.read()
            # save the image in calib_images folder
            cv2.imwrite(f'calib_images/image{images_captured}.jpg', frame)

            if not ret:
                print("Failed to grab frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                    cv2.CALIB_CB_FAST_CHECK +
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
                images_captured += 1
                print(f"Image {images_captured}/{max_images} captured")
                # play a sound of a gong before capturing the next image
                play_gong_for_one_second()

            cv2.imshow('Calibration', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    if images_captured == max_images:
        # frame is the 19th image from the calib_folder
        # frame = cv2.imread(f'calib_images/image{images_captured-1}.jpg')
        #frame  = cap.read()[1]
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        shape_of_last_image = cv2.imread(f'calib_images/image{images_captured-1}.jpg').shape
        print("shape of last image: ", shape_of_last_image)
        # print("gray shape: ", gray.shape)
        print("objpoints shape: ", objpoints[0].shape)
        print("imgpoints shape: ", imgpoints[0].shape)
        print("almost last gray ", gray.shape[::-1])
        print("\n\n\n")
        

        
            
        print("\n\n\n We have captured enough images. Now we're calling the calibration function from pyOCamCalib")
        # Calibrate camera with OpenCV first
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray[::-1].shape, None, None)

        # Save the OpenCV calibration results as csv
        # np.savetxt('calibration_data_cvCAL.csv', np.concatenate((mtx, dist, rvecs, tvecs), axis=0), delimiter=',')
        print("\nOpenCV calibration data saved to the file 'calibration_data_cvCAL.csv'")
        
        # save the taken images in the folder called calib_images
        for i in range(images_captured):
            img = cv2.imread(f'calib_images/image{i}.jpg')
            cv2.imwrite(f'calib_images/image{i}.jpg', img)
            
        print("Calibration images saved in the folder 'calib_images'")
        
        # Now run the pyOCamCalib calibration
        try:

            from pyocamcalib.script.calibration_script import CalibrationEngine
            
            print("Starting fisheye calibration with pyOCamCalib...")
            
            # Create the calibration engine with our captured images
            calib_dir = 'calib_images'  # Directory where we saved the calibration images
            calib_engine = CalibrationEngine(
                working_dir=calib_dir,
                chessboard_size=CHECKERBOARD,  # Using the same checkerboard dimensions
                camera_name="razer_camera",
                square_size=0.024  # Same square size as defined earlier (in meters)
            )
            
            # Detect corners in the images
            calib_engine.detect_corners(check=False)
            print("Corners detected in images.")
            # Estimate the fisheye camera parameters
            calib_engine.estimate_fisheye_parameters()
            print("Fisheye parameters estimated.")
            
            # Find the inverse polynomial
            calib_engine.find_poly_inv()
            print("Inverse polynomial found.")
            # Save the calibration in pyOCamCalib's native format (JSON).
            # NOTE: cant do this because this function has been designed to work only when ran by calibration_scropt.py from within the lib. so we save it ouselves. 
            #calib_engine.save_calibration()
            #print("Calibration saved in pyOCamCalib format.")
            print("Saving calibration data in CSV format...")
            save_the_cal_in_csv(calib_engine)
            
            # Also save important parameters in CSV format
            
            
            return mtx, dist, calib_engine
        
        except ImportError:
            print("pyOCamCalib library not found. Only OpenCV calibration results are available.")
            return mtx, dist
        except Exception as e:
            print(f"Error during pyOCamCalib calibration: {str(e)}")
            return mtx, dist
    else:
        print("Not enough images captured for calibration.")
        return None, None


if __name__ == "__main__":
    calibrate_camera()