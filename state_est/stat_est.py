# adapted from: brio_state_estimation/brio_state_estimation_nonros.py

import cv2 as cv
import numpy as np
import time

from estimation_pipeline import EstimationPipeline
from  divers import init_capture
import matplotlib.pyplot as plt


def mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN: 
        ball_pos = np.array([x,y])[::-1]
        estimation_pipeline.measurements.detector.reset(ball_pos)

def init_windows(): 
    winUndist = cv.namedWindow("ori", cv.WINDOW_NORMAL)
    cv.moveWindow("ori", 50,50)
    cv.resizeWindow("ori", 640, 400)

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

    processed_frames = 0
    while True: 
        
        #
        if not pause:
            ret, frame = cap.read() 
            timenew = time.time() 
            tframe = timenew
            
        processed_frames += 1
        frame = cv.resize(frame, (int(width/3), int(height/3))) # why are we resizing here?
        # TODO: ask Thomas why were doing it. 
        # transpose the image
        # frame = cv.transpose(frame) # TODO: ASK THOSMAS IF THIS IS NNEEDED, board_estimation.py suggest the image should be passd  dim: (400,640)
        # frame = cv.imread("testimg_jan.png") # TODO: remove this line, it is just for testing

        print("\n\n\n using frame of shape: ", frame.shape)
        
        # display the markers on the frame
        # for i, marker in enumerate(markers):
        #     cv.circle(frame, (int(marker[0]), int(marker[1])), 5, (0, 255, 0), -1)
        #     cv.putText(frame, str(i), (int(marker[0]), int(marker[1])), cv.FONT_HERSHEY_SIMPLEX, 1 , (255, 255, 255), 2)
        cv.imshow("board", frame)
        cv.waitKey(0)
            
        
        b,g,r = np.mean(np.mean(frame, axis = 0), axis = 0)
        if g > 100 and b < 40 and r < 40 :
            print("SKIP THIS FRAME")
            # cv2.waitKey(1)
            continue
        x_hat, P, inputs, xb, yb = estimation_pipeline.estimate(frame) # NOTE: we have to unpack more values here than what was in the repo
        alpha, beta = inputs 
        alpha = np.rad2deg(alpha)
        beta = np.rad2deg(beta)
        
        print("#"*5, "frame number: ", processed_frames, "#"*5)
        print("x_hat:", x_hat)
        print("estimated P:", P)
        print("estimated plate angles: ", alpha, beta)

    cap.release()
    cv.destroyAllWindows()