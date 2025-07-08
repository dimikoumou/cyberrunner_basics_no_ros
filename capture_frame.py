import cv2
import time
import threading
from collections import deque


# ---------------------------------------------------------------------
# (A) CameraCaptureThread
# ---------------------------------------------------------------------
class CameraCaptureThread(threading.Thread):
    def __init__(self, camera_index=0, queue_size=2):
        super().__init__()
        self.camera_index = camera_index
        self.cap = None
        self.stopped = False
        self.frames = deque(maxlen=queue_size)
        self.daemon = True  # Thread dies with the main program

    def run(self):
        # Try using AVFOUNDATION; if issues persist, consider using the default backend:
        # self.cap = cv2.VideoCapture(self.camera_index)
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)

        # Set MJPEG fourcc and camera properties
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        if not self.cap.isOpened():
            print("Camera could not be opened!")
            self.stopped = True
            return

        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frames.appendleft(frame)
            else:
                # Optional: add a small delay if a frame wasn't grabbed successfully.
                time.sleep(0.01)

        self.cap.release()

    def read(self):
        if self.frames:
            return self.frames[0]
        return None

    def stop(self):
        self.stopped = True
        self.join()


# ---------------------------------------------------------------------
# (B) Main Function: Capture and Save an Image
# ---------------------------------------------------------------------
def main():
    cam_thread = CameraCaptureThread(camera_index=0)
    cam_thread.start()

    # Wait up to 5 seconds for a frame to be captured
    timeout = 5
    start_time = time.time()
    frame = None
    while time.time() - start_time < timeout:
        frame = cam_thread.read()
        if frame is not None:
            break
        time.sleep(0.1)

    if frame is not None:
        cv2.imwrite("board.png", frame)
        print("Image saved as board.png")
    else:
        print(f"No frame captured after waiting {timeout} seconds.")

    cam_thread.stop()


if __name__ == "__main__":
    main()