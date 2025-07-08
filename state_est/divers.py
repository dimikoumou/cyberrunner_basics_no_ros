import cv2 as cv
import matplotlib

import platform


def init_capture(device, idx_cam, video_path, init_video_frame_idx):
    """
    Initializes a video capture source, either from a camera or a video file.

    Parameters:
    -----------
    device : str
        The type of device, either "CAM" for a camera or "VIDEO" for a video file.
    idx_cam : int
        The index of the camera (only used if `device` is "CAM").
    video_path : str
        The path to the video file (only used if `device` is "VIDEO").
    init_video_frame_idx : int
        The initial frame index to start from in the video file (only used if `device` is "VIDEO").

    Returns:
    --------
    cap : cv.VideoCapture
        The initialized OpenCV video capture object.
    width : float
        The width of the video frames.
    height : float
        The height of the video frames.

    Notes:
    ------
    - Camera initialization settings vary depending on the operating system.
    - If using a camera, the function sets default resolution to 1920x1200 and FPS to 55.
    - If using a video file, it sets the initial frame position.
    """

    if device == "CAM":
        osname = platform.platform()
        if osname.startswith("Windows"):
            cap = cv.VideoCapture(idx_cam, cv.CAP_DSHOW)
        elif osname.startswith("Linux"):
            cap = cv.VideoCapture(idx_cam)
        else:
            cap = cv.VideoCapture(idx_cam, cv.CAP_AVFOUNDATION)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080) # TODO: figure out if this couldnt be loaded automatically.
        cap.set(cv.CAP_PROP_FPS, 55)
    elif device == "VIDEO":
        cap = cv.VideoCapture(video_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, init_video_frame_idx)

    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float `height`

    return cap, width, height


def init_win_subimages():
    """
    Initializes and positions OpenCV windows for displaying subimages.

    This function creates five named windows labeled as "sub_0" to "sub_4"
    and positions them at predefined screen coordinates.

    Returns:
    --------
    None

    Notes:
    ------
    - The predefined coordinates correspond to the upper-left corners of the plate's subimage windows.
    - Windows are repositioned relative to a fixed offset.
    """
    cs = [
        (450, 330),
        (650, 330),
        (650, 100),
        (450, 100),
        (550, 200),
    ]  # ul corners coordinates of plate corners subimages windows
    for i, c in enumerate(cs):
        cv.namedWindow("sub_" + str(i))
        cv.moveWindow("sub_" + str(i), c[0] - 200, c[1] + 500)


def move_figure(f, x, y):
    """
    Moves a Matplotlib figure window to a specified screen position.

    Parameters:
    -----------
    f : matplotlib.figure.Figure
        The Matplotlib figure to move.
    x : int
        The x-coordinate (in pixels) of the new window position.
    y : int
        The y-coordinate (in pixels) of the new window position.

    Returns:
    --------
    None

    Notes:
    ------
    - The function handles different backends (`TkAgg`, `WXAgg`, `MacOSX`, `QT`, `GTK`).
    - On `MacOSX`, positioning may not be supported directly and requires manual adjustment.
    - Prints a warning if positioning fails for an unsupported backend.
    """
    
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        f.canvas.manager.window.SetPosition((x, y))
    elif backend == "MacOSX":
        # For MacOSX backend, we need to use a different approach
        # Position will work after the figure is shown
        try:
            f.canvas.manager.window.setGeometry(x, y, f.get_figwidth()*f.dpi, f.get_figheight()*f.dpi)
        except AttributeError:
            # MacOSX backend may not support window positioning this way
            print("Warning: Could not position figure on MacOSX. Position it manually.")
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        try:
            f.canvas.manager.window.move(x, y)
        except AttributeError:
            print(f"Warning: Could not position figure with backend {backend}. Position it manually.")