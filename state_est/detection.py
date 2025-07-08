import numpy as np
import cv2

colors = [(255, 0, 255), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
c_name = ["blue", "green", "red", "yellow"]

from gaussian_robust import detect_gaussian
from masking import mask_hsv


# from profileFIle import profile
class Detector:
    """
    Detector class for identifying corners and a ball in an image using HSV masking
    and Gaussian-based detection.

    Attributes:
    -----------
    hsv_params_corners : tuple
        HSV thresholds for corner detection.
    q_corners : float
        Gaussian detection quantile for corners.
    th_corners : float
        Threshold for Gaussian detection of corners.
    hsv_params_ball : tuple
        HSV thresholds for ball detection.
    q_ball : float
        Gaussian detection quantile for the ball.
    th_ball : float
        Threshold for Gaussian detection of the ball.
    ball_pos : np.ndarray or None
        Detected position of the ball.
    corners : np.ndarray or None
        Detected positions of the corners.
    show_subimages : bool
        Whether to display subimages for debugging.
    fixed_corners : np.ndarray or None
        Fixed positions of the corners (if available).
    is_ball_found : bool
        Flag indicating if the ball has been detected.
    corner_subimage_half_size : int
        Size of the corner subimage cropping window.
    """

    DEFAULT_HSV_CORNERS = (
        (43, 140),  # (minHue, maxHue)
        (125, 255),  # (minSat, maxSat)
        (9, 255),
    )  # (minVal, maxVal)
    DEFAULT_Q_CORNERS = 5  # gaussian detection param -> q-th quentile
    DEFAULT_TH_CORNERS = 0.002  # gaussian detection threshold

    DEFAULT_HSV_BALL = (
        (89, 121),  # (minHue, maxHue)
        (172, 255),  # (minSat, maxSat)
        (21, 255),
    )  # (minVal, maxVal)
    DEFAULT_Q_BALL = 6  # gaussian detection param -> q-th quentile
    DEFAULT_TH_BALL = 10 ** (-4)  # gaussian detection threshold

    DEFAULT_SIZE_CROP_CORNERS = 95 / 3 # actually for tracking and plotting!!!
    DEFAULT_SIZE_CROP_BALL = 150 / 3

    DEFAULT_INIT_BALL_POS = np.array([47, 330])  # np.array([55,485]) # NOTE: needs changing for every camera. prollly should be a part of the manual marker detection process.

    def __init__(
        self,
        markers,
        hsv_params_corners: list = DEFAULT_HSV_CORNERS,
        q_corners: float = DEFAULT_Q_CORNERS,
        th_corners: float = DEFAULT_TH_CORNERS,
        hsv_params_ball: list = DEFAULT_HSV_BALL,
        q_ball: float = DEFAULT_Q_BALL,
        th_ball: float = DEFAULT_TH_BALL,
        ball_init_pos: np.ndarray = DEFAULT_INIT_BALL_POS,
        corner_subimage_half_size=17, #for the default, wwhen the corner was lost!
        show_subimages=False,
    ):

        self.hsv_params_corners = hsv_params_corners
        self.q_corners = q_corners
        self.th_corners = th_corners
        self.hsv_params_ball = hsv_params_ball
        self.q_ball = q_ball
        self.th_ball = th_ball

        self.ball_pos = None
        self.corners = None
        self.show_subimages = show_subimages

        self.corners_missing = True

        self.fixed_corners = None
        self.is_ball_found = False

        self.corner_subimage_half_size = corner_subimage_half_size
        corners = np.repeat(
            np.expand_dims(np.asarray(markers)[:, ::-1] / 3.0, axis=1), 2, axis=1
        )
        
        corners[:, 0] -= self.corner_subimage_half_size
        corners[:, 1] += self.corner_subimage_half_size
        
        # NOTE: trying the scaled down version
        #corners[:, 0] = corners[:, 0] / 3
        #corners[:, 1] = corners[:, 1] / 3 # NOTE: when i scale them down they at least compile...
        
        self.default_coords_subimages_corners = corners.astype(int) # NOTE: outer corners, original resolution?  # TODO: ASK THOMAS! 
        
        self.default_coords_subimage_ball = (
            self.default_coords_subimages_corners[3, 0],
            self.default_coords_subimages_corners[1, 1],
        )

    # @profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
    def process_frame(self, frame):
        """
        Process frame to get raw image coordinates of the four corners and the ball.

        Args :
            frame: np.ndarray, dim: (400, 640)
        Returns :
            corners: np.ndarray, dim: (4,2)
                     the raw image coordinates of the four corners dots in (x,y) = (line, column) convention.
            ball: np.ndarray, dim: (2,)
                     the raw image coordinates of the ball in (x,y) = (line, column) convention.
        """
        # NOTE: these print statements are useful to debug corner and ball detection
        # print("delect corners")
        corners = self.detect_corners(frame)
        # print("corners: ", corners)
        # print("detect ball")
        ball = self.detect_ball(frame, show_rectangle=True)
        # print("ball: ", ball)
        return corners, ball  # both in (x,y) conventions

    def get_cropped(self, im: np.ndarray, pos: np.ndarray, h_p: float, w_p: float):
        """
        Return cropped image and its top-left and
                down-right corners coordinates in the given image.
        Args :
            im: np.ndarray
                image
            pos: np.ndarray
                 position of the center of the subimage
            h_p: float
                 height of the subimage
            w_p: float
                 width of the subimage
        Returns :
            im_cropped: np.ndarray
            ul: np.ndarray, dim: (2,)
                top-left corner coordinates in the given image.
            dr: np.ndarray, dim: (2,)
                down-right corner coordinates in the given image.
        """
        h, w = im.shape[:2]
        # NOTE: useful for debugging the corner detection:
        # print(" \n\ngetting the cropped image: ")
        # print("im shape: ", im.shape)
        # print("pos: ", pos)
        # print("h_p: ", h_p)
        # print("w_p: ", w_p)

        ul_x = min(h - 1, max(0, int(pos[0] - h_p / 2)))
        ul_y = min(w - 1, max(0, int(pos[1] - w_p / 2)))
        dr_x = min(h - 1, max(0, int(pos[0] + h_p / 2)))
        dr_y = min(w - 1, max(0, int(pos[1] + w_p / 2)))
        im_cropped = im[ul_x:dr_x, ul_y:dr_y]
        ul = np.array([ul_x, ul_y])
        dr = np.array([dr_x, dr_y])
        
        # print("ul: ", ul)
        # print("dr: ", dr)
        # print("im_cropped shape: ", im_cropped.shape)
        return im_cropped, ul, dr

    def predictive_cropping_corners(self, im: np.ndarray):
        """
        Performs predictive cropping for detecting corners based on previous detections.

        Parameters:
        -----------
        im : np.ndarray
            Input image.

        Returns:
        --------
        tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]
            - `subimgs`: List of cropped images for each corner.
            - `subcoords`: List of tuples containing upper-left and lower-right coordinates.
        """
        print("using predictive cropping corners")
        h, w = im.shape[:2]
        h_p, w_p = (
            Detector.DEFAULT_SIZE_CROP_CORNERS,
            Detector.DEFAULT_SIZE_CROP_CORNERS,
        )
        subimgs = []
        subcoords = []
        for i in range(4):
            subimg, ul, dr = self.get_cropped(im, self.corners[i, :], h_p, w_p)
            subimgs.append(subimg)
            subcoords.append((ul, dr))
        return subimgs, subcoords

    def predictive_cropping_ball(self, im: np.ndarray, draw: bool = False):
        """
        Performs predictive cropping for detecting the ball.

        Parameters:
        -----------
        im : np.ndarray
            Input image.
        draw : bool, optional
            Whether to draw the cropping region on the image.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            - `im_cropped`: Cropped image of the ball.
            - `ul`: Upper-left corner of the cropped region.
        """
        
        h_p, w_p = Detector.DEFAULT_SIZE_CROP_BALL, Detector.DEFAULT_SIZE_CROP_BALL
        im_cropped, ul, dr = self.get_cropped(im, self.ball_pos, h_p, w_p)
        if draw:
            im = cv2.rectangle(
                im, tuple(ul[::-1]), tuple(dr[::-1]), (0, 255, 0), 1
            )  # need to do im =.. ? or just remove the im = ??
        return im_cropped, ul

    def is_ball_in_corner(self):  # ball pos in in (x,y)
        """
        Determines if the detected ball is in one of the four corners.

        Returns:
        --------
        int or None
            Corner index (0-3) if the ball is in a corner, otherwise None.
        """
        if self.ball_pos[0] < 100 and self.ball_pos[1] < 200:
            return 3
        if self.ball_pos[0] < 100 and self.ball_pos[1] > 450:
            return 2
        if self.ball_pos[0] > 300 and self.ball_pos[1] > 450:
            return 1
        if self.ball_pos[0] > 300 and self.ball_pos[1] < 200:
            return 0
        return None

    def get_default_subimages_corners(self, im: np.ndarray, show: bool = False):
        """
        Extracts default subimages of corners from the frame.

        Parameters:
        -----------
        im : np.ndarray
            Input image.
        show : bool, optional
            Whether to display the extracted subimages.

        Returns:
        --------
        tuple[list[np.ndarray], np.ndarray]
            - `subimages`: List of cropped corner subimages.
            - `self.default_coords_subimages_corners`: Array of corner coordinates.
        """
        h, w = im.shape[:2]
        print("when getting delault corners: im shape: ", im.shape)
        if show:
            for c in self.default_coords_subimages_corners:
                cv2.rectangle(im, c[0][::-1], c[1][::-1], (0, 0, 255), 1)
        # TODO use get_cropped
        subimages = [
            im[cs[0][0] : cs[1][0], cs[0][1] : cs[1][1]]
            for cs in self.default_coords_subimages_corners
        ]
        # print("subimages: ", subimages)
        print("self.default_coords_subimages_corners: \n\n", self.default_coords_subimages_corners) # RETURNS 24 x 24 square around the corners. NOTE: these are the outer corners of the board. we use the unscaled res???
        print("\n\nsubimages shape: ", subimages[0].shape)
        return subimages, self.default_coords_subimages_corners

    def detect_corners(self, frame):
        """
        Detects the four corners in the given frame.

        Parameters:
        -----------
        frame : np.ndarray
            Input image.

        Returns:
        --------
        np.ndarray
            Detected corner coordinates of shape (4,2).
        """
        corners = np.zeros((4, 2), dtype="float32")

        if self.corners is None or self.corners_missing:
            (
                cropped_corners_imgs,
                subcoords_corners_imgs,
            ) = self.get_default_subimages_corners(frame)
        else:
            (
                cropped_corners_imgs,
                subcoords_corners_imgs,
            ) = self.predictive_cropping_corners(frame)
 
        # useful for debugging the corner detection             
        # print("subcoords_corners_imgs: \n", subcoords_corners_imgs)
        # print("subimgs shape: ", cropped_corners_imgs[0].shape)
        missing = False
        for i, sub_im in enumerate(cropped_corners_imgs):
            corners[i, :], found = self.detect_corner(
                sub_im, i, subcoords_corners_imgs[i][0]
            )
            missing = missing or not found
        self.corners_missing = missing
        #print("found corners: \n", corners)
        self.corners = corners
        return corners

    def detect_corner(self, sub_im: np.ndarray, i: int, coords_ul_sub_im: np.ndarray):
        """
        Detects a single corner in a cropped subimage.

        Parameters:
        -----------
        sub_im : np.ndarray
            Cropped subimage containing a corner.
        i : int
            Index of the corner.
        coords_ul_sub_im : np.ndarray
            Upper-left coordinates of the subimage in the original frame.

        Returns:
        --------
        tuple[np.ndarray, bool]
            - `c`: Detected corner position.
            - `found`: Boolean indicating if the corner was found.
        """
        # print("detecing corner number: ", i, "using Detector (not fixed pts) class") 
        sub_masked, mask = mask_hsv(sub_im, self.hsv_params_corners)
        c_local, found = detect_gaussian(
            mask, i, self.q_corners, self.th_corners, show_sub=self.show_subimages
        )
        c = (coords_ul_sub_im + c_local).astype("float32")
        return c, found

    def detect_ball(
        self,
        im: np.ndarray,
        show_rectangle: bool = False,
        mask_corner=False,
        mask_initial=True,
    ):
        """
        Detects the ball in the given frame.

        Parameters:
        -----------
        im : np.ndarray
            Input image.
        show_rectangle : bool, optional
            Whether to draw the bounding box around the detected ball.
        mask_corner : bool, optional
            Whether to mask detected corners.
        mask_initial : bool, optional
            Whether to mask the initial ball position.

        Returns:
        --------
        np.ndarray
            Detected ball coordinates or NaN if not found.
        """

        if self.is_ball_found:
            if mask_corner:
                corner_ball = self.is_ball_in_corner()
                if (
                    corner_ball is not None
                ):  # masking the corner that is in vicinity of the ball
                    cv2.circle(
                        im,
                        tuple(self.corners[corner_ball, :].astype(int)[::-1]),
                        10,
                        (0, 0, 255),
                        -1,
                    )
            cropped_ball_im, coords_ul_cropped_img = self.predictive_cropping_ball(
                im, draw=show_rectangle
            )
        else:
            # if self.ball_pos is not None:
            print("\n\n\nBALL NOT FOUND\n\n\n")
            if mask_initial:
                for i in range(4):
                    cv2.circle(
                        im,
                        tuple(np.round(self.corners[i, :]).astype(int)[::-1]),
                        10,
                        (0, 0, 255),
                        -1,
                    )
                    cv2.circle(
                        im,
                        tuple(np.round(self.fixed_corners[i, :]).astype(int)[::-1]),
                        10,
                        (0, 0, 255),
                        -1,
                    )

            ul, dr = self.default_coords_subimage_ball
            cropped_ball_im, coords_ul_cropped_img = (
                im[ul[0] : dr[0], ul[1] : dr[1], :],
                ul,
            )
        sub_masked, mask = mask_hsv(cropped_ball_im, self.hsv_params_ball)
        # print("ball")
        c_local, self.is_ball_found = detect_gaussian(
            mask, 4, self.q_ball, self.th_ball, show_sub=self.show_subimages
        )
        if not self.is_ball_found:
            return np.array([np.nan, np.nan])

        # print(c_local)
        c = (coords_ul_cropped_img + c_local).astype("float32")  # (x,y)
        self.ball_pos = c
        print("ball pos [end of detect_ball]: ", self.ball_pos)
        return c

    def draw_corners(self, frame: np.ndarray):
        for i in range(self.corners.shape[0]):
            cv2.drawMarker(
                frame,
                (round(self.corners[i, 1]), round(self.corners[i, 0])),
                colors[i],
                cv2.MARKER_TILTED_CROSS,
                5,
                1,
            )  # (u,v)
        return

    def draw_ball(self, frame: np.ndarray):
        cv2.drawMarker(
            frame,
            tuple((np.round(self.ball_pos).astype(int))[::-1]),
            (0, 0, 255),
            cv2.MARKER_TILTED_CROSS,
            5,
            1,
        )  # (u,v)

    def reset(self, ball_pos_init: np.ndarray = DEFAULT_INIT_BALL_POS):
        self.corners = None
        self.ball_pos = ball_pos_init


# TODO remove
class DetectorFixedPts(Detector):
    """
    Detector class with fixed point detection for corners.

    This subclass overrides `detect_corner` and uses a predefined HSV range for detection.

    Parameters:
    -----------
    markers : list
        Initial corner marker positions.
    show_subimages : bool, optional
        Whether to display subimages for debugging.

    Attributes:
    -----------
    hsv_corners : tuple
        Custom HSV parameters for fixed-point detection.
    """
    def __init__(self, markers, show_subimages: bool = False):
        hsv_corners = (
            (43, 140),  # (minHue, maxHue)
            (125, 255),  # (minSat, maxSat)
            (40, 255),  # (minVal, maxVal)
        )
        super().__init__(
            markers,
            hsv_params_corners=hsv_corners,
            corner_subimage_half_size=12,
            show_subimages=show_subimages,
        )

    def detect_corner(self, sub_im: np.ndarray, i: int, coords_ul_sub_im: np.ndarray):
        """
        Detects a corner in a cropped subimage with fixed point constraints.

        Parameters:
        -----------
        sub_im : np.ndarray
            Cropped subimage containing a corner.
        i : int
            Index of the corner.
        coords_ul_sub_im : np.ndarray
            Upper-left coordinates of the subimage in the original frame.

        Returns:
        --------
        tuple[np.ndarray, bool]
            - `c`: Detected corner position.
            - `blob_found`: Boolean indicating if the corner was found.
        """
        # NOTE: this printf are useful for debugging the corner detection 
        # (in particular when a corner was lost this class will be used to catch it again)
        # print("detecing corner number: ", i, "using DetectorFixedPts class")
        # print("sub_im shape: ", sub_im.shape)
        # # print("hsv_params: ", self.hsv_params_corners)
        sub_masked, mask = mask_hsv(sub_im, self.hsv_params_corners)
        c_local, blob_found = detect_gaussian(
            mask, i, self.q_corners, self.th_corners, show_sub=self.show_subimages
        )
        c = (coords_ul_sub_im + c_local).astype("float32")
        
        if not blob_found:
            print("Unable to find corner {}".format(i + 1))
            exit()

        return c, blob_found