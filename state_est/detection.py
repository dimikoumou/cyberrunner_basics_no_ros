import numpy as np
import cv2

# It's good practice to keep these helper modules imported
try:
    from .gaussian_robust import detect_gaussian, detect_gaussian_robust
    from .masking import mask_hsv
except ImportError:
    from gaussian_robust import detect_gaussian, detect_gaussian_robust
    from masking import mask_hsv

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
c_name = ["blue", "green", "red", "yellow"]

class Detector:
    """
    Detector class for identifying corners and a ball in an image using HSV masking
    and Gaussian-based detection.
    (Documentation remains the same)
    """

    DEFAULT_HSV_CORNERS = ((100, 140), (100, 255), (100, 255))  # More specific blue for markers
    DEFAULT_Q_CORNERS = 5
    DEFAULT_TH_CORNERS = 0.002
    DEFAULT_HSV_BALL = ((73, 101), (111, 255), (42, 255)) # More specific blue for ball
    DEFAULT_Q_BALL = 6
    DEFAULT_TH_BALL = 10 ** (-4)
    DEFAULT_SIZE_CROP_CORNERS = 95 / 3
    DEFAULT_SIZE_CROP_BALL = 150 / 3
    DEFAULT_INIT_BALL_POS = np.array([47, 330])

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
        corner_subimage_half_size=17,
        min_area=50,
       show_subimages=False,
        max_area=1000,
        min_circularity=0.2,
        
    ):
        self.hsv_params_corners = hsv_params_corners
        self.q_corners = q_corners
        self.th_corners = th_corners
        self.hsv_params_ball = hsv_params_ball
        self.q_ball = q_ball
        self.th_ball = th_ball
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.ball_pos = ball_init_pos
        self.corners = None
        self.show_subimages = show_subimages
        self.corners_missing = True
        self.fixed_corners = None
        self.is_ball_found = False
        self.corner_subimage_half_size = corner_subimage_half_size

        ## CORRECTION: Remove the incorrect scaling factor `/ 3.0`.
        ## The code now uses the original marker coordinates, as the input
        ## frame is also at full resolution.
        corners = np.repeat(
            np.expand_dims(np.asarray(markers)[:, ::-1], axis=1), 2, axis=1
        )

        corners[:, 0] -= self.corner_subimage_half_size
        corners[:, 1] += self.corner_subimage_half_size

        self.default_coords_subimages_corners = corners.astype(int)



        # corners = np.repeat(
        #     np.expand_dims(np.asarray(markers)[:, ::-1] / 3.0, axis=1), 2, axis=1
        # )
        
        # corners[:, 0] -= self.corner_subimage_half_size
        # corners[:, 1] += self.corner_subimage_half_size
        
        # self.default_coords_subimages_corners = corners.astype(int)
        
        self.default_coords_subimage_ball = (
            self.default_coords_subimages_corners[3, 0],
            self.default_coords_subimages_corners[1, 1],
        )

    def process_frame(self, frame):
        corners = self.detect_corners(frame)
        ball = self.detect_ball(frame)
        return corners, ball

    def get_cropped(self, im: np.ndarray, pos: np.ndarray, h_p: float, w_p: float):
        """
        Return cropped image and its top-left and
                down-right corners coordinates in the given image.
        """
        h, w = im.shape[:2]
      
        
        # Assume input `pos` is in (row, column) format based on comments
        center_row, center_col = pos[0], pos[1]
        
        # Calculate half-width and half-height
        half_h = h_p / 2
        half_w = w_p / 2
        
        # Calculate row and column boundaries, ensuring they are within the image frame
        row_start = min(h - 1, max(0, int(center_row - half_h)))
        row_end   = min(h - 1, max(0, int(center_row + half_h)))
        col_start = min(w - 1, max(0, int(center_col - half_w)))
        col_end   = min(w - 1, max(0, int(center_col + half_w)))

        # Slice the image using the correct [rows, columns] convention
        im_cropped = im[row_start:row_end, col_start:col_end]
        
        # Define the upper-left and lower-right corner coordinates
        ul = np.array([row_start, col_start])
        dr = np.array([row_end, col_end])
        
        return im_cropped, ul, dr

    def predictive_cropping_corners(self, im: np.ndarray):
        h, w = im.shape[:2]
        h_p, w_p = (
            Detector.DEFAULT_SIZE_CROP_CORNERS,
            Detector.DEFAULT_SIZE_CROP_CORNERS,
        )
        subimgs = []
        subcoords = []
        for i in range(4):
            # self.corners[i, :] is already (row, col)
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
            corner, found = self.detect_corner(
                sub_im, i, subcoords_corners_imgs[i][0]
            )
            if found:
                corners[i, :] = corner
            elif self.corners is not None:
                corners[i, :] = self.corners[i, :]
            else:
                corners[i, :] = np.array([np.nan, np.nan])
            missing = missing or not found
        self.corners_missing = missing
        #print("found corners: \n", corners)
        self.corners = corners
        return corners

    def detect_corner(self, sub_im: np.ndarray, i: int, coords_ul_sub_im: np.ndarray):
        """
        Detects a single corner in a cropped subimage using robust detection.
        """
        return self.detect_corner_with_filtering(sub_im, i, coords_ul_sub_im)
    
    def detect_corner_with_filtering(self, sub_im: np.ndarray, i: int, coords_ul_sub_im: np.ndarray):
        """
        Detects a single corner in a subimage with advanced contour filtering.
        If a best contour exists, compute its centroid with image moments (M = cv2.moments). 
        Return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])). Otherwise, return None.
        """
        # HSV masking and contour detection
        sub_masked, mask = mask_hsv(sub_im, self.hsv_params_corners)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, False

        # Filter contours by area and circularity
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter**2)
            if circularity < self.min_circularity:
                continue

            valid_contours.append(c)

        if not valid_contours:
            return None, False

        # Select the best contour (e.g., largest area)
        best_contour = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return None, False

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Return center coordinates and success status
        return (coords_ul_sub_im + np.array([cy, cx])).astype("float32"), True

    def detect_corner_robust(self, sub_im: np.ndarray, i: int, coords_ul_sub_im: np.ndarray):
        """
        Robustly detects a single corner by trying multiple HSV ranges and using
        advanced contour filtering.
        """
        # Multiple HSV parameter sets to try
        hsv_ranges = [
            self.hsv_params_corners,  # Primary range
            ((80, 130), (200, 255), (120, 255)),  # Slightly wider range
            ((70, 140), (180, 255), (100, 255)),  # Even wider range
            ((60, 150), (160, 255), (80, 255)),   # Very wide range
        ]
        
        for attempt, hsv_params in enumerate(hsv_ranges):
            try:
                # Apply HSV masking
                sub_masked, mask = mask_hsv(sub_im, hsv_params)
                
                # Add Gaussian blur to smooth noise before contour extraction
                mask = cv2.GaussianBlur(mask, (5, 5), 0)

                # Use robust detection
                c_local, found = detect_gaussian_robust(
                    mask, i, self.q_corners, self.th_corners, show_sub=self.show_subimages
                )
                
                if found:
                    c = (coords_ul_sub_im + c_local).astype("float32")
                    if attempt > 0:
                        print(f"Corner {i} found with HSV range {attempt + 1}")
                    return c, True
                    
            except Exception as e:
                print(f"Error in corner detection attempt {attempt + 1}: {e}")
                continue
        
        # If all attempts failed, return center position
        print(f"Corner {i} detection failed with all HSV ranges")
        center_offset = np.array([sub_im.shape[0] // 2, sub_im.shape[1] // 2])
        c = (coords_ul_sub_im + center_offset).astype("float32")
        return c, False

    def detect_ball(
        self,
        im: np.ndarray,
        show_rectangle: bool = False,
        mask_corner=False,
        mask_initial=True,
    ):
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
            # Search the whole image if ball is not found
            cropped_ball_im = im
            coords_ul_cropped_img = np.array([0, 0])

        sub_masked, mask = mask_hsv(cropped_ball_im, self.hsv_params_ball)

        # Add Gaussian blur to smooth noise before contour extraction
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        c_local, self.is_ball_found = detect_gaussian_robust(
            mask, 4, self.q_ball, self.th_ball, show_sub=self.show_subimages
        )

        if not self.is_ball_found:
            return np.array([np.nan, np.nan])

        c = (coords_ul_cropped_img + c_local).astype("float32")  # (x,y)

        # Geometric safety check: ensure the detected ball is not a corner
        if self.corners is not None and not np.isnan(self.corners).any():
            min_dist_to_corner = np.min(np.linalg.norm(self.corners - c, axis=1))

            # If the ball is too close to any corner, it's likely a false positive
            if min_dist_to_corner < 30:  # 30 pixels threshold
                self.is_ball_found = False
                return np.array([np.nan, np.nan])

        self.ball_pos = c
        return c

    def draw_corners(self, frame: np.ndarray):
        for i in range(self.corners.shape[0]):
            if not np.isnan(self.corners[i, 0]):
                cv2.drawMarker(
                    frame,
                    (round(self.corners[i, 1]), round(self.corners[i, 0])),
                    colors[i],
                cv2.MARKER_TILTED_CROSS,
                20,
                2,
                )  # (u,v)
        return

    def draw_ball(self, frame: np.ndarray):
        if self.ball_pos is not None and not np.isnan(self.ball_pos).any():
            cv2.drawMarker(
                frame,
                tuple((np.round(self.ball_pos).astype(int))[::-1]),
                (0, 0, 255),
                cv2.MARKER_TILTED_CROSS,
                20,
                2,
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
    def __init__(self, markers, show_subimages: bool = True):
        # hsv_corners = (
        #     (43, 140),  # (minHue, maxHue)
        #     (125, 255),  # (minSat, maxSat)
        #     (40, 255),  # (minVal, maxVal)
        # )
        super().__init__(
            markers,
           # 
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

        # Add Gaussian blur to smooth noise before contour extraction
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        c_local, blob_found = detect_gaussian(
            mask, i, self.q_corners, self.th_corners, show_sub=self.show_subimages
        )
        c = (coords_ul_sub_im + c_local).astype("float32")
        
        if not blob_found:
            print("Unable to find corner {}".format(i + 1))
            exit()

        return c, blob_found