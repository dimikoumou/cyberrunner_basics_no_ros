# goal is to understand the camera calibration process. how to load it from the calibration file and how to use it to undistort the image.
# then we have to implement the equivalent of the world2cam and cam2world functions in the ocam_model.py file.

# ## Camera Calibration
# load the calibration_data.npz file and print the keys.
import numpy as np
import cv2 


calibration_data = np.load("calibration_data.npz")
print(calibration_data)


# this is how we saved the data;         np.savez('calibration_data.npz', mtx=mtx, dist=dist)
mtx = calibration_data['mtx']
dist = calibration_data['dist']

print(mtx.shape)
print(dist.shape)


# ## Undistort the image
# load the image and undistort it using the camera calibration data.
# the image is distorted, we can see the distortion in the image.
img = cv2.imread("board1.png")

# undistort the image
undistorted_img = cv2.undistort(img, mtx, dist)
import numpy as np
import cv2


class OpenCVCameraModel:
    """
    Camera model using OpenCV's calibration approach. This replaces the OcamModel
    class which was ported from the Omnidirectional Camera Calibration Toolbox for Matlab.
    This class provides functionality for 3D-2D projections and vice versa, as well as
    image undistortion.
    """

    def __init__(self, calibration_file=None, camera_matrix=None, dist_coeffs=None):
        """
        Initialize the camera model either from a calibration file or directly from
        camera matrix and distortion coefficients.

        Parameters
        ----------
        calibration_file : string, optional
            Path to the .npz file containing the calibration results from OpenCV.
        camera_matrix : ndarray, optional
            3x3 camera intrinsic matrix.
        dist_coeffs : ndarray, optional
            Distortion coefficients (k1, k2, p1, p2, k3, ...).
        """
        if calibration_file is not None:
            # Load calibration data from file
            calibration_data = np.load(calibration_file)
            self.camera_matrix = calibration_data['mtx']
            self.dist_coeffs = calibration_data['dist']
        elif camera_matrix is not None and dist_coeffs is not None:
            # Use provided camera matrix and distortion coefficients
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
        else:
            raise ValueError("Either calibration_file or both camera_matrix and dist_coeffs must be provided")

        # Extract camera parameters
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
        
        # Initialize other properties
        self.height = None
        self.width = None
        self.map1 = None
        self.map2 = None

    def set_image_size(self, width, height):
        """
        Set the image dimensions for the camera.

        Parameters
        ----------
        width : int
            Image width in pixels.
        height : int
            Image height in pixels.
        """
        self.width = int(width)
        self.height = int(height)

    def scale(self, factor):
        """
        Scale down parameters by a given factor.

        Parameters
        ----------
        factor : float
            Factor to scale by. A factor of e.g. 2 halves the camera's assumed resolution.
        """
        # Scale camera matrix
        scaled_matrix = self.camera_matrix.copy()
        scaled_matrix[0, 0] /= factor  # fx
        scaled_matrix[1, 1] /= factor  # fy
        scaled_matrix[0, 2] /= factor  # cx
        scaled_matrix[1, 2] /= factor  # cy
        self.camera_matrix = scaled_matrix
        
        # Update individual parameters
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
        
        # Scale image dimensions if they're set
        if self.height is not None and self.width is not None:
            self.height = int(self.height / factor)
            self.width = int(self.width / factor)
        
        # Reset maps since they're now invalid
        self.map1 = None
        self.map2 = None

    def world2cam(self, world_points, rvec=np.zeros(3), tvec=np.zeros(3)):
        """
        Convert 3D world coordinates into 2D pixel coordinates.

        Parameters
        ----------
        world_points : array_like, shape (N, 3) or (3,)
            Each row is a 3D point in (x, y, z) format.
        rvec : array_like, shape (3,), optional
            Rotation vector. Default is no rotation.
        tvec : array_like, shape (3,), optional
            Translation vector. Default is no translation.

        Returns
        -------
        img_points : numpy.ndarray, shape (N, 2) or (2,)
            Each row is a 2D point in (u, v) format.
        """
        world_points = np.asarray(world_points, dtype=np.float32)
        is1d = world_points.ndim == 1
        if is1d:
            world_points = np.reshape(world_points, (1, 3))
        
        # Use OpenCV to project 3D points to 2D
        img_points, _ = cv2.projectPoints(world_points, rvec, tvec, 
                                          self.camera_matrix, self.dist_coeffs)
        img_points = img_points.reshape(-1, 2)
        
        if is1d:
            img_points = img_points.reshape(2)
            
        return img_points

    def cam2world(self, img_points, z=1.0):
        """
        Converts 2D pixel coordinates into 3D rays emanating from the camera center.
        
        Parameters
        ----------
        img_points : array_like, shape (N, 2) or (2,)
            Each row is a 2D point in (u, v) format.
        z : float, optional
            The z coordinate for the 3D points. Default is 1.0.
            
        Returns
        -------
        world_points : numpy.ndarray, shape (N, 3) or (3,)
            Each row is a 3D ray direction in (x, y, z) format.
        """
        img_points = np.asarray(img_points, dtype=np.float32)
        is1d = img_points.ndim == 1
        if is1d:
            img_points = np.reshape(img_points, (1, 2))
        
        # Undistort points
        undistorted_points = cv2.undistortPoints(
            img_points.reshape(-1, 1, 2), 
            self.camera_matrix, 
            self.dist_coeffs
        )
        undistorted_points = undistorted_points.reshape(-1, 2)
        
        # Convert undistorted points to rays
        world_points = np.ones((undistorted_points.shape[0], 3), dtype=np.float32)
        world_points[:, 0] = undistorted_points[:, 0]
        world_points[:, 1] = undistorted_points[:, 1]
        world_points[:, 2] = 1.0
        
        # Normalize rays to unit vectors
        norms = np.linalg.norm(world_points, axis=1, keepdims=True)
        world_points = world_points / norms * z
        
        if is1d:
            world_points = world_points.reshape(3)
            
        return world_points

    def set_maps(self, map1, map2):
        """
        Set undistortion maps.

        Parameters
        ----------
        map1 : ndarray
            First undistortion map.
        map2 : ndarray
            Second undistortion map.
        """
        self.map1 = map1
        self.map2 = map2

    def create_undistortion_maps(self, new_camera_matrix=None):
        """
        Create undistortion maps based on the current camera parameters.
        
        Parameters
        ----------
        new_camera_matrix : ndarray, optional
            Optional new camera matrix to use. If not provided, the original camera matrix is used.
            
        Returns
        -------
        map1 : ndarray
            First undistortion map.
        map2 : ndarray
            Second undistortion map.
        """
        if self.width is None or self.height is None:
            raise ValueError("Image dimensions are not set. Call set_image_size() first.")
            
        if new_camera_matrix is None:
            new_camera_matrix = self.camera_matrix
            
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, new_camera_matrix,
            (self.width, self.height), cv2.CV_16SC2
        )
        
        return self.map1, self.map2

    def undistort(self, img, new_camera_matrix=None):
        """
        Undistort an image using the camera model's intrinsic parameters.

        Parameters
        ----------
        img : array_like, shape (height, width, channels)
            The image to be undistorted in matrix form.
        new_camera_matrix : ndarray, optional
            Optional new camera matrix to use. If not provided, the original camera matrix is used.
            
        Returns
        -------
        img_undist : numpy.ndarray, shape (height, width, channels)
            The undistorted image.
        """
        if img is None:
            return None
            
        img = np.asarray(img)
        
        # Set image dimensions if not already set
        if self.height is None or self.width is None:
            self.height, self.width = img.shape[:2]
        
        # Use existing maps if available, otherwise create new ones
        if self.map1 is None or self.map2 is None:
            self.create_undistortion_maps(new_camera_matrix)
            
        # Undistort using the maps
        img_undist = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
        
        return img_undist

    def to_pinhole(self, img, pinhole_mdl, r, t, r_ph, t_ph, plane_z_ph=-18.5, recompute=False, save_maps=False):
        """
        Transform an image to a pinhole camera model.

        Parameters
        ----------
        img : array_like
            The input image.
        pinhole_mdl : OpenCVCameraModel
            The target pinhole camera model.
        r : object
            Rotation object for the current camera.
        t : ndarray
            Translation vector for the current camera.
        r_ph : object
            Rotation object for the pinhole camera.
        t_ph : ndarray
            Translation vector for the pinhole camera.
        plane_z_ph : float, optional
            Z-coordinate of the plane in the pinhole camera. Default is -18.5.
        recompute : bool, optional
            Whether to recompute the transformation maps. Default is False.
        save_maps : bool, optional
            Whether to save the transformation maps to files. Default is False.

        Returns
        -------
        img_undist : numpy.ndarray
            The transformed image.
        """
        if not recompute and self.map1 is not None and self.map2 is not None:
            img_undist = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
            return img_undist

        # Generate a grid of pixel coordinates in the pinhole model
        pixel_coord_ph = np.mgrid[0:pinhole_mdl.height, 0:pinhole_mdl.width].reshape(2, -1).transpose()
        
        # Convert pinhole pixel coordinates to 3D rays
        grid_points_ph = pinhole_mdl.cam2world(pixel_coord_ph)
        
        # Scale rays to intersect with the plane at z=plane_z_ph
        grid_points_ph /= grid_points_ph[:, 2, None]
        grid_points_ph *= plane_z_ph
        
        # Transform points from pinhole frame to world frame
        if hasattr(r_ph, 'apply'):
            # If r_ph is a rotation object with an apply method (e.g., from custom library)
            grid_points_fem = r_ph.apply(grid_points_ph - t_ph, inverse=True)
            grid_points_ocam = r.apply(grid_points_fem) + t
        else:
            # Using OpenCV's approach
            r_ph_mat, _ = cv2.Rodrigues(r_ph)
            r_mat, _ = cv2.Rodrigues(r)
            
            # Convert points from pinhole to world coordinates
            grid_points_fem = (np.linalg.inv(r_ph_mat) @ (grid_points_ph - t_ph[:, None]).T).T
            
            # Convert points from world to our camera coordinates
            grid_points_ocam = (r_mat @ grid_points_fem.T).T + t
        
        # Project 3D points to our camera's pixel coordinates
        pixel_coord_ocam = self.world2cam(grid_points_ocam)
        
        # Reshape to create maps
        pixel_coord_ocam = pixel_coord_ocam.reshape(pinhole_mdl.height, pinhole_mdl.width, 2).astype(np.float32)
        
        # Convert to OpenCV's map format
        self.map1, self.map2 = cv2.convertMaps(
            pixel_coord_ocam[..., 1], pixel_coord_ocam[..., 0], cv2.CV_16SC2
        )

        # Apply the transformation
        img_undist = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

        if save_maps:
            np.savetxt("map_1_0.txt", self.map1[:, :, 0], fmt="%d", delimiter="\n")
            np.savetxt("map_1_1.txt", self.map1[:, :, 1], fmt="%d", delimiter="\n")
            np.savetxt("map_2.txt", self.map2, fmt="%d", delimiter="\n")

        return img_undist


# Example usage:
if __name__ == "__main__":
    # Load calibration data
    calibration_file = 'calibration_data.npz'
    camera_model = OpenCVCameraModel(calibration_file=calibration_file)
    
    # Set image dimensions
    camera_model.set_image_size(width=1280, height=720)
    
    # Undistort an image
    import cv2
    img = cv2.imread('board1.png')
    if img is not None:
        undistorted_img = camera_model.undistort(img)
        
        ## test cam2world and world2cam
        # Define 3D points
        points_3d = np.array([
            [0, 0, 1],
            [0, 0, 0.5],
            [0, -0.1, .5], # positive y makes the poiint disappear. maayve its rendered behind the camera?
            [-0.1, -0.1, .5],
            [-0.3, -0.1,  0.5],
        ])
        
        # Project 3D points to 2D
        points_2d = camera_model.world2cam(points_3d)
        print("3D points: [original]")
        print(points_3d)
        # print("\n\n 2D points:")
        # print(points_2d)
        
        # test how cam2world works
        
        # Convert 2D points back to 3D
        points_3d_reconstructed = camera_model.cam2world(points_2d)
        print("\n\n 3D points reconstructed: [make sure this is the same as the original]" ) 
        # print the first 3 digits        
        print(np.round(points_3d_reconstructed, 3))  # Simple approach
        # Display the undistorted image
        
    
        
        
        # mark the 2d points on the image. mark them with a number
        for i, point in enumerate(points_2d):
            cv2.circle(undistorted_img, tuple(point.astype(int)), 5, (0, 255, 0), -1)
            cv2.putText(undistorted_img, str(i), tuple(point.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            
        cv2.imshow('undistorted_image', undistorted_img)
        cv2.waitKey(0)
        
        # start with 2d points
        points2d = np.array([
            [0, 0],
            [0, 100],
            [100, 100],
            [100, 0],
        ])
        print("2D points[original]")
        print(points2d) 
        
        # convert to 3d
        points3d = camera_model.cam2world(points2d)
        print("3D points from 2D points")
        print(points3d)
        
        # convert back to 2d
        points2d_reconstructed = camera_model.world2cam(points3d)
        print("2D points from 3D points")
        print(points2d_reconstructed)
        
        
        
        cv2.imwrite('undistorted_image.jpg', undistorted_img)
        print("Image undistorted and saved successfully.")
    else:
        print("Failed to load image.")