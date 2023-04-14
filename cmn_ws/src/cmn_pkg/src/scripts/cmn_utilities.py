 #!/usr/bin/env python3

"""
Functions that will be useful in more than one node in this project.
"""

import rospkg, yaml, cv2
import numpy as np
from math import sin, cos, remainder, tau, ceil
from random import random, randrange

from scripts.rotated_rectangle_crop_opencv.rotated_rect_crop import crop_rotated_rectangle


def clamp(val:float, min_val:float, max_val:float):
    """
    Clamp the value val in the range [min_val, max_val].
    @return float, the clamped value.
    """
    return min(max(min_val, val), max_val)


class ObservationGenerator:
    """
    Class to handle map/vehicle coordinate transforms.
    Also uses the rotated_rectangle_crop_opencv functions to crop out an observation region corresponding to a particular vehicle pose.
    Used both for ground-truth observation generation as well as particle likelihood evalutation.
    """
    initialized = False
    map = None # 2D numpy array of the global map
    map_resolution = None # float, meters/pixel for the map.

    def __init__(self):
        """
        Create instance and set important global params.
        """
        self.read_params()

    def read_params(self):
        """
        Read configuration params from the yaml.
        """
        # Determine filepath.
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('cmn_pkg')
        # Open the yaml and get the relevant params.
        with open(pkg_path+'/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            # Map params. NOTE this will eventually be unknown and thus non-constant as it is estimated.
            self.map_resolution = config["map"]["resolution"]
            self.map_downscale_ratio = config["map"]["downscale_ratio"]
            self.map_resolution /= self.map_downscale_ratio
            # Observation region size.
            self.obs_resolution = config["observation"]["resolution"] / self.map_downscale_ratio
            self.obs_height_px = config["observation"]["height"]
            self.obs_width_px = config["observation"]["width"]
            self.obs_height_px_on_map = int(self.obs_height_px * self.obs_resolution / self.map_resolution)
            self.obs_width_px_on_map = int(self.obs_width_px * self.obs_resolution / self.map_resolution)
            # Vehicle position relative to observation region.
            self.veh_px_horz_from_center_on_obs = (config["observation"]["veh_horz_pos_ratio"] - 0.5) * self.obs_width_px
            self.veh_px_vert_from_bottom_on_obs = config["observation"]["veh_vert_pos_ratio"] * self.obs_width_px
            self.veh_px_horz_from_center_on_map = self.veh_px_horz_from_center_on_obs * self.obs_resolution / self.map_resolution
            self.veh_px_vert_from_bottom_on_map = self.veh_px_vert_from_bottom_on_obs * self.obs_resolution / self.map_resolution

    def set_map(self, map):
        """
        Set the map that will be used to crop out observation regions.
        Determine valid vehicle bounds, and add padding around the map border.
        """
        # Set the map as it is temporarily so we can use it to determine vehicle bounds.
        self.map = map
        # Set the map bounds in meters. This prevents true vehicle pose from leaving the map.
        self.map_x_min_meters, self.map_y_min_meters = self.transform_map_px_to_m(map.shape[1]-1, 0)
        self.map_x_max_meters, self.map_y_max_meters = self.transform_map_px_to_m(0, map.shape[0]-1)

        """
        When generating an observation, it is possible the desired region will be partially outside the bounds of the map.
        To prevent potential errors, create a padded version of the map with enough extra rows/cols to ensure this won't happen.
        Expand all dimensions by the diagonal of the observation area to cover all possible situations.
        All extra space will be assumed to be occluded cells (value = 0.0).
        """
        max_obs_dim = ceil(np.sqrt(self.obs_height_px_on_map**2 + self.obs_width_px_on_map**2))
        self.map = cv2.copyMakeBorder(map, max_obs_dim, max_obs_dim, max_obs_dim, max_obs_dim, cv2.BORDER_CONSTANT, None, 0.0)
        self.initialized = True

    def transform_map_px_to_m(self, row:int, col:int):
        """
        Given coordinates of a cell on the ground-truth map, compute the equivalent position in meters.
        Origin (0,0) in meters corresponds to center of map.
        Origin in pixels is top left, and coords are strictly nonnegative.
        @return tuple of floats (x, y)
        """
        # Clamp inputs within legal range of values.
        row = int(clamp(row, 0, self.map.shape[0]-1))
        col = int(clamp(col, 0, self.map.shape[1]-1))
        # Get pixel difference from map center.
        row_offset = row - self.map.shape[0] // 2
        col_offset = col - self.map.shape[1] // 2
        # Convert from pixels to meters.
        x = self.map_resolution * col_offset
        y = self.map_resolution * -row_offset
        return x, y

    def transform_map_m_to_px(self, x:float, y:float):
        """
        Given coordinates of a vehicle pose in meters, compute the equivalent cell in pixels.
        Origin (0,0) in meters corresponds to center of map.
        Origin in pixels is top left, and coords are strictly nonnegative.
        @return tuple of ints (row, col)
        """
        # Convert from meters to pixels.
        col_offset = x / self.map_resolution
        row_offset = -y / self.map_resolution
        # Shift origin from center to corner.
        row = row_offset + self.map.shape[0] // 2
        col = col_offset + self.map.shape[1] // 2
        # Clamp within legal range of values.
        row = int(clamp(row, 0, self.map.shape[0]-1))
        col = int(clamp(col, 0, self.map.shape[1]-1))
        return row, col

    def extract_observation_region(self, vehicle_pose):
        """
        Crop out the map region for the observation corresponding to the given vehicle pose.
        @param vehicle_pose, numpy array of [x,y,yaw] in meters and radians.
        """
        # Compute vehicle pose in pixels.
        veh_row, veh_col = self.transform_map_m_to_px(vehicle_pose[0], vehicle_pose[1])

        # Project ahead of vehicle pose to determine center of observation region.
        center_col = veh_col + (self.obs_height_px_on_map / 2 - self.veh_px_vert_from_bottom_on_map) * cos(vehicle_pose[2])
        center_row = veh_row - (self.obs_height_px_on_map / 2 - self.veh_px_vert_from_bottom_on_map) * sin(vehicle_pose[2])
        center = (center_col, center_row)
        # Create the rotated rectangle.
        angle = -np.rad2deg(vehicle_pose[2])
        rect = (center, (self.obs_height_px_on_map, self.obs_width_px_on_map), angle)

        # Crop out the rotated rectangle and reorient it.
        obs_img = crop_rotated_rectangle(image = self.map, rect = rect)

        # Resize observation to desired resolution.
        obs_img = cv2.resize(obs_img, (self.obs_height_px, self.obs_width_px))

        # Return both the image and the rect points for the viz to use for plotting.
        return obs_img, rect
    
    def generate_random_valid_veh_pose(self):
        """
        Generate a vehicle pose at random.
        Ensure this position is within map bounds, and is located in free space.
        @return 3x1 numpy array of the created pose.
        """
        # Choose any yaw at random, with no validity condition.
        yaw = remainder(random() * tau, tau)
        # Generate a particle at random on the map, and ensure it's in a valid cell.
        while True:
            # Choose a random vehicle cell in px so we can first check if it's occluded.
            r = randrange(0, self.map.shape[0])
            c = randrange(0, self.map.shape[1])
            if self.map[r, c] == 1: # this cell is free.
                x, y = self.transform_map_px_to_m(r, c)
                return np.array([x,y,yaw])
            
    def veh_pose_m_in_collision(self, vehicle_pose) -> bool:
        """
        Given a vehicle pose, determine if this is in collision on the map.
        @param vehicle_pose, numpy array of [x,y,yaw] in meters and radians.
        @return true if the vehicle pose is in collision on the occupancy grid map.
        """
        r, c = self.transform_map_m_to_px(vehicle_pose[0], vehicle_pose[1])
        return self.map[r, c] != 1 # return false if this cell is free, and true otherwise.
