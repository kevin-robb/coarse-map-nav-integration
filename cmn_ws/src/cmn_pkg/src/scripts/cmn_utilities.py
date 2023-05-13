 #!/usr/bin/env python3

"""
Functions that will be useful in more than one node in this project.
"""

import rospkg, yaml, cv2, rospy
import numpy as np
from math import sin, cos, remainder, tau, ceil
from random import random, randrange
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from scripts.rotated_rectangle_crop_opencv.rotated_rect_crop import crop_rotated_rectangle

#### GLOBAL VARIABLES ####
bridge = CvBridge()
#########################

def clamp(val:float, min_val:float, max_val:float):
    """
    Clamp the value val in the range [min_val, max_val].
    @return float, the clamped value.
    """
    return min(max(min_val, val), max_val)


class MapFrameManager:
    """
    Class to handle map/vehicle coordinate transforms.
    Also uses the rotated_rectangle_crop_opencv functions to crop out an observation region corresponding to a particular vehicle pose.
    Used both for ground-truth observation generation as well as particle likelihood evalutation.
    """
    initialized = False
    map = None # 2D numpy array of the global map
    map_resolution = None # float, meters/pixel for the map.
    # Occupancy map values.

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

class Simulator(MapFrameManager):
    """
    Class to support running the project in simulation, without the robot providing real data.
    """
    # Ground-truth vehicle pose (x,y,yaw) in meters and radians, in the global map frame (origin at center).
    veh_pose_true = np.array([0.0, 0.0, 0.0])

    def __init__(self):
        self.read_params()

    def read_params(self):
        super().read_params()
        # Read params only needed for the simulator.
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('cmn_pkg')
        # Open the yaml and get the relevant params.
        with open(pkg_path+'/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            self.dt = config["dt"]
            # Constraints.
            self.max_fwd_cmd = config["constraints"]["fwd"]
            self.max_ang_cmd = config["constraints"]["ang"]
            self.allow_motion_through_occupied_cells = config["simulator"]["allow_motion_through_occupied_cells"]

    def propagate_with_vel(self, lin:float, ang:float):
        """
        Given a commanded velocity twist, move the robot by v*dt.
        @param lin, Commanded linear velocity (m/s).
        @param ang, Commanded angular velocity (rad/s).
        """
        # Clamp commands to allowed values (redundant since clamping is done in MOT, but just to be safe).
        fwd_dist = self.dt * clamp(lin, 0, self.max_fwd_cmd) # dt * meters/sec forward
        dtheta = self.dt * clamp(ang, -self.max_ang_cmd, self.max_ang_cmd) # dt * radians/sec CCW
        # Use our other function to command this distance.
        self.propagate_with_dist(fwd_dist, dtheta)

    def propagate_with_dist(self, lin:float, ang:float):
        """
        Given a commanded motion, move the robot accordingly.
        @param lin, Commanded linear distance (m).
        @param ang, Commanded angular distance (rad).
        """
        # TODO Perturb with some noise.
        # Compute a proposed new vehicle pose, and check if it's allowed before moving officially.
        veh_pose_proposed = np.copy(self.veh_pose_true)
        veh_pose_proposed[0] += lin * cos(veh_pose_proposed[2])
        veh_pose_proposed[1] += lin * sin(veh_pose_proposed[2])
        # Clamp the vehicle pose to remain inside the map bounds.
        veh_pose_proposed[0] = clamp(veh_pose_proposed[0], self.map_x_min_meters, self.map_x_max_meters)
        veh_pose_proposed[1] = clamp(veh_pose_proposed[1], self.map_y_min_meters, self.map_y_max_meters)
        # Keep yaw normalized to (-pi, pi).
        veh_pose_proposed[2] = remainder(veh_pose_proposed[2] + ang, tau)
        # Determine if this vehicle pose is allowed.
        if not self.allow_motion_through_occupied_cells and self.veh_pose_m_in_collision(veh_pose_proposed):
            rospy.logwarn("SIM: Command would move vehicle to invalid pose. Only allowing angular motion.")
            self.veh_pose_true[2] = veh_pose_proposed[2]
        else:
            self.veh_pose_true = veh_pose_proposed
            rospy.loginfo("SIM: Got command. Veh pose is now " + str(self.veh_pose_true))

    def get_true_observation(self):
        """
        @return the ground-truth observation, using our ground-truth map and vehicle pose.
        """
        # Use more general utilities class to generate the observation, using the known true vehicle pose.
        return self.extract_observation_region(self.veh_pose_true)


class CoarseMapProcessor:
    """
    Class to handle reading the coarse map from file, and doing any pre-processing.
    """
    # Global variables for class
    raw_map = None # Original coarse map including color.
    occ_map = None # Thresholded & binarized coarse map to create an occupancy grid.

    def __init__(self):
        """
        Create instance and set important global params.
        """
        self.read_params()
        self.read_coarse_map_from_file()

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
            # Map params.
            self.show_map_images = config["map"]["show_images_during_pre_proc"]
            self.map_fpath = pkg_path + "/config/maps/" + config["map"]["fname"]
            self.obs_balloon_radius = config["map"]["obstacle_balloon_radius"]
            # NOTE the scale will eventually be unknown and thus non-constant as it is estimated at runtime.
            self.map_resolution = config["map"]["resolution"]
            self.map_downscale_ratio = config["map"]["downscale_ratio"]
            self.map_resolution /= self.map_downscale_ratio

    def read_coarse_map_from_file(self):
        """
        Read the coarse map image from the provided filepath.
        Save the map itself to use for visualizations.
        Process the image by converting it to an occupancy grid.
        """
        # Read map image and account for possible white = transparency that cv2 will think is black.
        # https://stackoverflow.com/questions/31656366/cv2-imread-and-cv2-imshow-return-all-zeros-and-black-image/62985765#62985765
        img = cv2.imread(self.map_fpath, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4: # we have an alpha channel.
            a1 = ~img[:,:,3] # extract and invert that alpha.
            img = cv2.add(cv2.merge([a1,a1,a1,a1]), img) # add up values (with clipping).
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # strip alpha channels.

        rospy.loginfo("CMP: Read raw coarse map image with shape {:}".format(img.shape))
        if self.show_map_images:
            cv2.imshow('initial map', img); cv2.waitKey(0); cv2.destroyAllWindows()

        # Downsize the image to the desired resolution.
        img = cv2.resize(img, (int(img.shape[0] * self.map_downscale_ratio), int(img.shape[1] * self.map_downscale_ratio)))
        rospy.loginfo("CMP: Resized coarse map to shape {:}".format(img.shape))
        if self.show_map_images:
            cv2.imshow('resized map', img); cv2.waitKey(0); cv2.destroyAllWindows()

        # Convert from BGR to RGB and save the color map for any viz.
        self.raw_map = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Turn this into a grayscale img and then to a binary map.
        occ_map_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)[1]
        # Normalize to range [0,1].
        occ_map_img = np.divide(occ_map_img, 255)
        rospy.loginfo("CMP: Thresholded/binarized map to shape {:}".format(img.shape))
        if self.show_map_images:
            cv2.imshow("Thresholded Map", occ_map_img); cv2.waitKey(0); cv2.destroyAllWindows()
        
        # Consider anything not completely white (1) as occluded (0).
        self.occ_map = np.floor(occ_map_img)

        # Expand occluded cells so path planning won't take us right next to obstacles.
        if self.obs_balloon_radius == 0:
            rospy.logwarn("CMP: For some reason everything breaks if we skip the ballooning step.")
            self.obs_balloon_radius = 1
        # Determine index pairs to select all neighbors when ballooning obstacles.
        nbrs = []
        for i in range(-self.obs_balloon_radius, self.obs_balloon_radius+1):
            for j in range(-self.obs_balloon_radius, self.obs_balloon_radius+1):
                nbrs.append((i, j))
        # Remove 0,0 which is just the parent cell.
        nbrs.remove((0,0))
        # Expand all occluded cells outwards.
        for i in range(len(self.occ_map)):
            for j in range(len(self.occ_map[0])):
                if occ_map_img[i][j] != 1: # occluded.
                    # Mark all neighbors as occluded.
                    for chg in nbrs:
                        self.occ_map[clamp(i+chg[0], 0, self.occ_map.shape[0]-1)][clamp(j+chg[1], 0, self.occ_map.shape[1]-1)] = 0
        self.occ_map = np.float32(np.array(self.occ_map))
        if self.show_map_images:
            cv2.imshow("Ballooned Occ Map", self.occ_map); cv2.waitKey(0); cv2.destroyAllWindows()

        if self.show_map_images:
            # Show value distribution in occ_map.
            freqs = [0, 0]
            for i in range(len(self.occ_map)):
                for j in range(len(self.occ_map[0])):
                    if self.occ_map[i][j] == 0:
                        freqs[0] += 1
                    else:
                        freqs[1] += 1
            rospy.loginfo("CMP: Occ map value frequencies: "+str(freqs[1])+" free, "+str(freqs[0])+" occluded.")

    def get_raw_map_msg(self):
        """
        @usage occ_map_pub.publish(map_proc.get_occ_map_msg())
        @return the original coarse map image (in color) as a ROS Image message.
        """
        try:
            rospy.loginfo("CMP: Returning original coarse map with shape {:}.".format(self.raw_map.shape))
            return bridge.cv2_to_imgmsg(self.raw_map, encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr("CMP: Unable to convert raw_map to a ROS Image. Error: " + e)

    def get_occ_map_msg(self):
        """
        @usage raw_map_pub.publish(map_proc.get_raw_map_msg())
        @return the processed coarse map occupancy grid as a ROS Image message.
        """
        try:
            rospy.loginfo("CMP: Returning processed (coarse) occupancy map with shape {:}.".format(self.occ_map.shape))
            return bridge.cv2_to_imgmsg(self.occ_map, encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr("CMP: Unable to convert occ_map to a ROS Image. Error: " + e)


