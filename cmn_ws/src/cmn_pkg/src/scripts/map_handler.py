 #!/usr/bin/env python3

"""
Functions that will be useful in more than one node in this project.
"""

import rospkg, yaml, cv2, rospy, os
import numpy as np
from math import sin, cos, remainder, tau, ceil, pi
from random import random, randrange
from cv_bridge import CvBridge, CvBridgeError

from scripts.rotated_rectangle_crop_opencv.rotated_rect_crop import crop_rotated_rectangle
from scripts.basic_types import PoseMeters, PosePixels, Pose

#### GLOBAL VARIABLES ####
bridge = CvBridge()
#########################

def clamp(val:float, min_val:float, max_val:float):
    """
    Clamp the value val in the range [min_val, max_val].
    @return float, the clamped value.
    """
    return min(max(min_val, val), max_val)


class CoarseMapProcessor:
    """
    Class to handle reading the coarse map from file, and doing any pre-processing.
    """
    pkg_path = None # Filepath to the cmn_pkg package.
    # Global variables for class
    verbose = False
    show_map_images = False
    # Map configs
    map_fpath = None # Full filepath to map image.
    obs_balloon_radius = 0 # Radius in pixels to balloon occupied pixels by during pre-processing.
    map_resolution_raw = None # Resolution of the raw map image in meters/pixel. NOTE this will eventually be unknown.
    map_resolution_desired = None # Resolution in meters/pixel the map will be scaled to.
    map_downscale_ratio = None # Side length of original map times this ratio yields the new side length to satisfy desired resolution.
    # Map images
    raw_map = None # Original coarse map including color.
    occ_map = None # Thresholded & binarized coarse map to create an occupancy grid. Free=1, Occupied=0.
    inv_occ_map = None # Inverse of occ_map. Free=0, Occupied=1.

    def __init__(self):
        """
        Create instance and set important global params.
        """
        # Determine filepath.
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('cmn_pkg')
        # Open the yaml and get the relevant params.
        with open(self.pkg_path+'/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            self.verbose = config["verbose"]
            # Map processing params.
            self.show_map_images = config["map"]["show_images_during_pre_proc"]
            self.map_fpath = self.pkg_path + "/config/maps/" + config["map"]["fname"]
            self.obs_balloon_radius = config["map"]["obstacle_balloon_radius"]

            # Some map params are in a separate yaml unique to each map.
            map_name = os.path.splitext(config["map"]["fname"])[0]
            map_yaml_fpath = os.path.join(self.pkg_path, "config/maps/"+map_name+".yaml")
            # Use the default if this path doesn't exist.
            if not os.path.exists(map_yaml_fpath):
                rospy.logwarn("CMP: map-specific yaml {:} not found. Using maps/default.yaml instead.".format(map_yaml_fpath))
                map_yaml_fpath = os.path.join(self.pkg_path, "config/maps/default.yaml")
            with open(map_yaml_fpath, 'r') as file2:
                map_config = yaml.safe_load(file2)
                self.map_resolution_raw = map_config["resolution"]
                self.map_occ_thresh_min = map_config["occ_thresh_min"]
                self.map_occ_thresh_max = map_config["occ_thresh_max"]

            # Based on the image resolution and the desired resolution, determine a downscaling ratio.
            self.map_resolution_desired = config["map"]["desired_meters_per_pixel"]
            self.map_downscale_ratio = self.map_resolution_raw / self.map_resolution_desired
            if self.verbose:
                rospy.loginfo("CMP: downscale ratio is {:.3f}".format(self.map_downscale_ratio))

        # Read in the map and perform all necessary pre-processing.
        self.read_coarse_map_from_file()

    def read_coarse_map_from_file(self):
        """
        Read the coarse map image from the provided filepath.
        Save the map itself to use for visualizations.
        Process the image by converting it to an occupancy grid.
        """
        # If the map "image" is a numpy array, skip initial image processing.
        if os.path.splitext(self.map_fpath)[1] == ".npy":
            img = np.load(self.map_fpath)
        else:
            # Read map image and account for possible white = transparency that cv2 will think is black.
            # https://stackoverflow.com/questions/31656366/cv2-imread-and-cv2-imshow-return-all-zeros-and-black-image/62985765#62985765
            img = cv2.imread(self.map_fpath, cv2.IMREAD_UNCHANGED)
            if len(img.shape) >= 3 and img.shape[2] == 4: # we have an alpha channel.
                a1 = ~img[:,:,3] # extract and invert that alpha.
                img = cv2.add(cv2.merge([a1,a1,a1,a1]), img) # add up values (with clipping).
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # strip alpha channels.

        if self.verbose:
            rospy.loginfo("CMP: Read raw coarse map image with shape {:}".format(img.shape))
        if self.show_map_images:
            cv2.imshow('initial map', img); cv2.waitKey(0); cv2.destroyAllWindows()

        # Downsize the image to the desired resolution.
        img = cv2.resize(img, (int(img.shape[1] * self.map_downscale_ratio), int(img.shape[0] * self.map_downscale_ratio)), 0, 0, cv2.INTER_AREA)
        if self.verbose:
            rospy.loginfo("CMP: Resized coarse map to shape {:}".format(img.shape))
        if self.show_map_images:
            cv2.imshow('resized map', img); cv2.waitKey(0); cv2.destroyAllWindows()

        if len(img.shape) >= 3 and img.shape[2] >= 3:
            # Convert from BGR to RGB and save the color map for any viz.
            self.raw_map = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Turn this into a grayscale img and then to a binary map.
            occ_map_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), self.map_occ_thresh_min, self.map_occ_thresh_max, cv2.THRESH_BINARY)[1]
            # Normalize to range [0,1].
            occ_map_img = np.divide(occ_map_img, 255)
            if self.verbose:
                rospy.loginfo("CMP: Thresholded/binarized map to shape {:}".format(img.shape))
            if self.show_map_images:
                cv2.imshow("Thresholded Map", occ_map_img); cv2.waitKey(0); cv2.destroyAllWindows()
        else:
            # Image is already single-channel.
            occ_map_img = img
        
        # Round so all cells are either completely free (1) or occluded (0).
        self.occ_map = np.round(occ_map_img)

        if self.obs_balloon_radius != 0:
            # Expand occluded cells so path planning won't take us right next to obstacles.
            if len(img.shape) >= 3 and img.shape[2] >= 3:
                if self.obs_balloon_radius == 0:
                    rospy.logwarn("CMP: For some reason everything breaks if we skip the ballooning step, so running with minimal radius of 1.")
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
            if self.verbose:
                rospy.loginfo("CMP: Occ map value frequencies: "+str(freqs[1])+" free, "+str(freqs[0])+" occluded.")

        # Create inverted occupancy map which meets the CMN expected format.
        self.inv_occ_map = np.logical_not(self.occ_map).astype(int)


class MapFrameManager(CoarseMapProcessor):
    """
    Class to handle map/vehicle coordinate transforms.
    Also uses the rotated_rectangle_crop_opencv functions to crop out an observation region corresponding to a particular vehicle pose.
    Used both for ground-truth observation generation as well as particle likelihood evalutation.
    """
    initialized = False
    map_with_border = None # 2D numpy array of the global map, including a border region. Free=1, Occupied=0.
    inv_map_with_border = None # Inverse of map_with_border. Free=0, Occupied=1.

    # Config flags. If using discrete state space, robot's yaw must be axis-aligned.
    use_discrete_state_space = False
    show_obs_gen_debug:bool = False

    def __init__(self, use_discrete_state_space:bool):
        """
        Create instance and set important global params.
        @param use_discrete_state_space - Flag to represent pose discretely, so angle is locked to cardinal directions.
        """
        super().__init__()
        self.use_discrete_state_space = use_discrete_state_space
        # Open the yaml and get the relevant params.
        with open(os.path.join(self.pkg_path, "config/config.yaml"), 'r') as file:
            config = yaml.safe_load(file)
            # Observation region size.
            self.obs_resolution = config["observation"]["resolution"] / self.map_downscale_ratio
            self.obs_height_px = config["observation"]["height"]
            self.obs_width_px = config["observation"]["width"]
            self.obs_height_px_on_map = int(self.obs_height_px * self.obs_resolution / self.map_resolution_desired)
            self.obs_width_px_on_map = int(self.obs_width_px * self.obs_resolution / self.map_resolution_desired)
            # Vehicle position relative to observation region.
            self.veh_px_horz_from_center_on_obs = (config["observation"]["veh_horz_pos_ratio"] - 0.5) * self.obs_width_px
            self.veh_px_vert_from_bottom_on_obs = config["observation"]["veh_vert_pos_ratio"] * self.obs_width_px
            self.veh_px_horz_from_center_on_map = self.veh_px_horz_from_center_on_obs * self.obs_resolution / self.map_resolution_desired
            self.veh_px_vert_from_bottom_on_map = self.veh_px_vert_from_bottom_on_obs * self.obs_resolution / self.map_resolution_desired
        # Add borders to the map to prepare for observation extraction.
        self.setup_map()

    def setup_map(self):
        """
        Setup the map that will be used to crop out observation regions.
        Determine valid vehicle bounds, and add padding around the map border.
        """
        # Set the map as it is temporarily so we can use it to determine vehicle bounds.
        self.map_with_border = self.occ_map.copy()
        # Set the map bounds in meters. This prevents true vehicle pose from leaving the map.
        self.map_x_min_meters, self.map_y_min_meters = self.transform_map_px_to_m(self.map_with_border.shape[1]-1, 0)
        self.map_x_max_meters, self.map_y_max_meters = self.transform_map_px_to_m(0, self.map_with_border.shape[0]-1)

        """
        When generating an observation, it is possible the desired region will be partially outside the bounds of the map.
        To prevent potential errors, create a padded version of the map with enough extra rows/cols to ensure this won't happen.
        Expand all dimensions by the diagonal of the observation area to cover all possible situations.
        All extra space will be assumed to be occluded cells (value = 0.0).
        """
        max_obs_dim = ceil(np.sqrt(self.obs_height_px_on_map**2 + self.obs_width_px_on_map**2))
        max_obs_dim = 3 # DEBUG to prevent huge borders.
        self.map_with_border = cv2.copyMakeBorder(self.map_with_border, max_obs_dim, max_obs_dim, max_obs_dim, max_obs_dim, cv2.BORDER_CONSTANT, None, 0.0)
        self.initialized = True

        # Update inverse map with a border as well.
        self.inv_map_with_border = np.logical_not(self.map_with_border).astype(int)

    def transform_pose_px_to_m(self, pose_px:PosePixels) -> PoseMeters:
        """
        Convert a pose from pixels to meters.
        """
        if pose_px is None:
            return None
        x, y = self.transform_map_px_to_m(pose_px.r, pose_px.c)
        # Return new values, preserving yaw.
        return PoseMeters(x, y, pose_px.yaw)

    def transform_map_px_to_m(self, row:int, col:int):
        """
        Given coordinates of a cell on the ground-truth map, compute the equivalent position in meters.
        Origin (0,0) in meters corresponds to center of map.
        Origin in pixels is top left, and coords are strictly nonnegative.
        @return tuple of floats (x, y)
        """
        # Clamp inputs within legal range of values.
        row = int(clamp(row, 0, self.map_with_border.shape[0]-1))
        col = int(clamp(col, 0, self.map_with_border.shape[1]-1))
        # Get pixel difference from map center.
        row_offset = row - self.map_with_border.shape[0] // 2
        col_offset = col - self.map_with_border.shape[1] // 2
        # Convert from pixels to meters.
        x = self.map_resolution_desired * col_offset
        y = self.map_resolution_desired * -row_offset
        return x, y

    def transform_pose_m_to_px(self, pose_m:PoseMeters) -> PosePixels:
        """
        Convert a pose from meters to pixels.
        """
        if pose_m is None:
            return None
        r, c = self.transform_map_m_to_px(pose_m.x, pose_m.y)
        # Return new values, preserving yaw.
        return PosePixels(r, c, pose_m.yaw)

    def transform_map_m_to_px(self, x:float, y:float):
        """
        Given coordinates of a vehicle pose in meters, compute the equivalent cell in pixels.
        Origin (0,0) in meters corresponds to center of map.
        Origin in pixels is top left, and coords are strictly nonnegative.
        @return tuple of ints (row, col)
        """
        # Convert from meters to pixels.
        col_offset = x / self.map_resolution_desired
        row_offset = -y / self.map_resolution_desired
        # Shift origin from center to corner.
        row = row_offset + self.map_with_border.shape[0] // 2
        col = col_offset + self.map_with_border.shape[1] // 2
        # Clamp within legal range of values.
        row = int(clamp(row, 0, self.map_with_border.shape[0]-1))
        col = int(clamp(col, 0, self.map_with_border.shape[1]-1))
        return row, col

    def extract_observation_region(self, veh_pose:Pose, pose_in_meters:bool=True):
        """
        Crop out the map region for the observation corresponding to the given vehicle pose.
        @param veh_pose PoseMeters or PosePixels instance containing a vehicle pose.
        @param pose_in_meters - Flag to tell us whether veh_pose arg is PosePixels or PoseMeters.
        @return tuple containing (observation image, bounding box region in map frame)
        """
        # Compute vehicle pose in pixels.
        if pose_in_meters:
            veh_pose_px = self.transform_pose_m_to_px(veh_pose)
        else:
            veh_pose_px = veh_pose
        # Project ahead of vehicle pose to determine center of observation region.
        center_col = veh_pose_px.c + (self.obs_height_px_on_map / 2 - self.veh_px_vert_from_bottom_on_map) * cos(veh_pose_px.yaw)
        center_row = veh_pose_px.r - (self.obs_height_px_on_map / 2 - self.veh_px_vert_from_bottom_on_map) * sin(veh_pose_px.yaw)
        center = (center_col, center_row)
        # Rotate by agent yaw to get relative observation.
        angle = -np.rad2deg(veh_pose_px.yaw)
        rect = None
        # Extract and rotate the region differently based on run mode.
        if not self.use_discrete_state_space:
            # Create the rotated rectangle.
            rect = (center, (self.obs_height_px_on_map, self.obs_width_px_on_map), angle)
            # Crop out the rotated rectangle and reorient it.
            obs_img = crop_rotated_rectangle(image = self.map_with_border, rect = rect)
            # If area was partially outside the image, this will return None.
            if obs_img is None:
                rospy.logerr("MFM: Could not generate observation image.")
                return None, None
        
        else: # Discrete case. Only need to rotate by 90 degree increments.
            # NOTE assumes obs is square for simplicity. Can expand complexity later if needed.
            half_obs_dim = self.obs_height_px_on_map // 2 # rounds down.
            obs_img = self.map_with_border[int(center_row)-half_obs_dim:int(center_row)+half_obs_dim+1, int(center_col)-half_obs_dim:int(center_col)+half_obs_dim+1]
            # Rotate by agent yaw.
            agent_dir_str = veh_pose_px.get_direction()
            # Rotate the egocentric local occupancy to face EAST
            if agent_dir_str == "east":
                pass
            elif agent_dir_str == "north":
                obs_img = np.rot90(obs_img, k=-1)
            elif agent_dir_str == "west":
                obs_img = np.rot90(obs_img, k=2)
            elif agent_dir_str == "south":
                obs_img = np.rot90(obs_img, k=1)
            else:
                raise Exception("Invalid agent direction")

        if self.show_obs_gen_debug:
            # make copy of the map image to show exactly what is being selected.
            img = cv2.cvtColor(self.map_with_border.copy(), cv2.COLOR_GRAY2BGR)
            img = cv2.circle(img, [int(p) for p in center], 1, (255,0,0), -1)
            cv2.imshow("obs center on map", img)
            # show the cropped observation before resizing.
            obs_img_viz = cv2.cvtColor(obs_img.copy(), cv2.COLOR_GRAY2BGR)
            cv2.imshow("cropped obs at full res", obs_img_viz)
            cv2.waitKey(0)

        # Resize observation to desired resolution.
        obs_img = cv2.resize(obs_img, (self.obs_height_px, self.obs_width_px))
        # Return both the image and the rect points for the viz to use for plotting.
        return obs_img, rect
    
    def choose_random_free_cell(self) -> PosePixels:
        """
        Choose a random free cell on the map to return.
        @return PosePixels
        """
        # Keep choosing random cells until one is free.
        while True:
            r = randrange(0, self.map_with_border.shape[0])
            c = randrange(0, self.map_with_border.shape[1])
            if self.map_with_border[r, c] == 1: # this cell is free.
                return PosePixels(r, c)

    def generate_random_valid_veh_pose(self, in_meters:bool = True) -> Pose:
        """
        Generate a vehicle pose at random.
        Ensure this position is within map bounds, and is located in free space.
        @param in_meters - True to generate PoseMeters, False to generate PosePixels.
        @return PoseMeters of the created pose.
        """
        if self.use_discrete_state_space:
            # Choose yaw randomly from cardinal directions.
            angles = [0.0, pi/2, pi, -pi/2]
            yaw = angles[randrange(0, len(angles))]
        else:
            # Choose any yaw at random, with no validity condition.
            yaw = remainder(random() * tau, tau)
        # Choose a random vehicle cell in px so we can first check if it's occluded.
        free_cell:PosePixels = self.choose_random_free_cell()
        # Set yaw, since it will be kept through coords transform.
        free_cell.yaw = yaw

        if in_meters:
            # Convert this cell into meters.
            return self.transform_pose_px_to_m(free_cell)
        else:
            # Return the pose in pixels.
            return free_cell
            
    def veh_pose_m_in_collision(self, veh_pose_m:PoseMeters) -> bool:
        """
        Given a vehicle pose, determine if this is in collision on the map.
        @param veh_pose_m - PoseMeters instance containing x,y,yaw.
        @return true if the vehicle pose is in collision on the occupancy grid map.
        """
        r, c = self.transform_map_m_to_px(veh_pose_m.x, veh_pose_m.y)
        return self.map_with_border[r, c] != 1 # return false if this cell is free, and true otherwise.
    
    def veh_pose_px_in_collision(self, veh_pose_px:PosePixels) -> bool:
        """
        Given a vehicle pose, determine if this is in collision on the map.
        @param veh_pose_px - PosePixels instance containing r,c,yaw.
        @return true if the vehicle pose is in collision on the occupancy grid map.
        """
        return self.map_with_border[veh_pose_px.r, veh_pose_px.c] != 1 # return false if this cell is free, and true otherwise.

class Simulator(MapFrameManager):
    """
    Class to support running the project in simulation, without the robot providing real data.
    """
    # Ground-truth vehicle pose in the global map frame (origin at center).
    veh_pose_true_px:PosePixels = None # (r,c,yaw) in pixels and radians.
    veh_pose_true_meters:PoseMeters = None # (x,y,yaw) in meters and radians.
    veh_pose_true_se2 = None # 3x3 matrix of SE(2) representation of veh_pose_true_meters.

    discrete_forward_dist:float = None # Forward distance corresponding to a discrete "move_forward" action. Used to check if a motion would result in collision.

    def __init__(self, use_discrete_state_space):
        """
        Use the MapFrameManager's setup functions to assign the map and setup all validity conditions such as bounds and free cells.
        Also initialize the simulator's state.
        @param use_discrete_state_space - Flag to represent pose discretely, so angle is locked to cardinal directions.
        """
        super().__init__(use_discrete_state_space)
        # Read params only needed for the simulator.
        with open(self.pkg_path+'/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            self.dt = config["dt"]
            # Constraints.
            self.max_lin_vel = config["constraints"]["max_lin_vel"]
            self.min_ang_vel = config["constraints"]["min_ang_vel"]
            self.max_ang_vel = config["constraints"]["max_ang_vel"]
            self.allow_motion_through_occupied_cells = config["simulator"]["allow_motion_through_occupied_cells"]
            # Params for discrete motion that are good to know.
            self.discrete_forward_dist = abs(config["actions"]["discrete_forward_dist"])
            # Debug flag (can only be true when using simulator).
            self.show_obs_gen_debug = config["simulator"]["show_obs_gen_debug"]
        # Initialize the ground truth vehicle pose randomly on the map.
        self.veh_pose_true_px = self.generate_random_valid_veh_pose(False)
        self.veh_pose_true_meters = self.transform_pose_px_to_m(self.veh_pose_true_px)

    def propagate_with_vel(self, lin:float, ang:float):
        """
        Given a commanded velocity twist, move the robot by v*dt.
        @param lin, Commanded linear velocity (m/s).
        @param ang, Commanded angular velocity (rad/s).
        """
        # Clamp commands to allowed values (redundant since clamping is done in MOT, but just to be safe).
        fwd_dist = self.dt * clamp(lin, 0, self.max_lin_vel) # dt * meters/sec forward
        dtheta = self.dt * clamp(ang, -self.max_ang_vel, self.max_ang_vel) # dt * radians/sec CCW
        # Use our other function to command this distance.
        self.propagate_with_dist(fwd_dist, dtheta)

    def get_veh_pose_after_motion(self, lin:float, ang:float) -> PoseMeters:
        """
        Apply a motion to the ground truth vehicle pose to get the resulting pose.
        @param lin, Commanded linear distance (m).
        @param ang, Commanded angular distance (rad).
        @return PoseMeters, the resulting pose.
        """
        # TODO Perturb with some noise.
        # Compute a proposed new vehicle pose, and check if it's allowed before moving officially.
        veh_pose_proposed = PoseMeters()
        veh_pose_proposed.x = self.veh_pose_true_meters.x + lin * cos(self.veh_pose_true_meters.yaw)
        veh_pose_proposed.y = self.veh_pose_true_meters.y + lin * sin(self.veh_pose_true_meters.yaw)
        # Clamp the vehicle pose to remain inside the map bounds.
        veh_pose_proposed.x = clamp(veh_pose_proposed.x, self.map_x_min_meters, self.map_x_max_meters)
        veh_pose_proposed.y = clamp(veh_pose_proposed.y, self.map_y_min_meters, self.map_y_max_meters)
        # Keep yaw normalized to (-pi, pi).
        veh_pose_proposed.yaw = remainder(self.veh_pose_true_meters.yaw + ang, tau)
        return veh_pose_proposed

    def propagate_with_dist(self, lin:float, ang:float):
        """
        Given a commanded motion, move the robot accordingly.
        @param lin, Commanded linear distance (m).
        @param ang, Commanded angular distance (rad).
        """
        veh_pose_proposed = self.get_veh_pose_after_motion(lin, ang)
        # Determine if this vehicle pose is allowed.
        if not self.allow_motion_through_occupied_cells and self.veh_pose_m_in_collision(veh_pose_proposed):
            rospy.logwarn("SIM: Command would move vehicle to invalid pose. Only allowing angular motion.")
            self.veh_pose_true_meters.yaw = veh_pose_proposed.yaw
        else:
            self.veh_pose_true_meters = veh_pose_proposed
            if self.verbose:
                rospy.loginfo("SIM: Allowing command. Veh pose is now " + str(self.veh_pose_true_meters))
        # Update the pixels representation as well.
        self.veh_pose_true_px = self.transform_pose_m_to_px(self.veh_pose_true_meters)

    def propagate_with_discrete_motion(self, action:str):
        """
        Apply a discrete action to the true robot pose.
        @param action - one of "move_forward", "turn_left", "turn_right"
        """
        if action == "turn_left":
            self.veh_pose_true_px.yaw = remainder(self.veh_pose_true_px.yaw + pi/2, tau)
        elif action == "turn_right":
            self.veh_pose_true_px.yaw = remainder(self.veh_pose_true_px.yaw - pi/2, tau)
        elif action == "move_forward":
            # Move forward one pixel, if the pixel we would move to is free.
            agent_dir = self.veh_pose_true_px.get_direction()
            dr = {"east" : 0, "west" : 0, "north" : -1, "south" : 1}
            dc = {"east" : 1, "west" : -1, "north" : 0, "south" : 0}
            proposed_new_pose = PosePixels(self.veh_pose_true_px.r + dr[agent_dir], self.veh_pose_true_px.c + dc[agent_dir], self.veh_pose_true_px.yaw)
            if not self.veh_pose_px_in_collision(proposed_new_pose):
                self.veh_pose_true_px = proposed_new_pose
            # self.propagate_with_dist(self.map_resolution_desired, 0)
        # Update the meters representation as well.
        self.veh_pose_true_meters = self.transform_pose_px_to_m(self.veh_pose_true_px)

    def get_true_observation(self):
        """
        @return the ground-truth observation, using our ground-truth map and vehicle pose.
        """
        # Use more general utilities class to generate the observation, using the known true vehicle pose.
        return self.extract_observation_region(self.veh_pose_true_px, False)

    def agent_is_facing_wall(self) -> bool:
        """
        Check if the agent is currently facing a wall. If so, a forward motion will not be allowed to happen.
        """
        # Propagate the true vehicle pose forward by the discrete forward motion distance, and check if that cell is occupied.
        return self.veh_pose_m_in_collision(self.get_veh_pose_after_motion(self.discrete_forward_dist, 0.0))

