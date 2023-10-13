#!/usr/bin/env python3

import rospy
import rospkg, yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import cv2
import numpy as np
from math import sin, cos

from scripts.basic_types import PosePixels
from scripts.map_handler import MapFrameManager

class Visualizer:
    """
    Class to handle updating the live viz with any dynamically changing data.
    """
    verbose = False
    # Keep track of most recent data for all vars we want to plot.
    occ_map = None # Occupancy grid map that will be displayed in the background of the main viz window.
    observation = None # Most recent observation image.
    observation_region = None # Area in front of robot being used to generate observations.
    veh_pose_true = None # Most recent ground-truth vehicle pose.
    veh_pose_estimate = None # Most recent localization estimate of the vehicle pose.
    particle_set = None # Set of all particles currently in the particle filter. Only set if the PF is being used. (Nx3 numpy array)
    planned_path = None # Full path being planned by the motion controller, as list of PosePixels.
    goal_cell = None # Current goal cell in pixels. Instance of PosePixels.
    veh_pose_in_obs_region = None # Dict of veh pose details relative to observation frame. This is constant once set.
    veh_pose_displ_len, veh_pose_displ_wid = None, None # Size to show veh pose(s) on the plot. This is constant once set.

    mfm = None # Reference to MapFrameManager allows access to important configs that get computed during map setup.

    def __init__(self):
        """
        Initialize the visualization.
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
            self.verbose = config["verbose"]

    def set_observation(self, obs_img, obs_rect=None):
        """
        Set a new observation image to be displayed in the viz from now on.
        @param obs_img The observation image.
        @param obs_rect (optional) The region in front of the robot being used to generate observations.
        """
        self.observation = obs_img
        self.observation_region = obs_rect

    def set_map_frame_manager(self, mfm:MapFrameManager):
        """
        Set our reference to the map frame manager, which allows us to use the map and coordinate transform functions.
        @param mfg MapFrameManager instance that has already been initialized with a map.
        """
        self.mfm = mfm
        # Setup the map on the figure.
        self.occ_map = mfm.map_with_border
        # Compute vehicle pose relative to observation region, now that we've set mfm and have the needed configs.
        self.set_veh_pose_in_obs_region()

        # Choose size to show robot pose(s) to ensure consistency and staying in frame.
        self.veh_pose_displ_len = 10 * self.mfm.map_resolution_desired / self.mfm.map_downscale_ratio # Must be nonzero so direction is stored, but make very small so we only see the triangle.
        self.veh_pose_displ_wid = 0.01 * self.mfm.map_resolution_desired / self.mfm.map_downscale_ratio # This controls the size of the triangle part of the arrow (what we care about).


    def set_veh_pose_in_obs_region(self):
        """
        Add vehicle pose relative to observation region for clarity.
        """
        if not self.mfm.initialized:
            return
        # Determine cell location from config vals.
        col = self.mfm.veh_px_vert_from_bottom_on_obs-0.5
        row = self.mfm.obs_width_px // 2 + self.mfm.veh_px_horz_from_center_on_obs
        # NOTE since it's plotted sideways, robot pose is always facing to the right.
        d_col = 0.5*self.mfm.obs_resolution
        d_row = 0.0
        # Size depends on relative resolutions of observation and map.
        wid = 0.01/self.mfm.obs_resolution/self.mfm.map_downscale_ratio
        # Set member var so we can add it to the plots later.
        self.veh_pose_in_obs_region = {"x" : col, "y" : row, "dx" : d_col, "dy" : d_row, "width" : wid}

    def get_updated_img(self):
        """
        Update the plot with all the most recent data, and redraw the viz.
        @ref https://stackoverflow.com/a/62040123/14783583
        @return new viz image as a cv/numpy matrix.
        """
        # make a Figure and attach it to a canvas.
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasAgg(fig)

        ######### LEFT SUBPLOT ###########
        ax0 = fig.add_subplot(1, 4, (1,3))
        # Add the occupancy grid map to the background.
        ax0.imshow(self.occ_map, cmap="gray", vmin=0, vmax=1)

        # Add the new (ground truth) vehicle pose to the viz.
        if self.veh_pose_true is not None:
            ax0.scatter(self.veh_pose_true.c, self.veh_pose_true.r, color="blue", label="True Vehicle Pose")
            ax0.arrow(self.veh_pose_true.c, self.veh_pose_true.r, self.veh_pose_displ_len*cos(self.veh_pose_true.yaw), -self.veh_pose_displ_len*sin(self.veh_pose_true.yaw), color="blue", width=self.veh_pose_displ_wid, head_width=0.01, head_length=0.1)

        # Add the most recent localization estimate to the viz.
        if self.veh_pose_estimate is not None:
            ax0.scatter(self.veh_pose_estimate.c, self.veh_pose_estimate.r, color="green", label="Vehicle Pose Estimate")
            ax0.arrow(self.veh_pose_estimate.c, self.veh_pose_estimate.r, self.veh_pose_displ_len*cos(self.veh_pose_estimate.yaw), -self.veh_pose_displ_len*sin(self.veh_pose_estimate.yaw), color="green", width=self.veh_pose_displ_wid, zorder = 3, head_width=0.01, head_length=0.1)

        # Plot the set of particles in the PF.
        if self.particle_set is not None:
            particles_r = [self.particle_set[i].r for i in range(len(self.particle_set))]
            particles_c = [self.particle_set[i].c for i in range(len(self.particle_set))]
            ax0.scatter(particles_c, particles_r, s=10, color="red", zorder=0, label="All Particles")
        
        # Plot the full path the motion controller is attempting to follow.
        if self.planned_path is not None:
            path_r = [self.planned_path[i].r for i in range(len(self.planned_path))]
            path_c = [self.planned_path[i].c for i in range(len(self.planned_path))]
            ax0.scatter(path_c, path_r, s=3, color="purple", zorder=1, label="Planned Path")

        # Plot the current goal point.
        if self.goal_cell is not None:
            ax0.scatter(self.goal_cell.c, self.goal_cell.r, color="yellow", label="Goal Cell")

        # Plot the bounding box on the base map.
        if self.observation_region is not None:
            box = cv2.boxPoints(self.observation_region)
            box_x_coords = [box[i,0] for i in range(box.shape[0])] + [box[0,0]]
            box_y_coords = [box[i,1] for i in range(box.shape[0])] + [box[0,1]]
            ax0.plot(box_x_coords, box_y_coords, "r-", zorder=2, label="Observed Area")

        ######### RIGHT SUBPLOT ###########
        ax1 = fig.add_subplot(1, 4, 4)
        # Plot the new observation.
        if self.observation is not None:
            ax1.imshow(self.observation, cmap="gray", vmin=0, vmax=1)
            # Plot the vehicle pose relative to the observation.
            if self.veh_pose_in_obs_region is not None:
                # Unpack the dictionary of its data that we computed earlier.
                # ax1.arrow(**self.veh_pose_in_obs_region, color="blue", zorder = 2)

                ax1.scatter(self.veh_pose_in_obs_region["x"], self.veh_pose_in_obs_region["y"], color="blue")
                ax1.arrow(self.veh_pose_in_obs_region["x"], self.veh_pose_in_obs_region["y"], self.veh_pose_in_obs_region["dx"], self.veh_pose_in_obs_region["dy"], color="blue", width=self.veh_pose_in_obs_region["width"], zorder = 2, head_width=0.01, head_length=0.25)

        # Add the legend, including info from both plots.
        ax0.legend(loc="upper left")

        # Retrieve a view on the renderer buffer
        canvas.draw()
        buf = canvas.buffer_rgba()
        # convert to a NumPy array
        result_img = np.asarray(buf)
        # Convert to the correct color scheme.
        return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    