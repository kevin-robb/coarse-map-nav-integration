#!/usr/bin/env python3

import rospy
import rospkg, yaml
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import cv2
import numpy as np
from math import sin, cos

from scripts.basic_types import PoseMeters, PosePixels
from scripts.map_handler import MapFrameManager

### GLOBAL VARS ###
g_goal_cell_px = None
###################

def on_click(event):
    """
    Do something when the user clicks on the viz plot.
    """
    if event.button is MouseButton.LEFT:
        # Record clicked point as new goal for path planning.
        global g_goal_cell_px
        g_goal_cell_px = PosePixels(event.ydata, event.xdata)
        rospy.loginfo("VIZ: Recording new goal point ({:}, {:}).".format(event.ydata, event.xdata))
    elif event.button is MouseButton.RIGHT:
        # kill the node.
        rospy.loginfo("VIZ: Killing node because you right-clicked on the plot.")
        exit()

class Visualizer:
    """
    Class to handle updating the live viz with any dynamically changing data.
    """
    # Flag describing if visualization is enabled.
    enabled = False
    # Set time to pause after updating the plot.
    dt = 0.0001
    # Place to store plots so we can remove/update them some time later.
    plots = {}
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
            self.enabled = config["enable_live_viz"]
    
    def remove_plot(self, name):
        """
        Remove the plot if it already exists.
        """
        if name in self.plots.keys():
            try:
                # this works for stuff like scatter(), arrow()
                self.plots[name].remove()
            except:
                # the first way doesn't work for plt.plot()
                line = self.plots[name].pop(0)
                line.remove()
            # remove the key.
            del self.plots[name]

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
        self.occ_map = mfm.map
        # Compute vehicle pose relative to observation region, now that we've set mfm and have the needed configs.
        self.set_veh_pose_in_obs_region()

    def set_veh_pose_in_obs_region(self):
        """
        Add vehicle pose relative to observation region for clarity.
        """
        if not self.mfm.initialized:
            return
        # This is static, so only need to plot it once.
        if "veh_pose_obs" in self.plots.keys():
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

    def set_true_veh_pose_px(self, pose_px:PosePixels):
        """
        Set a new true vehicle pose, which will be displayed on the viz from now on.
        @note We only have this when running in simulation.
        @param pose_px
        """
        self.veh_pose_true = pose_px

    def set_estimated_veh_pose_px(self, pose_px:PosePixels):
        """
        Set a new estimated vehicle pose, which will be displayed on the viz from now on.
        @param pose_px
        """
        self.veh_pose_estimate = pose_px

    def set_particle_set(self, particle_set_px):
        """
        Get the full set of particle positions from the current PF iteration.
        @param particle_set_px - List of PosePixels.
        """
        self.particle_set = particle_set_px

    def set_planned_path(self, path):
        """
        Get the full planned path, and save it to display on viz.
        @param path List of PosePixels making up the path.
        """
        self.planned_path = path

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
            ax0.arrow(self.veh_pose_true.c, self.veh_pose_true.r, 0.5*cos(self.veh_pose_true.yaw), -0.5*sin(self.veh_pose_true.yaw), color="blue", width=1.0, label="True Vehicle Pose")

        # Add the most recent localization estimate to the viz.
        if self.veh_pose_estimate is not None:
            ax0.arrow(self.veh_pose_estimate.c, self.veh_pose_estimate.r, 0.5*cos(self.veh_pose_estimate.yaw), -0.5*sin(self.veh_pose_estimate.yaw), color="green", width=1.0, zorder = 3, label="PF Estimate")

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
            ax1.arrow(**self.veh_pose_in_obs_region, color="blue", zorder = 2)

        # Add the legend, including info from both plots.
        ax0.legend(loc="upper left")

        # Retrieve a view on the renderer buffer
        canvas.draw()
        buf = canvas.buffer_rgba()
        # convert to a NumPy array
        result_img = np.asarray(buf)
        # Convert to the correct color scheme.
        return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    