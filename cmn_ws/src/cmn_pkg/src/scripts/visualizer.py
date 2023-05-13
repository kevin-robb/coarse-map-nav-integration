#!/usr/bin/env python3

import rospy
import rospkg, yaml
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backend_bases import MouseButton
import cv2
from math import sin, cos

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
        g_goal_cell_px = (event.ydata, event.xdata)
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
    veh_pose_true = None # Most recent ground-truth vehicle pose. (3x1 numpy array of (x,y,yaw)^T)
    veh_pose_estimate = None # Most recent localization estimate of the vehicle pose. (3x1 numpy array of (x,y,yaw)^T)
    particle_set = None # Set of all particles currently in the particle filter. Only set if the PF is being used. (Nx3 numpy array)
    planned_path = None # Full path being planned by the motion controller, in form [(row1,col1),...,(rowN,colN)]
    goal_cell = None # Current goal cell in pixels.

    def __init__(self):
        """
        Initialize the visualization.
        """
        self.read_params()
        if not self.enabled:
            return

        # make live plot bigger.
        plt.rcParams["figure.figsize"] = (9,9)

        # startup the plot.
        self.fig = plt.figure(figsize=(8, 6)) 
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
        self.ax0 = plt.subplot(gs[0])
        plt.title("Ground Truth Map & Vehicle Pose")
        plt.axis("off")
        self.ax1 = plt.subplot(gs[1])
        plt.title("Observation")
        # set constant plot params.
        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        # allow clicking on the plot to do things (like kill the node).
        plt.connect('button_press_event', on_click)
        plt.show()

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

    def set_veh_pose_in_obs_region(self, veh_px_vert_from_bottom_on_obs, veh_px_horz_from_center_on_obs, obs_width_px, obs_resolution, map_downscale_ratio):
        """
        Add vehicle pose relative to observation region for clarity.
        """
        if not self.enabled:
            return
        # This is static, so only need to plot it once.
        if "veh_pose_obs" in self.plots.keys():
            return
        # Determine cell location from config vals.
        col = veh_px_vert_from_bottom_on_obs-0.5
        row = obs_width_px // 2 + veh_px_horz_from_center_on_obs
        # NOTE since it's plotted sideways, robot pose is always facing to the right.
        d_col = 0.5*obs_resolution
        d_row = 0.0
        # Size depends on relative resolutions of observation and map.
        wid = 0.01/obs_resolution/map_downscale_ratio

        # Plot it.
        self.plots["veh_pose_obs"] = self.ax1.arrow(col, row, d_col, d_row, color="blue", width=wid, zorder = 2)

    def set_true_veh_pose_px(self, row, col, angle):
        """
        Set a new true vehicle pose, which will be displayed on the viz from now on.
        @note We only have this when running in simulation.
        @param row, col - Cell location on map.
        @param angle - Orientation (yaw) in radians.
        """
        self.veh_pose_true = [row, col, angle]

    def set_map(self, occ_map):
        """
        Set the occupancy grid map which will be displayed in the background of the viz's main window.
        @note This should be called after allowing the Simulator instance to process the map; otherwise, coordinate systems will not align in the plot.
        @param occ_map Image of the map.
        """
        self.occ_map = occ_map
        if not self.enabled:
            return
        # Add the map to our figure.
        self.ax0.imshow(occ_map, cmap="gray", vmin=0, vmax=1)

    def set_estimated_veh_pose_px(self, row, col, angle):
        """
        Set a new estimated vehicle pose, which will be displayed on the viz from now on.
        @param row, col - Cell location on map.
        @param angle - Orientation (yaw) in radians.
        """
        self.veh_pose_estimate = [row, col, angle]

    def set_particle_set(self, particle_set):
        """
        Get the full set of particle positions from the current PF iteration.
        @param particle_set numpy array of dimension num_particles x 3, where each row is a vehicle pose's (row,col,yaw).
        """
        self.particle_set = particle_set

    def set_planned_path(self, path):
        """
        Get the full planned path, and save it to display on viz.
        @param path List of (row, col) cell values making up the path.
        """
        self.planned_path = path

    def update(self):
        """
        Update the plot with all the most recent data, and redraw the viz.
        """
        if not self.enabled:
            return
        
        # First, check if a new goal cell has been chosen by the user clicking the plot.
        global g_goal_cell_px
        if g_goal_cell_px is not None:
            self.goal_cell = g_goal_cell_px
            # Wipe it so we don't check constantly.
            g_goal_cell_px = None

        if self.veh_pose_true is not None:
            # Add the new (ground truth) vehicle pose to the viz.
            self.remove_plot("veh_pose_true")
            self.plots["veh_pose_true"] = self.ax0.arrow(self.veh_pose_true[1], self.veh_pose_true[0], 0.5*cos(self.veh_pose_true[2]), -0.5*sin(self.veh_pose_true[2]), color="blue", width=1.0, label="True Vehicle Pose")

        if self.veh_pose_estimate is not None:
            # Add the most recent localization estimate to the viz.
            self.remove_plot("veh_pose_est")
            self.plots["veh_pose_est"] = self.ax0.arrow(self.veh_pose_estimate[1], self.veh_pose_estimate[0], 0.5*cos(self.veh_pose_estimate[2]), -0.5*sin(self.veh_pose_estimate[2]), color="green", width=1.0, zorder = 3, label="PF Estimate")

        if self.particle_set is not None:
            # Plot the set of particles in the PF.
            self.remove_plot("particle_set")
            self.plots["particle_set"] = self.ax0.scatter(self.particle_set[:,1], self.particle_set[:,0], s=10, color="red", zorder=0, label="All Particles")
        
        if self.planned_path is not None:
            # Plot the full path the motion controller is attempting to follow.
            self.remove_plot("planned_path")
            path_r = [self.planned_path[i][0] for i in range(len(self.planned_path))]
            path_c = [self.planned_path[i][1] for i in range(len(self.planned_path))]
            self.plots["planned_path"] = self.ax0.scatter(path_c, path_r, s=3, color="purple", zorder=1, label="Planned Path")

        if self.observation_region is not None:
            # Plot the bounding box on the base map.
            box = cv2.boxPoints(self.observation_region)
            box_x_coords = [box[i,0] for i in range(box.shape[0])] + [box[0,0]]
            box_y_coords = [box[i,1] for i in range(box.shape[0])] + [box[0,1]]
            self.remove_plot("obs_bounding_box")
            self.plots["obs_bounding_box"] = self.ax0.plot(box_x_coords, box_y_coords, "r-", zorder=2, label="Observed Area")

        if self.observation is not None:
            # Plot the new observation.
            self.remove_plot("obs_img")
            self.plots["obs_img"] = self.ax1.imshow(self.observation, cmap="gray", vmin=0, vmax=1)

        self.ax0.legend(loc="upper left")
        plt.draw()
        plt.pause(self.dt)
