#!/usr/bin/env python3

"""
Node for extremely basic testing of localization node in best-case-scenario:
 - "coarse map" used by pf is exactly the same as the ground truth map.
 - observations will be ground truth.
 - motion commands will be followed exactly, with no noise.
"""

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
import rospkg, yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backend_bases import MouseButton
import cv2
from cv_bridge import CvBridge
from math import remainder, tau, sin, cos, pi, ceil

from rotated_rectangle_crop_opencv.rotated_rect_crop import crop_rotated_rectangle

############ GLOBAL VARIABLES ###################
# ROS stuff.
bridge = CvBridge()
# Ground-truth vehicle pose (x,y,yaw) in meters and radians, in the global map frame (origin at center).
veh_pose_true = np.array([0.0, 0.0, 0.0])
veh_pose_est = np.array([0.0, 0.0, 0.0])
# Place to store plots so we can remove/update them some time later.
plots = {}
#################################################

########################### UTILITY/HELPER FUNCTIONS ###############################
def remove_plot(name):
    """
    Remove the plot if it already exists.
    """
    if name in plots.keys():
        try:
            # this works for stuff like scatter(), arrow()
            plots[name].remove()
        except:
            # the first way doesn't work for plt.plot()
            line = plots[name].pop(0)
            line.remove()
        # remove the key.
        del plots[name]

def on_click(event):
    # global clicked_points
    if event.button is MouseButton.LEFT:
        # kill the node.
        rospy.loginfo("Killing simulation_node because you clicked on the plot.")
        exit()
    elif event.button is MouseButton.RIGHT:
        # may want to use this for something eventually
        pass

def clamp(val:float, min_val:float, max_val:float):
    """
    Clamp the value val in the range [min_val, max_val].
    @return float, the clamped value.
    """
    return min(max(min_val, val), max_val)

def transform_map_px_to_m(row:int, col:int):
    """
    Given coordinates of a cell on the ground-truth map, compute the equivalent position in meters.
    Origin (0,0) in meters corresponds to center of map.
    Origin in pixels is top left, and coords are strictly nonnegative.
    @return tuple of floats (x, y)
    """
    # Clamp inputs within legal range of values.
    row = int(clamp(row, 0, occ_map_true.shape[0]-1))
    col = int(clamp(col, 0, occ_map_true.shape[1]-1))
    # Get pixel difference from map center.
    row_offset = row - occ_map_true.shape[0] // 2
    col_offset = col - occ_map_true.shape[1] // 2
    # Convert from pixels to meters.
    x = g_map_resolution * col_offset
    y = g_map_resolution * -row_offset
    return x, y

def transform_map_m_to_px(x:float, y:float):
    """
    Given coordinates of a vehicle pose in meters, compute the equivalent cell in pixels.
    Origin (0,0) in meters corresponds to center of map.
    Origin in pixels is top left, and coords are strictly nonnegative.
    @return tuple of ints (row, col)
    """
    # Convert from meters to pixels.
    col_offset = x / g_map_resolution
    row_offset = -y / g_map_resolution
    # Shift origin from center to corner.
    row = row_offset + occ_map_true.shape[0] // 2
    col = col_offset + occ_map_true.shape[1] // 2
    # Clamp within legal range of values.
    row = int(clamp(row, 0, occ_map_true.shape[0]-1))
    col = int(clamp(col, 0, occ_map_true.shape[1]-1))
    return row, col

def read_params():
    """
    Read configuration params from the yaml.
    """
    # Determine filepath.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('perception_pkg')
    # Open the yaml and get the relevant params.
    with open(pkg_path+'/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        global g_debug_mode, g_dt
        g_debug_mode = config["test"]["run_debug_mode"]
        g_dt = config["perception_node_dt"]
        # Map params.
        global g_map_resolution
        g_map_resolution = config["map"]["resolution"]
        # Rostopics.
        global g_topic_observations, g_topic_occ_map, g_topic_localization, g_topic_commands
        g_topic_observations = config["topics"]["observations"]
        g_topic_occ_map = config["topics"]["occ_map"]
        g_topic_localization = config["topics"]["localization"]
        g_topic_commands = config["topics"]["commands"]
        # Observation params.
        global g_obs_height_px, g_obs_width_px, g_obs_height_px_on_map, g_obs_width_px_on_map
        g_obs_resolution = config["observation"]["resolution"]
        g_obs_height_px = config["observation"]["height"]
        g_obs_width_px = config["observation"]["width"]
        g_obs_height_px_on_map = int(g_obs_height_px * g_obs_resolution / g_map_resolution)
        g_obs_width_px_on_map = int(g_obs_width_px * g_obs_resolution / g_map_resolution)
        # Vehicle position relative to observation region.
        global g_veh_px_horz_from_center_on_map, g_veh_px_horz_from_center_on_obs, g_veh_px_vert_from_bottom_on_map, g_veh_px_vert_from_bottom_on_obs
        g_veh_px_horz_from_center_on_obs = (config["observation"]["veh_horz_pos_ratio"] - 0.5) * g_obs_width_px
        g_veh_px_vert_from_bottom_on_obs = config["observation"]["veh_vert_pos_ratio"] * g_obs_width_px
        g_veh_px_horz_from_center_on_map = g_veh_px_horz_from_center_on_obs * g_obs_resolution / g_map_resolution
        g_veh_px_vert_from_bottom_on_map = g_veh_px_vert_from_bottom_on_obs * g_obs_resolution / g_map_resolution

################################ CALLBACKS #########################################
def get_command(msg:Vector3):
    """
    Receive a commanded motion, which will move the robot accordingly.
    """
    global veh_pose_true
    # TODO Perturb with some noise.
    # Clamp the commands at the allowed motion in a single timestep.
    fwd_dist = clamp(msg.x, 0, 0.1) # meters forward
    dtheta = clamp(msg.z, -0.0546, 0.0546) # radians CCW
    veh_pose_true[0] += fwd_dist * cos(veh_pose_true[2])
    veh_pose_true[1] += fwd_dist * sin(veh_pose_true[2])
    # Clamp the vehicle pose to remain inside the map bounds.
    veh_pose_true[0] = clamp(veh_pose_true[0], g_map_x_min_meters, g_map_x_max_meters)
    veh_pose_true[1] = clamp(veh_pose_true[1], g_map_y_min_meters, g_map_y_max_meters)
    # Keep yaw normalized to (-pi, pi).
    veh_pose_true[2] = remainder(veh_pose_true[2] + dtheta, tau)

    print("Veh pose is now " + str(veh_pose_true))

    # Generate an observation for the new vehicle pose.
    generate_observation()


def get_occ_map(msg):
    """
    Get the processed occupancy grid map to use as the "ground truth" map.
    """
    global occ_map_true, occ_map_true_padded
    # Convert from ROS Image message to an OpenCV image.
    occ_map_true = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    # Set the map bounds in meters.
    global g_map_x_min_meters, g_map_y_min_meters, g_map_x_max_meters, g_map_y_max_meters
    g_map_x_min_meters, g_map_y_min_meters = transform_map_px_to_m(occ_map_true.shape[1]-1, 0)
    g_map_x_max_meters, g_map_y_max_meters = transform_map_px_to_m(0, occ_map_true.shape[0]-1)
    print("Setting vehicle bounds ({:}, {:}), ({:}, {:})".format(g_map_x_min_meters, g_map_x_max_meters, g_map_y_min_meters, g_map_y_max_meters))

    # # Check all values appearing in the map.
    # vals_in_map = set()
    # for i in range(occ_map_true.shape[0]):
    #     for j in range(occ_map_true.shape[1]):
    #         vals_in_map.add(occ_map_true[i,j])
    # print("Values appearing in the map: {:}".format(vals_in_map))

    # When generating an observation, it is possible the desired region will be partially outside the bounds of the map.
    # To prevent potential errors, create a padded version of the map with enough extra rows/cols to ensure this won't happen.
    # Expand all dimensions by the diagonal of the observation area to cover all possible situations.
    # All extra space will be assumed to be occluded cells (value = 0.0).
    max_obs_dim = ceil(np.sqrt(g_obs_height_px_on_map**2 + g_obs_width_px_on_map**2))
    occ_map_true = cv2.copyMakeBorder(occ_map_true, max_obs_dim, max_obs_dim, max_obs_dim, max_obs_dim, cv2.BORDER_CONSTANT, None, 0.0)

    # Add the full map to our visualization.
    ax0.imshow(occ_map_true, cmap="gray", vmin=0, vmax=1)

    # NOTE this architecture forms a cycle, observation -> localization -> command, so to complete it we will generate a new observation upon receiving a command.
    # To kick-start this cycle, we will wait until the map has been processed, and then assume we've just received a zero command.
    generate_observation()


def get_localization_est(msg):
    """
    Get localization estimate from the particle filter, and save it to be displayed on the viz.
    """
    global veh_pose_est
    # Yaw is always estimated globally in radians.
    veh_pose_est[0] = msg.x
    veh_pose_est[1] = msg.y
    veh_pose_est[2] = msg.z


############################ SIMULATOR FUNCTIONS ####################################
def generate_observation():
    """
    Use the map and known ground-truth robot pose to generate the best possible observation.
    """
    global last_obs_img
    # Compute vehicle pose in pixels.
    veh_row, veh_col = transform_map_m_to_px(veh_pose_true[0], veh_pose_true[1])

    # Add the new (ground truth) vehicle pose to the viz.
    remove_plot("veh_pose_true")
    plots["veh_pose_true"] = ax0.arrow(veh_col, veh_row, 0.5*cos(veh_pose_true[2]), -0.5*sin(veh_pose_true[2]), color="blue", width=1.0)

    # Add the most recent localization estimate to the viz.
    veh_row_est, veh_col_est = transform_map_m_to_px(veh_pose_est[0], veh_pose_est[1])
    remove_plot("veh_pose_est")
    plots["veh_pose_est"] = ax0.arrow(veh_col_est, veh_row_est, 0.5*cos(veh_pose_est[2]), -0.5*sin(veh_pose_est[2]), color="green", width=1.0, zorder = 3)

    # Project ahead of vehicle pose to determine center.
    center_col = veh_col + (g_obs_height_px_on_map / 2 - g_veh_px_vert_from_bottom_on_map) * cos(veh_pose_true[2])
    center_row = veh_row - (g_obs_height_px_on_map / 2 - g_veh_px_vert_from_bottom_on_map) * sin(veh_pose_true[2])
    center = (center_col, center_row)
    # Create the rotated rectangle.
    angle = -np.rad2deg(veh_pose_true[2])
    rect = (center, (g_obs_height_px_on_map, g_obs_width_px_on_map), angle)

    # Plot the bounding box on the base map.
    box = cv2.boxPoints(rect)
    box_x_coords = [box[i,0] for i in range(box.shape[0])] + [box[0,0]]
    box_y_coords = [box[i,1] for i in range(box.shape[0])] + [box[0,1]]
    remove_plot("obs_bounding_box")
    plots["obs_bounding_box"] = ax0.plot(box_x_coords, box_y_coords, "r-", zorder=2)

    # Crop out the rotated rectangle and reorient it.
    obs_img = crop_rotated_rectangle(image = occ_map_true, rect = rect)

    if obs_img is None:
        # Rotated rect could not be cropped since it's partially outside the bounds of the image.
        print("Observation region could not be cropped out, since it's partially outside the map bounds.")
        obs_img = last_obs_img

    # Resize observation to desired resolution.
    obs_img = cv2.resize(obs_img, (g_obs_height_px, g_obs_width_px))

    # Publish this observation for the localization node to use.
    observation_pub.publish(bridge.cv2_to_imgmsg(obs_img, encoding="passthrough"))
    # Save last observation to keep using in case we've gone out of bounds.
    # TODO do something to handle vehicle going near the edges.
    last_obs_img = obs_img

    # Update the plot.
    remove_plot("obs_img")
    plots["obs_img"] = ax1.imshow(obs_img, cmap="gray", vmin=0, vmax=1)
    # Add vehicle pose relative to observation region for clarity.
    # NOTE since it's plotted sideways, robot pose is on the left side.
    if "veh_pose_obs" not in plots.keys():
        plots["veh_pose_obs"] = ax1.arrow(g_veh_px_vert_from_bottom_on_obs, g_obs_width_px // 2 + g_veh_px_horz_from_center_on_obs, 0.5, 0.0, color="blue", width=1.0, zorder = 2)

    plt.draw()
    plt.pause(g_dt)


def main():
    global observation_pub
    rospy.init_node('simulation_node')

    read_params()

    # make live plot bigger.
    plt.rcParams["figure.figsize"] = (9,9)

    # Subscribe to occupancy grid map to use as ground-truth.
    rospy.Subscriber(g_topic_occ_map, Image, get_occ_map, queue_size=1)
    # Subscribe to commanded motion.
    rospy.Subscriber(g_topic_commands, Vector3, get_command, queue_size=1)

    # Subscribe to localization est (for viz only).
    rospy.Subscriber(g_topic_localization, Vector3, get_localization_est, queue_size=1)

    # Publish ground-truth observation
    observation_pub = rospy.Publisher(g_topic_observations, Image, queue_size=1)

    # startup the plot.
    global fig, ax0, ax1
    fig = plt.figure(figsize=(8, 6)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    plt.title("Ground Truth Map & Vehicle Pose")
    plt.axis("off")
    ax1 = plt.subplot(gs[1])
    plt.title("Observation")
    # set constant plot params.
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    # allow clicking on the plot to do things (like kill the node).
    plt.connect('button_press_event', on_click)
    plt.show()

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass