#!/usr/bin/env python3

"""
Main node for running the project. This should be run on the locobot itself.
"""

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import rospkg, yaml, sys, os
from cv_bridge import CvBridge
from math import pi, atan2, asin
import numpy as np

# from scripts.map_handler import Simulator
# from scripts.motion_planner import DiscreteMotionPlanner
# from scripts.particle_filter import ParticleFilter
# from scripts.visualizer import Visualizer
from scripts.cmn_interface import CoarseMapNavInterface
import cv2

############ GLOBAL VARIABLES ###################
g_cv_bridge = CvBridge()

g_cmn_interface:CoarseMapNavInterface = None

# RealSense measurements buffer.
g_most_recent_realsense_measurement = None
# Configs.
g_run_mode = None # "discrete" or "continuous"
g_use_ground_truth_map_to_generate_observations = False
g_show_live_viz = False
g_verbose = False
# Live flags.
g_viz_paused = False
#################################################

def timer_update_loop(event=None):
    # Update the visualization, if enabled.
    if g_cmn_interface.visualizer is not None:
        viz_img = g_cmn_interface.visualizer.get_updated_img()
        cv2.imshow('viz image', viz_img)
        key = cv2.waitKey(int(g_dt * 1000))
        # Special keypress conditions.
        if key == 113: # q for quit.
            cv2.destroyAllWindows()
            rospy.signal_shutdown("User pressed Q key.")
            exit()
        elif key == 32: # spacebar.
            global g_viz_paused
            g_viz_paused = not g_viz_paused

        if g_viz_paused:
            # Skip all operations, so the same viz image will just keep being displayed until unpaused.
            return
        
    # Only gather a pano RGB if needed.
    pano_rgb = None
    if not g_use_ground_truth_map_to_generate_observations:
        pano_rgb = get_pano_meas()

    # Run an iteration. (It will internally run either continuous or discrete case).
    g_cmn_interface.run(pano_rgb, g_dt)


# TODO make intermediary control_node that receives our commanded motion and either passes it through to the robot or uses sensors to perform reactive obstacle avoidance

##################### UTILITY FUNCTIONS #######################
def read_params():
    """
    Read configuration params from the yaml.
    """
    # Determine filepath.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('cmn_pkg')
    global g_yaml_path
    g_yaml_path = os.path.join(pkg_path, 'config/config.yaml')
    # Open the yaml and get the relevant params.
    with open(g_yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        global g_verbose, g_dt, g_enable_localization
        g_verbose = config["verbose"]
        g_dt = config["dt"]
        g_enable_localization = config["particle_filter"]["enable"]
        # Settings for interfacing with CMN.
        global g_meas_topic, g_meas_width, g_meas_height
        g_meas_topic = config["measurements"]["topic"]
        g_meas_height = config["measurements"]["height"]
        g_meas_width = config["measurements"]["width"]

######################## CALLBACKS ########################
def get_pano_meas():
    """
    Get a panoramic measurement.
    Since the robot has only a forward-facing camera, we must pivot in-place four times.
    @return panoramic image created by concatenating four individual measurements.
    """
    if g_verbose:
        rospy.loginfo("Attempting to generate a panoramic measurement by commanding four 90 degree pivots.")
    pano_meas = {}
    pano_meas["color_sensor_front"] = pop_from_RS_buffer()
    # Resize to desired shape for input to CMN code.
    if g_verbose:
        rospy.loginfo("Raw RS image has shape {:}".format(pano_meas["color_sensor_front"].shape))
    pano_meas["color_sensor_front"] = cv2.resize(pano_meas["color_sensor_front"], (g_meas_height,g_meas_width,3))
    # Pivot in-place 90 deg CW to get another measurement.
    g_cmn_interface.motion_planner.cmd_discrete_action("turn_right")
    pano_meas["color_sensor_right"] = pop_from_RS_buffer()
    pano_meas["color_sensor_right"] = cv2.resize(pano_meas["color_sensor_right"], (g_meas_height,g_meas_width,3))
    g_cmn_interface.motion_planner.cmd_discrete_action("turn_right")
    pano_meas["color_sensor_back"] = pop_from_RS_buffer()
    pano_meas["color_sensor_back"] = cv2.resize(pano_meas["color_sensor_back"], (g_meas_height,g_meas_width,3))
    g_cmn_interface.motion_planner.cmd_discrete_action("turn_right")
    pano_meas["color_sensor_left"] = pop_from_RS_buffer()
    pano_meas["color_sensor_left"] = cv2.resize(pano_meas["color_sensor_left"], (g_meas_height,g_meas_width,3))
    g_cmn_interface.motion_planner.cmd_discrete_action("turn_right")
    # Vehicle should now be facing forwards again (its original direction).
    # Combine these images into a panorama.
    pano_rgb = np.concatenate([pano_meas['color_sensor_front'][:, :, 0:3],
                               pano_meas['color_sensor_right'][:, :, 0:3],
                               pano_meas['color_sensor_back'][:, :, 0:3],
                               pano_meas['color_sensor_left'][:, :, 0:3]], axis=1)
    return pano_rgb

def pop_from_RS_buffer():
    """
    Wait for a new RealSense measurement to be available, and return it.
    """
    global g_most_recent_realsense_measurement
    while g_most_recent_realsense_measurement is None:
        rospy.logwarn("Waiting on measurement from RealSense!")
        rospy.sleep(0.01)
    # Convert from ROS Image message to an OpenCV image.
    cv_img_meas = g_cv_bridge.imgmsg_to_cv2(g_most_recent_realsense_measurement, desired_encoding='passthrough')
    # Ensure this same measurement will not be used again.
    g_most_recent_realsense_measurement = None
    return cv_img_meas

def get_RS_image(msg):
    """
    Get a measurement Image from the RealSense camera.
    Could be changed multiple times before we need a measurement, so this allows skipping measurements to prefer recency.
    """
    global g_most_recent_realsense_measurement
    g_most_recent_realsense_measurement = msg

def get_odom(msg):
    """
    Get an odometry message from the robot's mobile base.
    Parse the message to extract the desired position and orientation information.
    """
    # Extract x,y position.
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    # Extract orientation from quaternion.
    # NOTE our "yaw" is the "roll" from https://stackoverflow.com/a/18115837/14783583
    q = msg.pose.pose.orientation
    # yaw = atan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)
    # pitch = asin(-2.0*(q.x*q.z - q.w*q.y))
    roll = atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
    # Set the odom.
    g_cmn_interface.set_new_odom(x, y, roll)
    
    if g_verbose:
        rospy.loginfo("Got odom {:}.".format((x, y, roll)))

def main():
    rospy.init_node('runner_node')

    read_params()

    # Get any params specified in args from launch file.
    if len(sys.argv) > 3:
        global g_run_mode, g_use_ground_truth_map_to_generate_observations, g_show_live_viz
        g_run_mode = sys.argv[1]
        g_use_ground_truth_map_to_generate_observations = sys.argv[2].lower() == "true"
        g_show_live_viz = sys.argv[3].lower() == "true"
    

    if g_run_mode not in ["continuous", "discrete"]:
        rospy.logerr("Invalid run_mode {:}. Exiting.".format(g_run_mode))
        exit()


    # Subscribe to sensor images from RealSense.
    # TODO may want to check /locobot/camera/color/camera_info
    rospy.Subscriber(g_meas_topic, Image, get_RS_image, queue_size=1)

    # Subscribe to robot odometry.
    rospy.Subscriber("/locobot/mobile_base/odom", Odometry, get_odom, queue_size=1)

    # Publish control commands (velocities in m/s and rad/s).
    cmd_vel_pub = rospy.Publisher("/locobot/mobile_base/commands/velocity", Twist, queue_size=1)

    # Init the main (non-ROS-specific) part of the project.
    global g_cmn_interface
    g_cmn_interface = CoarseMapNavInterface(g_use_ground_truth_map_to_generate_observations, g_run_mode == "discrete", g_show_live_viz, cmd_vel_pub, g_yaml_path, g_enable_localization)

    rospy.Timer(rospy.Duration(g_dt), timer_update_loop)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass