#!/usr/bin/env python3

"""
Main node for running the project. This should be run on the locobot itself.
"""

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Bool
import rospkg, yaml, sys, os
from cv_bridge import CvBridge
from math import pi, atan2, asin
import numpy as np
import cv2
from time import strftime, time

from scripts.cmn_interface import CoarseMapNavInterface, CmnConfig
from scripts.basic_types import PoseMeters

############ GLOBAL VARIABLES ###################
g_cv_bridge = CvBridge()

g_cmn_interface:CoarseMapNavInterface = None

# RealSense measurements buffer.
g_most_recent_realsense_measurement = None
g_desired_meas_shape = None # Shape (h, w, c) to resize each color image from RS to.
# Odom measurements.
g_first_odom:PoseMeters = None # Used to offset odom frame to always have origin at start pose.
# Configs.
g_run_modes = ["continuous", "discrete", "discrete_random"] # Allowed/supported run modes.
g_run_mode = None # "discrete" or "continuous"
g_use_ground_truth_map_to_generate_observations = False
g_show_live_viz = False
g_verbose = False
# Data saving params.
g_save_training_data:bool = False # Flag to save data when running on robot for later training/evaluation.
g_training_data_dirpath:str = None # Location of directory to save data to.
# Live flags.
g_viz_paused = False

# Publish visualizer images so we can view them with rqt without disrupting the run loop.
g_pub_viz_images:bool = False
g_sim_viz_pub = None
g_cmn_viz_pub = None
#################################################

def timer_update_loop(event=None):
    # Update the visualization, if enabled.
    if g_cmn_interface.visualizer is not None:
        # Simulator viz.
        sim_viz_img = None
        if g_use_ground_truth_map_to_generate_observations:
            sim_viz_img = g_cmn_interface.visualizer.get_updated_img()
        # CMN viz.
        cmn_viz_img = None
        if g_cmn_interface.cmn_node is not None and g_cmn_interface.cmn_node.visualizer is not None:
            cmn_viz_img = g_cmn_interface.cmn_node.visualizer.get_updated_img()
            
        if g_pub_viz_images:
            # Publish to rostopics for some external viz to use.
            if sim_viz_img is not None:
                g_sim_viz_pub.publish(g_cv_bridge.cv2_to_imgmsg(sim_viz_img))
            if cmn_viz_img is not None:
                g_cmn_viz_pub.publish(g_cv_bridge.cv2_to_imgmsg(cmn_viz_img))
        else:
            # Manage the viz here.
            if sim_viz_img is not None:
                cv2.imshow('viz image', sim_viz_img)
            if cmn_viz_img is not None:
                cv2.imshow('cmn viz image', cmn_viz_img)
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
    if g_cmn_interface.pano_rgb is not None:
        pano_rgb = g_cmn_interface.pano_rgb
    elif not g_use_ground_truth_map_to_generate_observations:
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
        global g_verbose, g_dt, g_enable_localization, g_enable_ml_model
        g_verbose = config["verbose"]
        g_dt = config["dt"]
        g_enable_localization = config["particle_filter"]["enable"]
        g_enable_ml_model = not config["model"]["skip_loading"]
        # Settings for interfacing with CMN.
        global g_meas_topic, g_desired_meas_shape
        g_meas_topic = config["measurements"]["topic"]
        g_desired_meas_shape = (config["measurements"]["height"], config["measurements"]["width"])
        # Settings for saving data for later training/evaluation.
        global g_save_training_data, g_training_data_dirpath
        g_save_training_data = config["save_data_for_training"]
        if g_save_training_data:
            g_training_data_dirpath = config["training_data_dirpath"]
            if g_training_data_dirpath[0] != "/":
                # Make path relative to cmn_pkg directory.
                g_training_data_dirpath = os.path.join(pkg_path, g_training_data_dirpath)
            # Append datetime and create data directory.
            g_training_data_dirpath = os.path.join(g_training_data_dirpath, strftime("%Y%m%d-%H%M%S"))
            os.makedirs(g_training_data_dirpath, exist_ok=True)


def set_global_params(run_mode:str, use_sim:bool, use_viz:bool, cmd_vel_pub=None):
    """
    Set global params specified by the launch file/runner.
    @param run_mode - Mode to run the project in.
    @param use_sim - Flag to use the simulator instead of requiring robot sensor data.
    @param use_viz - Flag to show the live visualization. Only possible on host PC.
    @param cmd_vel_pub (optional) - ROS publisher for command velocities.
    """
    # Set the global params.
    global g_run_mode, g_use_ground_truth_map_to_generate_observations, g_show_live_viz
    g_run_mode = run_mode
    g_use_ground_truth_map_to_generate_observations = use_sim
    g_show_live_viz = use_viz

    # Setup configs for CMN interface.
    config = CmnConfig()
    config.run_mode = run_mode
    config.enable_sim = use_sim
    config.enable_viz = use_viz
    config.enable_ml_model = g_enable_ml_model
    config.enable_localization = g_enable_localization

    # Init the main (non-ROS-specific) part of the project.
    global g_cmn_interface
    g_cmn_interface = CoarseMapNavInterface(config, cmd_vel_pub)

    # Set data saving params.
    g_cmn_interface.save_training_data = g_save_training_data
    g_cmn_interface.training_data_dirpath = g_training_data_dirpath


######################## CALLBACKS ########################
def get_pano_meas():
    """
    Get a panoramic measurement.
    Since the robot has only a forward-facing camera, we must pivot in-place four times.
    @return panoramic image created by concatenating four individual measurements.
    """
    if g_verbose:
        rospy.loginfo("Attempting to generate a panoramic measurement by commanding four 90 degree pivots.")
    pano_meas_front = pop_from_RS_buffer()
    # Pivot in-place 90 deg CW to get another measurement.
    g_cmn_interface.motion_planner.cmd_discrete_action("turn_right")
    pano_meas_right = pop_from_RS_buffer()
    g_cmn_interface.motion_planner.cmd_discrete_action("turn_right")
    pano_meas_back = pop_from_RS_buffer()
    g_cmn_interface.motion_planner.cmd_discrete_action("turn_right")
    pano_meas_left = pop_from_RS_buffer()
    g_cmn_interface.motion_planner.cmd_discrete_action("turn_right")
    # Vehicle should now be facing forwards again (its original direction).
    # Combine these images into a panorama.
    pano_rgb = np.concatenate([pano_meas_front[:, :, 0:3],
                               pano_meas_right[:, :, 0:3],
                               pano_meas_back[:, :, 0:3],
                               pano_meas_left[:, :, 0:3]], axis=1)
    # Convert from RGB to BGR so OpenCV will show/save it properly.
    # TODO determine if this should be done for the model input or not.
    pano_rgb = cv2.cvtColor(pano_rgb, cv2.COLOR_RGB2BGR)
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

    # Resize the image to the size expected by CMN.
    if g_verbose:
        # cv2.imshow("color image", cv_img_meas); cv2.waitKey(0); cv2.destroyAllWindows()
        rospy.loginfo("Trying to resize image from shape {:} to {:}".format(cv_img_meas.shape, g_desired_meas_shape))
    cv_img_meas = cv2.resize(cv_img_meas, g_desired_meas_shape)

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
    # Create a pose object.
    odom_pose = PoseMeters(x, y, roll)

    # Since resetting the locobot's odom doesn't seem to work, just save the first odom and use it to offset all future measurements.
    global g_first_odom
    if g_first_odom is None:
        # This is the first measurement received. Use it as the origin for all future measurements.
        g_first_odom = odom_pose
    else:
        odom_pose.make_relative(g_first_odom)

    # Set this odom in our code.
    g_cmn_interface.set_new_odom(odom_pose)
    
    if g_verbose:
        rospy.loginfo("Got odom: {:}".format(odom_pose))

def main():
    rospy.init_node('runner_node')

    read_params()

    # # Reset the robot odometry. NOTE this doesn't seem to work.
    # odom_reset_pub = rospy.Publisher("/locobot/mobile_base/commands/reset_odometry", Empty, queue_size=10)
    # # NOTE: odom reset messages take some time to actually get through, so keep publishing for a duration.
    # odom_reset_pub_duration = 0.25 # seconds.
    # timer = time()
    # while time() - timer < odom_reset_pub_duration:
    #     odom_reset_pub.publish(Empty())

    # Publish control commands (velocities in m/s and rad/s).
    cmd_vel_pub = rospy.Publisher("/locobot/mobile_base/commands/velocity", Twist, queue_size=1)

    # Get any params specified in args from launch file.
    if len(sys.argv) > 3:
        set_global_params(sys.argv[1], sys.argv[2].lower() == "true", sys.argv[3].lower() == "true", cmd_vel_pub)
    else:
        print("Missing required arguments.")
        exit()

    if g_run_mode not in g_run_modes:
        rospy.logerr("Invalid run_mode {:}. Exiting.".format(g_run_mode))
        exit()


    # Subscribe to sensor images from RealSense.
    # TODO may want to check /locobot/camera/color/camera_info
    rospy.Subscriber(g_meas_topic, Image, get_RS_image, queue_size=1)

    # Subscribe to robot odometry.
    rospy.Subscriber("/locobot/mobile_base/odom", Odometry, get_odom, queue_size=1)

    # Publish viz images so we can view them in rqt without messing up the run loop.
    global g_sim_viz_pub, g_cmn_viz_pub
    g_sim_viz_pub = rospy.Publisher("/cmn/viz/sim", Image, queue_size=1)
    g_cmn_viz_pub = rospy.Publisher("/cmn/viz/cmn", Image, queue_size=1)

    rospy.Timer(rospy.Duration(g_dt), timer_update_loop)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass