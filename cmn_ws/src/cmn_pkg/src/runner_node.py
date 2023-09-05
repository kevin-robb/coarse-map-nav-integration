#!/usr/bin/env python3

"""
Main node for running the project. This should be run on the locobot itself.
"""

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import rospkg, yaml, sys
from cv_bridge import CvBridge
from math import pi, atan2, asin
import numpy as np

from scripts.map_handler import Simulator
from scripts.motion_planner import DiscreteMotionPlanner
from scripts.particle_filter import ParticleFilter
from scripts.visualizer import Visualizer
import cv2

############ GLOBAL VARIABLES ###################
g_cv_bridge = CvBridge()
# Instances of utility classes defined in src/scripts folder.
g_simulator = Simulator() # Subset of MapFrameManager that will allow us to do coordinate transforms.
g_motion_planner = DiscreteMotionPlanner() # Subset of MotionPlanner that can be used to plan paths and command continuous or discrete motions.
g_particle_filter = ParticleFilter() # PF for continuous state-space localization.
g_visualizer = None # Will be initialized only if a launch file arg is true.
# RealSense measurements buffer.
g_most_recent_realsense_measurement = None
# Configs.
g_run_mode = None # "discrete" or "continuous"
g_use_ground_truth_map_to_generate_observations = False
g_show_live_viz = False
g_verbose = False
#################################################

def run_loop(event=None):
    """
    Choose which run loop to use.
    """
    if g_run_mode == "discrete":
        run_loop_discrete()
    elif g_run_mode == "continuous":
        run_loop_continuous()
    else:
        rospy.logerr("run_loop called with invalid run_mode {:}.".format(g_run_mode))

def run_loop_discrete(event=None):
    """
    Main run loop for discrete case.
    Commands motions on the robot to collect a panoramic measurement,
    uses the model to get an observation,
    localizes with a discrete bayesian filter,
    and commands discrete actions.
    """
    # # DEBUG just command a 90 degree turn, then stop.
    # g_motion_planner.cmd_discrete_action("90_LEFT")
    # rospy.sleep(2)
    # return

    if g_use_ground_truth_map_to_generate_observations:
        # TODO sim
        observation = None
    else:
        # Get a panoramic measurement.
        pano_rgb = get_pano_meas()
        # TODO Pass this panoramic measurement through the model to obtain an observation.
        observation = None

    # TODO Use discrete bayesian filter to localize the robot.
    robot_pose_estimate = None
    # TODO Determine path from estimated pose to the goal, using the coarse map, and determine a discrete action to command.
    # TODO for now just command random discrete action.
    # fwd, ang = g_motion_planner.cmd_random_discrete_action()
    # TODO for now, always go forwards.
    fwd, ang = g_motion_planner.cmd_discrete_action("FORWARD")

    if g_use_ground_truth_map_to_generate_observations:
        # Propagate the true vehicle pose by this discrete action.
        g_simulator.propagate_with_dist(fwd, ang)


def run_loop_continuous(event=None):
    """
    Main run loop for continuous case.
    Uses only the current RS measurement,
    generates an observation,
    localizes with a particle filter,
    and commands continuous velocities.
    """
    observation, rect = None, None
    if g_use_ground_truth_map_to_generate_observations:
        # Do not attempt to use the utilities class until the map has been processed.
        while not g_simulator.initialized:
            rospy.logwarn("Waiting for sim to be initialized!")
            rospy.sleep(0.1)
        observation, rect = g_simulator.get_true_observation()
    else:
        # Get an image from the RealSense.
        meas = pop_from_RS_buffer()
        # TODO Pass this measurement through the ML model to obtain an observation.
        observation = None # i.e., get_observation_from_meas(meas)

    # Use the particle filter to get a localization estimate from this observation.
    pf_estimate = g_particle_filter.update_with_observation(observation)

    if g_visualizer is not None:
        # Update data for the viz.
        g_visualizer.set_observation(observation, rect)
        # Convert meters to pixels using our map transform class.
        # g_visualizer.veh_pose_estimate = g_simulator.transform_pose_m_to_px(pf_estimate)
        # Update ground-truth data if we're running the sim.
        if g_use_ground_truth_map_to_generate_observations:
            g_visualizer.veh_pose_true = g_simulator.transform_pose_m_to_px(g_simulator.veh_pose_true)
        # Convert particle set to pixels as well.
        g_visualizer.particle_set = g_particle_filter.get_particle_set_px()

    # Run the PF resampling step.
    g_particle_filter.resample()

    # Choose velocity commands for the robot based on the pose estimate.
    fwd, ang = g_motion_planner.plan_path_to_goal(pf_estimate)
    g_motion_planner.pub_velocity_cmd(fwd, ang)
    g_simulator.propagate_with_vel(fwd, ang) # Apply to the ground truth vehicle pose.

    if g_visualizer is not None:
        # Update data for the viz.
        g_visualizer.planned_path = g_motion_planner.path_px_reversed

    # Propagate all particles by the commanded motion.
    g_particle_filter.propagate_particles(fwd * g_dt, ang * g_dt)

    if g_visualizer is not None:
        # Update the viz.
        viz_img = g_visualizer.get_updated_img()
        cv2.imshow('viz image', viz_img)
        key = cv2.waitKey(int(g_dt * 1000))
        # Special keypress conditions.
        if key == 113: # q for quit.
            cv2.destroyAllWindows()
            rospy.signal_shutdown("User pressed Q key.")
            exit()


# TODO make intermediary control_node that receives our commanded motion and either passes it through to the robot or uses sensors to perform reactive obstacle avoidance

##################### UTILITY FUNCTIONS #######################
def read_params():
    """
    Read configuration params from the yaml.
    """
    # Determine filepath.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('cmn_pkg')
    # Open the yaml and get the relevant params.
    with open(pkg_path+'/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        global g_verbose, g_dt
        g_verbose = config["verbose"]
        g_dt = config["dt"]
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
    g_motion_planner.cmd_discrete_action("90_RIGHT")
    pano_meas["color_sensor_right"] = pop_from_RS_buffer()
    pano_meas["color_sensor_right"] = cv2.resize(pano_meas["color_sensor_right"], (g_meas_height,g_meas_width,3))
    g_motion_planner.cmd_discrete_action("90_RIGHT")
    pano_meas["color_sensor_back"] = pop_from_RS_buffer()
    pano_meas["color_sensor_back"] = cv2.resize(pano_meas["color_sensor_back"], (g_meas_height,g_meas_width,3))
    g_motion_planner.cmd_discrete_action("90_RIGHT")
    pano_meas["color_sensor_left"] = pop_from_RS_buffer()
    pano_meas["color_sensor_left"] = cv2.resize(pano_meas["color_sensor_left"], (g_meas_height,g_meas_width,3))
    g_motion_planner.cmd_discrete_action("90_RIGHT")
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
    # Set the odom var.
    most_recent_odom = (x,y,roll)
    g_motion_planner.set_odom(most_recent_odom)
    
    if g_verbose:
        rospy.loginfo("Got odom {:}.".format(most_recent_odom))

def main():
    rospy.init_node('runner_node')

    read_params()

    # Get any params specified in args from launch file.
    if len(sys.argv) > 3:
        global g_run_mode, g_use_ground_truth_map_to_generate_observations, g_show_live_viz
        g_run_mode = sys.argv[1]
        g_use_ground_truth_map_to_generate_observations = sys.argv[2].lower() == "true"
        g_show_live_viz = sys.argv[3].lower() == "true"
        # Let other module(s) know what mode is active.
        global g_simulator
        g_simulator.use_discrete_state_space = g_run_mode == "discrete"

    # Init the visualizer only if it's enabled.
    if g_show_live_viz:
        global g_visualizer
        g_visualizer = Visualizer()

    # Subscribe to sensor images from RealSense.
    # TODO may want to check /locobot/camera/color/camera_info
    rospy.Subscriber(g_meas_topic, Image, get_RS_image, queue_size=1)

    # Subscribe to robot odometry.
    rospy.Subscriber("/locobot/mobile_base/odom", Odometry, get_odom, queue_size=1)

    # Publish control commands (velocities in m/s and rad/s).
    cmd_vel_pub = rospy.Publisher("/locobot/mobile_base/commands/velocity", Twist, queue_size=1)
    g_motion_planner.set_vel_pub(cmd_vel_pub)

    # Give reference to sim so other classes can use the map and perform coordinate transforms.
    g_motion_planner.set_map_frame_manager(g_simulator)
    g_particle_filter.set_map_frame_manager(g_simulator)
    if g_visualizer is not None:
        g_visualizer.set_map_frame_manager(g_simulator)
    # Select a random goal point.
    g_motion_planner.set_goal_point_random()
    if g_visualizer is not None:
        g_visualizer.goal_cell = g_motion_planner.goal_pos_px

    rospy.Timer(rospy.Duration(g_dt), run_loop)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass