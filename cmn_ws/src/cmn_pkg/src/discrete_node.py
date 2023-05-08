#!/usr/bin/env python3

"""
Node to mimic the discrete state space and action space used in the original Habitat simulation.
This node will handle all necessary steps that would otherwise be done by the rest of the nodes together.
"""

import rospy, sys
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import rospkg, yaml
import numpy as np
import cv2
from cv_bridge import CvBridge
from random import random
from math import pi, radians
from time import time
from random import randint, random

from scripts.cmn_utilities import clamp, ObservationGenerator
from scripts.astar import Astar
from scripts.pure_pursuit import PurePursuit

############ GLOBAL VARIABLES ###################
bridge = CvBridge()
obs_gen = ObservationGenerator()
astar = Astar()
# RealSense measurements buffer.
most_recent_measurement = None
#################################################

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
        global g_debug_mode, g_do_path_planning
        g_debug_mode = config["test"]["run_debug_mode"]
        g_do_path_planning = config["path_planning"]["do_path_planning"]
        PurePursuit.use_finite_lookahead_dist = g_do_path_planning
        # Rostopics.
        global g_topic_measurements, g_topic_commands, g_topic_localization, g_topic_occ_map, g_topic_planned_path, g_topic_goal
        g_topic_measurements = config["topics"]["measurements"]
        g_topic_occ_map = config["topics"]["occ_map"]
        g_topic_localization = config["topics"]["localization"]
        g_topic_goal = config["topics"]["goal"]
        g_topic_commands = config["topics"]["commands"]
        g_topic_planned_path = config["topics"]["planned_path"]
        # Constraints.
        global g_max_fwd_cmd, g_max_ang_cmd
        g_max_fwd_cmd = config["constraints"]["fwd"]
        g_max_ang_cmd = config["constraints"]["ang"]
        # In motion test mode, only this node will run, so it will handle the timer.
        global g_dt
        g_dt = config["dt"]
        # Params for discrete actions.
        global g_use_discrete_actions, g_discrete_forward_dist, g_discrete_forward_skip_probability, g_test_motion_type
        g_use_discrete_actions = config["actions"]["use_discrete_actions"]
        g_test_motion_type = "random_discrete" if g_use_discrete_actions else "circle"
        rospy.logwarn("Set g_test_motion_type: {:}".format(g_test_motion_type))
        g_discrete_forward_dist = abs(config["actions"]["discrete_forward_dist"])
        g_discrete_forward_skip_probability = config["actions"]["discrete_forward_skip_probability"]
        if g_discrete_forward_skip_probability < 0.0 or g_discrete_forward_skip_probability > 1.0:
            rospy.logwarn("DSC: Invalid value of discrete_forward_skip_probability. Must lie in range [0, 1]. Setting to 0.")
            g_discrete_forward_skip_probability = 0

def pub_velocity_cmd(fwd, ang):
    """
    Clamp a velocity command within valid values, and publish it to the vehicle.
    """
    # Clamp to allowed velocity ranges.
    fwd = clamp(fwd, 0, g_max_fwd_cmd)
    ang = clamp(ang, -g_max_ang_cmd, g_max_ang_cmd)
    rospy.loginfo("DSC: Publishing a command ({:}, {:})".format(fwd, ang))
    # Create ROS message.
    msg = Twist(Vector3(fwd, 0, 0), Vector3(0, 0, ang))
    cmd_pub.publish(msg)

######################## CALLBACKS ########################
def run_loop(event=None):
    """
    Main run loop.
    """
    # Get a panoramic measurement. Since the robot has only a forward-facing camera, we must pivot in-place four times.
    pano_meas = {}
    pano_meas["front"] = pop_from_RS_buffer()
    cmd_discrete_ang_motion(radians(90)) # pivot in-place 90 deg CCW, and then stop.
    pano_meas["left"] = pop_from_RS_buffer()
    cmd_discrete_ang_motion(radians(90))
    pano_meas["back"] = pop_from_RS_buffer()
    cmd_discrete_ang_motion(radians(90))
    pano_meas["right"] = pop_from_RS_buffer()
    cmd_discrete_ang_motion(radians(90))
    # Vehicle should now be facing forwards again (its original direction).

    # Pass this panoramic measurement through the model to obtain an observation.
    # TODO

    # Use discrete bayesian filter to localize the robot.
    # TODO
    robot_pose_estimate = None

    # Determine path from estimated pose to the goal, using the coarse map.
    # TODO

    # Command the next action, and wait for it to finish.
    # TODO Possible discrete motions that can be commanded.
    discrete_motion_types = ["90_LEFT", "90_RIGHT", "FORWARD"]
    action = discrete_motion_types[randint(0,len(discrete_motion_types)-1)]
    if action == "90_LEFT":
        cmd_discrete_ang_motion(radians(90))
    elif action == "90_RIGHT":
        cmd_discrete_ang_motion(radians(-90))
    elif action == "FORWARD":
        if random() > g_discrete_forward_skip_probability:
            cmd_discrete_fwd_motion(g_discrete_forward_dist)

    # Proceed to the next iteration, where another measurement will be taken.

def pop_from_RS_buffer():
    """
    Wait for a new RealSense measurement to be available, and return it.
    """
    global most_recent_measurement
    while most_recent_measurement is None:
        rospy.sleep(0.01)
    # Convert from ROS Image message to an OpenCV image.
    cv_img_meas = bridge.imgmsg_to_cv2(most_recent_measurement, desired_encoding='passthrough')
    # Ensure this same measurement will not be used again.
    most_recent_measurement = None
    return cv_img_meas

def get_RS_image(msg):
    """
    Get a measurement Image from the RealSense camera.
    Could be changed multiple times before we need a measurement, so this allows skipping measurements to prefer recency.
    """
    global most_recent_measurement
    most_recent_measurement = msg
    
def get_map(msg:Image):
    """
    Get the global occupancy map to use for path planning.
    NOTE Map was already processed into an occupancy grid before being sent.
    """
    rospy.loginfo("DSC: Got occupancy map.")
    # Convert from ROS Image message to an OpenCV image.
    occ_map = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    obs_gen.set_map(occ_map)
    astar.map = obs_gen.map

################ PATH PLANNING FUNCTIONS #####################
def cmd_discrete_ang_motion(angle:float):
    """
    Turn the robot in-place by a discrete amount, and then stop.
    @param angle - the angle to turn (radians). Positive for CCW, negative for CW.
    """
    # Determine the amount of time it will take to turn this amount at the max turn speed.
    time_to_turn = abs(angle / g_max_ang_cmd) # rad / (rad/s) = seconds.
    rospy.logwarn("DSC: Determined time_to_move: {:} seconds.".format(time_to_turn))
    start_time = time()
    # Get turn direction.
    turn_dir_sign = angle / abs(angle)
    # Send the command velocity for the duration.
    while time() - start_time < time_to_turn:
        rospy.logwarn("DSC: Elapsed time: {:} seconds.".format(time() - start_time))
        pub_velocity_cmd(0, g_max_ang_cmd * turn_dir_sign)
        rospy.sleep(0.001)
    # When the time has elapsed, send a command to stop.
    pub_velocity_cmd(0, 0)


def cmd_discrete_fwd_motion(dist:float):
    """
    Move the robot forwards by a discrete distance, and then stop.
    @param dist - distance in meters to move. Positive is forwards, negative for backwards.
    """
    # Determine the amount of time it will take to move the specified distance.
    time_to_move = abs(dist / g_max_fwd_cmd) # rad / (rad/s) = seconds.
    rospy.logwarn("DSC: Determined time_to_move: {:} seconds.".format(time_to_move))
    start_time = time()
    # Get direction of motion.
    motion_sign = dist / abs(dist)
    # Send the command velocity for the duration.
    while time() - start_time < time_to_move:
        rospy.logwarn("DSC: Elapsed time: {:} seconds.".format(time() - start_time))
        pub_velocity_cmd(g_max_fwd_cmd * motion_sign, 0)
        rospy.sleep(0.1)
    # When the time has elapsed, send a command to stop.
    pub_velocity_cmd(0, 0)


# Possible discrete motions that can be commanded.
discrete_motion_types = ["90_LEFT", "90_RIGHT", "FORWARD"]
def random_discrete_action():
    """
    Choose a random action from our list of possible discrete actions, and command it.
    """
    action = discrete_motion_types[randint(0,len(discrete_motion_types)-1)]
    if action == "90_LEFT":
        cmd_discrete_ang_motion(radians(90))
    elif action == "90_RIGHT":
        cmd_discrete_ang_motion(radians(-90))
    elif action == "FORWARD":
        if random() > g_discrete_forward_skip_probability:
            cmd_discrete_fwd_motion(g_discrete_forward_dist)



def main():
    global cmd_pub, path_pub
    rospy.init_node('discrete_node')

    read_params()

    # Read command line args.
    if len(sys.argv) > 1:
        global g_run_test_motion
        g_run_test_motion = sys.argv[1].lower() == "true"


    # Subscribe to sensor images from RealSense.
    # TODO may want to check /locobot/camera/color/camera_info
    rospy.Subscriber(g_topic_measurements, Image, get_RS_image, queue_size=1)

    # Subscribe to pre-processed map.
    rospy.Subscriber(g_topic_occ_map, Image, get_map, queue_size=1)

    # Publish control commands (velocities in m/s and rad/s).
    cmd_pub = rospy.Publisher(g_topic_commands, Twist, queue_size=1)

    # Publish planned path to the goal (for viz).
    path_pub = rospy.Publisher(g_topic_planned_path, Float32MultiArray, queue_size=1)

    rospy.Timer(rospy.Duration(g_dt), run_loop)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass