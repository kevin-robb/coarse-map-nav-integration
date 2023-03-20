#!/usr/bin/env python3

"""
Node to handle ROS interface for getting localization estimate, global map, and performing path planning, navigation, and publishing a control command to the turtlebot.
"""

import rospy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
import rospkg, yaml
import numpy as np
import cv2
from cv_bridge import CvBridge
from random import random
from math import pi

from scripts.cmn_utilities import clamp

############ GLOBAL VARIABLES ###################
bridge = CvBridge()
cmd_pub = None
occ_map = None # global occupancy grid map
# Temporary stuff to test that commanding motion works.
cfg_dt = 3 # timer period for test commands (seconds).
cfg_test_spd_range = (-1, 1)
cfg_test_ang_range = (-pi/2, pi/2)
#################################################


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
        global g_debug_mode
        g_debug_mode = config["test"]["run_debug_mode"]
        # Rostopics.
        global g_topic_commands, g_topic_localization, g_topic_observations, g_topic_occ_map
        g_topic_occ_map = config["topics"]["occ_map"]
        g_topic_localization = config["topics"]["localization"]
        g_topic_commands = config["topics"]["commands"]
        # Constraints.
        global g_max_fwd_cmd, g_max_ang_cmd
        g_max_fwd_cmd = config["constraints"]["fwd"]
        g_max_ang_cmd = config["constraints"]["ang"]


# TODO do path planning.
# TODO do navigation & obstacle avoidance.
# TODO create control commands & pub them.


def get_localization_est(msg):
    """
    Get localization estimate from the particle filter.
    """
    # TODO process it and associate with a particular cell/orientation on the map.
    print("Got localization estimate {:}".format(msg))
    
    # DEBUG send a simple motion command.    
    publish_command(0.02, pi) # drive in a small circle.
    # publish_command(0.5, 0.0) # drive in a straight line.
    

def get_map(msg):
    """
    Get the global occupancy map to use for path planning.
    """
    global occ_map
    # Convert from ROS Image message to an OpenCV image.
    occ_map = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # NOTE Map was already processed into an occupancy grid before being sent.
    if g_debug_mode:
        print("map has shape {:}".format(occ_map.shape))
        cv2.imshow("Motion planning node received map", occ_map); cv2.waitKey(0); cv2.destroyAllWindows()


def publish_command(fwd, ang):
    """
    Clamp a command within valid values, and publish it to the vehicle/simulator.
    """
    fwd = clamp(fwd, 0, g_max_fwd_cmd)
    ang = clamp(ang, -g_max_ang_cmd, g_max_ang_cmd)
    # Create ROS message.
    # NOTE x-component = forward motion, z-component = angular motion. y-component = lateral motion, which is impossible for our system and is ignored.
    msg = Vector3(fwd, 0.0, ang)
    cmd_pub.publish(msg)


def main():
    global cmd_pub
    rospy.init_node('motion_planning_node')

    read_params()

    # Subscribe to localization est.
    rospy.Subscriber(g_topic_localization, Vector3, get_localization_est, queue_size=1)
    # Subscribe to (or just read the map from) file.
    rospy.Subscriber(g_topic_occ_map, Image, get_map, queue_size=1)

    # Publish control commands.
    cmd_pub = rospy.Publisher(g_topic_commands, Vector3, queue_size=1)
    # there is a way to command a relative position/yaw motion:
    # python navigation/base_position_control.py --base_planner none --base_controller ilqr --smooth --close_loop --relative_position 1.,1.,1.57 --botname locobot

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass