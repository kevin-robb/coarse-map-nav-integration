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
    global cfg_debug_mode, topic_occ_map, topic_localization, topic_commands
    # Determine filepath.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('perception_pkg')
    # Open the yaml and get the relevant params.
    with open(pkg_path+'/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        cfg_debug_mode = config["test"]["run_debug_mode"]
        # Rostopics:
        topic_occ_map = config["topics"]["occ_map"]
        topic_localization = config["topics"]["localization"]
        topic_commands = config["topics"]["commands"]


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
    # NOTE x-component = forward motion, z-component = angular motion. y-component = lateral motion, which is impossible for our system and is ignored.
    # msg = Vector3(0.02, 0.0, pi) # drive in a small circle.
    msg = Vector3(0.02, 0.0, 0.0) # drive in a straight line.
    cmd_pub.publish(msg)


def generate_test_command(event):
    """
    Send a simple twist command to test communication between the ros nodes.
    """
    # msg = Twist()
    # msg.linear.x = random() * (cfg_test_spd_range[1] - cfg_test_spd_range[0]) + cfg_test_spd_range[0]
    # msg.angular.z = random() * (cfg_test_ang_range[1] - cfg_test_ang_range[0]) + cfg_test_ang_range[0]
    # print("Commanding linear: " + str(msg.linear.x) + ", angular: " + str(msg.angular.z))
    msg = Vector3()
    msg.x = 1
    msg.y = 0
    msg.z = pi/2
    cmd_pub.publish(msg)


def get_map(msg):
    """
    Get the global occupancy map to use for path planning.
    """
    global occ_map
    # Convert from ROS Image message to an OpenCV image.
    occ_map = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # NOTE Map was already processed into an occupancy grid before being sent.
    if cfg_debug_mode:
        print("map has shape {:}".format(occ_map.shape))
        cv2.imshow("Motion planning node received map", occ_map); cv2.waitKey(0); cv2.destroyAllWindows()


def main():
    global cmd_pub
    rospy.init_node('motion_planning_node')

    read_params()

    # Subscribe to localization est.
    rospy.Subscriber(topic_localization, Vector3, get_localization_est, queue_size=1)
    # Subscribe to (or just read the map from) file.
    rospy.Subscriber(topic_occ_map, Image, get_map, queue_size=1)

    # Publish control commands.
    cmd_pub = rospy.Publisher(topic_commands, Vector3, queue_size=1)
    # there is a way to command a relative position/yaw motion:
    # python navigation/base_position_control.py --base_planner none --base_controller ilqr --smooth --close_loop --relative_position 1.,1.,1.57 --botname locobot

    if cfg_debug_mode:
        # Create a timer to send test commands every dt seconds.
        rospy.Timer(rospy.Duration(cfg_dt), generate_test_command)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass