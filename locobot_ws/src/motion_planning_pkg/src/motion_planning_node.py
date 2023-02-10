#!/usr/bin/env python3

"""
Node to handle ROS interface for getting localization estimate, global map, and performing path planning, navigation, and publishing a control command to the turtlebot.
"""

import rospy
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from random import random
from math import pi

# rostopic pub --once /locobot/mobile_base/commands/velocity geometry_msgs/Twist '{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}'

############ GLOBAL VARIABLES ###################
bridge = CvBridge()
map = None # raw global occupancy grid map
cmd_pub = None
test_spd_range = (-1, 1)
test_ang_range = (-pi/2, pi/2)
# Config parameters. TODO read from a yaml.
cfg_debug_mode = True
#################################################

# TODO do path planning.
# TODO do navigation & obstacle avoidance.
# TODO create control commands & pub them.

def get_localization_est(msg):
    """
    Get localization estimate from the particle filter.
    """
    print("Got localization estimate")


def generate_test_command(_msg):
    """
    Send a simple twist command to test communication between the ros nodes.
    """
    msg = Twist()
    msg.linear.x = random() * (test_spd_range[1] - test_spd_range[0]) + test_spd_range[0]
    msg.angular.z = random() * (test_ang_range[1] - test_ang_range[0]) + test_ang_range[0]
    print("Commanding linear: " + str(msg.linear.x) + ", angular: " + str(msg.angular.z))
    cmd_pub.publish(msg)

def get_map(msg):
    """
    Get the global occupancy map to use for path planning.
    """
    global map
    # Convert from ROS Image message to an OpenCV image.
    occ_map = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # Threshold it to get a binary occupancy grid.
    # TODO not sure if this is necessary, since it was already an occupancy grid before being sent.
    if cfg_debug_mode:
        cv2.imshow("Motion planning node received map", occ_map); cv2.waitKey(0); cv2.destroyAllWindows()

def main():
    global cmd_pub
    rospy.init_node('motion_planning_node')

    # Subscribe to localization est.
    rospy.Subscriber("/state/particle_filter", Vector3, get_localization_est, queue_size=1)
    # Subscribe to (or just read the map from) file.
    rospy.Subscriber("/map/occ", Image, get_map, queue_size=1)

    # Publish control commands.
    cmd_pub = rospy.Publisher("/locobot/mobile_base/commands/velocity", Twist, queue_size=1)

    if cfg_debug_mode:
        # Create a timer to send test commands every dt seconds.
        dt = 3 # timer period
        rospy.Timer(rospy.Duration(dt), generate_test_command)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass