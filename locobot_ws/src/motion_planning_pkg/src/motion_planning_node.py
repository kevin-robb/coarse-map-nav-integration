#!/usr/bin/env python3

"""
Node to handle ROS interface for getting localization estimate, global map, and performing path planning, navigation, and publishing a control command to the turtlebot.
"""

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from random import random
from math import pi

# rostopic pub --once /locobot/mobile_base/commands/velocity geometry_msgs/Twist '{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}'

############ GLOBAL VARIABLES ###################
cmd_pub = None
test_spd_range = (-1, 1)
test_ang_range = (-pi/2, pi/2)
#################################################

# TODO do path planning.
# TODO do navigation & obstacle avoidance.
# TODO create control commands & pub them.

def generate_test_command(_msg):
    """
    Send a simple twist command to test communication between the ros nodes.
    """
    msg = Twist()
    msg.linear.x = random() * (test_spd_range[1] - test_spd_range[0]) + test_spd_range[0]
    msg.angular.z = random() * (test_ang_range[1] - test_ang_range[0]) + test_ang_range[0]
    print("Commanding linear: " + str(msg.linear.x) + ", angular: " + str(msg.angular.z))
    cmd_pub.publish(msg)

def main():
    global cmd_pub
    rospy.init_node('motion_planning_node')

    # TODO subscribe to localization est.
    # TODO subscribe to or just read the map from file.

    # Publish control commands.
    cmd_pub = rospy.Publisher("/locobot/mobile_base/commands/velocity", Twist, queue_size=1)

    # Create a timer.
    dt = 3 # timer period
    rospy.Timer(rospy.Duration(dt), generate_test_command)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass