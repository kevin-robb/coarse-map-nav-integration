#!/usr/bin/env python3

"""
Node to handle ROS interface for getting localization estimate, global map, and performing path planning, navigation, and publishing a control command to the turtlebot.
"""

import rospy
import numpy as np


############ GLOBAL VARIABLES ###################
observation_pub = None
#################################################

# TODO do path planning.
# TODO do navigation & obstacle avoidance.
# TODO create control commands & pub them.

def main():
    global observation_pub
    rospy.init_node('motion_planning_node')

    # TODO subscribe to localization est.
    # TODO subscribe to or just read the map from file.

    # TODO publish control commands.

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass