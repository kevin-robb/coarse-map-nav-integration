#!/usr/bin/env python3

"""
Node to handle ROS interface for ML model that does the actual observation generation from sensor data.
"""

import rospy
import numpy as np
from math import sin, cos, remainder, tau, atan2
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3


############ GLOBAL VARIABLES ###################
observation_pub = None
# Most recent measurements.
lm_meas_queue = []
# current timestep number.
timestep = 0
#################################################

# TODO make way to interface with the ML model, which will be in a separate file.

# create a message and publish it.
def send_state():
    msg = Vector3()
    msg.x = 1
    # publish it.
    observation_pub.publish(msg)

# get measurement from RealSense.
def get_RS_image(msg):
    # format: [id1,range1,bearing1,...idN,rN,bN]
    global lm_meas_queue
    lm_meas_queue.append(msg.data)

def main():
    global observation_pub
    rospy.init_node('perception_node')

    # TODO subscribe to sensor images from RealSense.
    rospy.Subscriber("/realsense", Float32MultiArray, get_RS_image, queue_size=1)

    # TODO publish refined "observation".
    observation_pub = rospy.Publisher("/observation", Vector3, queue_size=1)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass