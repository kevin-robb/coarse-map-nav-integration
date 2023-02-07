#!/usr/bin/env python3

"""
Node to handle ROS interface for ML model that does the actual observation generation from sensor data.
"""

import rospy
import numpy as np
from sensor_msgs.msg import Image


############ GLOBAL VARIABLES ###################
observation_pub = None
# Most recent measurement.
meas_img = None
meas_img_height = 0
meas_img_width = 0
#################################################

# TODO make way to interface with the ML model, which will be in a separate file.

# create a message and publish it.
def send_state():
    msg = Image()
    # publish it.
    observation_pub.publish(msg)

# get measurement from RealSense.
def get_RS_image(msg):
    global meas_img, meas_img_height, meas_img_width
    meas_img_height = msg.height
    meas_img_width = msg.width
    meas_img = msg.data

def main():
    global observation_pub
    rospy.init_node('perception_node')

    # Subscribe to sensor images from RealSense.
    # TODO may want to check /locobot/camera/color/camera_info
    rospy.Subscriber("/locobot/camera/color/image_raw", Image, get_RS_image, queue_size=1)

    # Publish refined "observation".
    observation_pub = rospy.Publisher("/observation", Image, queue_size=1)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass