#!/usr/bin/env python3

"""
Node to handle ROS interface for ML model that does the actual observation generation from sensor data.
"""

import rospy
from sensor_msgs.msg import Image
import rospkg, yaml
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from scripts.cmn_utilities import clamp, CoarseMapProcessor

############ GLOBAL VARIABLES ###################
bridge = CvBridge()
map_proc = CoarseMapProcessor()
most_recent_measurement = None
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
        global g_show_map_images, g_map_fpath, g_obs_balloon_radius, g_dt, g_map_downscale_ratio
        g_show_map_images = config["test"]["run_debug_mode"]
        g_map_fpath = pkg_path + "/config/maps/" + config["map"]["fname"]
        g_map_downscale_ratio = config["map"]["downscale_ratio"]
        g_obs_balloon_radius = config["map"]["obstacle_balloon_radius"]
        g_dt = config["dt"]
        # Rostopics:
        global g_topic_measurements, g_topic_observations, g_topic_occ_map, g_topic_raw_map
        g_topic_measurements = config["topics"]["measurements"]
        g_topic_observations = config["topics"]["observations"]
        g_topic_occ_map = config["topics"]["occ_map"]
        g_topic_raw_map = config["topics"]["raw_map"]


def get_observation_from_model(event):
    """
    Pass the most recent measurement into the ML model, and get back a resulting observation.
    Publish the observation to be used for localization.
    """
    global most_recent_measurement
    if most_recent_measurement is None:
        return
    # Pull most recent measurement, and clear it so we'll be able to tell next iteration if there's nothing new.
    meas = most_recent_measurement
    most_recent_measurement = None

    # TODO make way to interface with the ML model, which will be in a separate file.
    observation = None # ml_model(meas)

    # Convert to ROS image and publish it.
    observation_pub.publish(bridge.cv2_to_imgmsg(observation, encoding="passthrough"))


def get_RS_image(msg):
    """
    Get a measurement Image from the RealSense camera.
    Could be changed multiple times before the model is ready again, so this allows skipping measurements to prefer recency.
    """
    global most_recent_measurement
    most_recent_measurement = msg

def main():
    rospy.init_node('perception_node')

    read_params()

    # Subscribe to sensor images from RealSense.
    # TODO may want to check /locobot/camera/color/camera_info
    rospy.Subscriber(g_topic_measurements, Image, get_RS_image, queue_size=1)

    global observation_pub, occ_map_pub, raw_map_pub
    # Publish refined "observation".
    observation_pub = rospy.Publisher(g_topic_observations, Image, queue_size=1)
    # Publish the coarse map for other nodes to use.
    occ_map_pub = rospy.Publisher(g_topic_occ_map, Image, queue_size=1)
    raw_map_pub = rospy.Publisher(g_topic_raw_map, Image, queue_size=1)

    # Wait to make sure the other nodes are ready to receive the map.
    rospy.sleep(1)
    occ_map_pub.publish(map_proc.get_occ_map_msg())
    raw_map_pub.publish(map_proc.get_raw_map_msg())


    # Main update loop will be a timer that checks for a new measurement, and attempts to generate an observation.
    # TODO find a way to just wait until the commanded motion has completed, and then generate an observation.
    rospy.Timer(rospy.Duration(g_dt), get_observation_from_model)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass