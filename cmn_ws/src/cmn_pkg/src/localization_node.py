#!/usr/bin/env python3

"""
Node to handle ROS interface for getting observations, running the localization filter, and publishing the best estimate of vehicle position on the map.
"""

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
import rospkg, yaml
import numpy as np
import cv2
from cv_bridge import CvBridge

from scripts.particle_filter import ParticleFilter

############ GLOBAL VARIABLES ###################
bridge = CvBridge()
localization_pub = None
pf = ParticleFilter()
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
        global g_topic_observations, g_topic_occ_map, g_topic_localization, g_topic_commands
        g_topic_observations = config["topics"]["observations"]
        g_topic_occ_map = config["topics"]["occ_map"]
        g_topic_localization = config["topics"]["localization"]
        g_topic_commands = config["topics"]["commands"]


def get_observation(msg):
    """
    Get an observation Image from the ML model's output.
    Use this to update the particle filter.
    Publish the best estimate from the pf as our localization result.
    """
    # Update the particle filter.
    pf_estimate = pf.update_with_observation(msg.data)
    pf.resample()
    # Convert pf estimate into a message and publish it.
    loc_est = Vector3(pf_estimate[0], pf_estimate[1], pf_estimate[2])
    localization_pub.publish(loc_est)


def get_occ_map(msg):
    """
    Get the processed occupancy grid map to use for PF measurement likelihood.
    """
    # Convert from ROS Image message to an OpenCV image.
    occ_map = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # Save the map in the particle filter for it to use later.
    pf.set_map(occ_map)


def get_command(msg:Vector3):
    """
    Receive a commanded motion, and propagate all particles.
    """
    pf.propagate_particles(msg.x, msg.z)


def main():
    global localization_pub
    rospy.init_node('localization_node')

    read_params()
    # Init the particle filter instance.

    # Subscribe to occupancy grid map. Needed for PF's measurement likelihood step.
    rospy.Subscriber(g_topic_occ_map, Image, get_occ_map, queue_size=1)
    # Subscribe to observations.
    rospy.Subscriber(g_topic_observations, Image, get_observation, queue_size=1)
    # Subscribe to commanded motion. Needed to propagate particles between iterations.
    rospy.Subscriber(g_topic_commands, Vector3, get_command, queue_size=1)

    # Publish localization estimate.
    localization_pub = rospy.Publisher(g_topic_localization, Vector3, queue_size=1)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass