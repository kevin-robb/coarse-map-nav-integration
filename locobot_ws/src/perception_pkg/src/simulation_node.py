#!/usr/bin/env python3

"""
Node for extremely basic testing of localization node in best-case-scenario:
 - "coarse map" used by pf is exactly the same as the ground truth map.
 - observations will be ground truth.
 - motion commands will be followed exactly, with no noise.
"""

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
import rospkg, yaml
import numpy as np
import cv2
from cv_bridge import CvBridge
from math import remainder, tau

############ GLOBAL VARIABLES ###################
bridge = CvBridge()
observation_pub = None
occ_map_pub = None
raw_map_pub = None
most_recent_measurement = None


occ_map_true = None # Occupancy grid of ground-truth map.
veh_pose_true = np.array([0.0, 0.0, 0.0]) # Ground-truth vehicle pose (x,y,yaw) in map coordinates.
#################################################
# use bilinear interpolation on map to query expected value at certain pt.

######### TRANSFORMS ############
def transform_map_px_to_m(row:int, col:int):
    """
    Given coordinates of a cell on the ground-truth map, compute the equivalent position in meters.
    """
    # TODO
    pass

def transform_map_m_to_px(x:int, y:int):
    """
    Given coordinates of a vehicle pose in meters, compute the equivalent cell in pixels.
    """
    # TODO
    pass


######### SETUP ############
def read_params():
    """
    Read configuration params from the yaml.
    """
    global cfg_debug_mode, cfg_map_filepath, cfg_obstacle_balloon_radius_px, cfg_dt, topic_observations, topic_occ_map, topic_commands
    # Determine filepath.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('perception_pkg')
    # Open the yaml and get the relevant params.
    with open(pkg_path+'/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        cfg_debug_mode = config["test"]["run_debug_mode"]
        cfg_map_filepath = pkg_path + "/config/maps/" + config["map"]["fname"]
        cfg_obstacle_balloon_radius_px = config["map"]["obstacle_balloon_radius"]
        cfg_dt = config["perception_node_dt"]
        # Rostopics:
        topic_observations = config["topics"]["observations"]
        topic_occ_map = config["topics"]["occ_map"]
        topic_commands = config["topics"]["commands"]


def generate_observation(event):
    """
    Use the map and known ground-truth robot pose to generate the best possible observation.
    """
    pass


def get_occ_map(msg):
    """
    Get the processed occupancy grid map to use as the "ground truth" map.
    """
    global occ_map_true
    # Convert from ROS Image message to an OpenCV image.
    occ_map_true = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')


def get_command(msg:Vector3):
    """
    Receive a commanded motion, which will move the robot accordingly.
    """
    global veh_pose_true
    # TODO perturb with some noise.
    veh_pose_true[0] += msg.x
    veh_pose_true[1] += msg.y
    # keep yaw normalized to (-pi, pi)
    veh_pose_true[2] = remainder(veh_pose_true[2] + msg.z, tau)


def main():
    global observation_pub, occ_map_pub, raw_map_pub
    rospy.init_node('simulation_node')

    read_params()

    # Subscribe to occupancy grid map to use as ground-truth.
    rospy.Subscriber(topic_occ_map, Image, get_occ_map, queue_size=1)
    # Subscribe to commanded motion.
    rospy.Subscriber(topic_commands, Vector3, get_command, queue_size=1)

    # Publish ground-truth observation
    observation_pub = rospy.Publisher(topic_observations + "/true", Image, queue_size=1)

    # NOTE this architecture forms a cycle, observation -> localization -> command, so to complete it we will generate a new observation upon receiving a command.
    # To kick-start this cycle, we will assume we've just received a zero command.

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass