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
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cv2
from cv_bridge import CvBridge
from math import remainder, tau, sin, cos, pi

from rotated_rectangle_crop_opencv.rotated_rect_crop import crop_rotated_rectangle

############ GLOBAL VARIABLES ###################
bridge = CvBridge()
observation_pub = None
occ_map_pub = None
raw_map_pub = None
most_recent_measurement = None


occ_map_true = None # Occupancy grid of ground-truth map.
veh_pose_true = np.array([0.0, 0.0, 0.0]) # Ground-truth vehicle pose (x,y,yaw) in map coordinates.
veh_pose_true_px = np.array([0, 0, 0]) # Ground-truth vehicle pose (col,row,yaw) in pixel map coordinates.
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


# # Helper functions
# def rotate_image(src_img, center, angle):
#     """
#     Rotate an image about the given point by the given angle.
#     """
#     rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rot_img = cv2.warpAffine(src_img, rot_mat, src_img.size())
#     return rot_mat


def generate_observation():
    """
    Use the map and known ground-truth robot pose to generate the best possible observation.
    """
    # desired observation size (always a square).
    obs_side_len_px = 100
    # project ahead of vehicle pose to determine center.
    center_col = veh_pose_true_px[0] + (obs_side_len_px / 2) * cos(veh_pose_true_px[2])
    center_row = veh_pose_true_px[0] + (obs_side_len_px / 2) * sin(veh_pose_true_px[2])
    center = (center_col, center_row)
    # create the rotated rectangle.
    width = obs_side_len_px
    height = obs_side_len_px
    angle = np.rad2deg(veh_pose_true_px[2])
    rect = (center, (width, height), angle)
    # crop out the rotated rectangle and reorient it.
    image_cropped = crop_rotated_rectangle(image = occ_map_true, rect = rect)

    # plot .
    fig = plt.figure(figsize=(8, 6)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax0.imshow(occ_map_true)
    ax1 = plt.subplot(gs[1])
    ax1.imshow(image_cropped)
    plt.tight_layout()
    plt.show()


def get_occ_map(msg):
    """
    Get the processed occupancy grid map to use as the "ground truth" map.
    """
    global occ_map_true, veh_pose_true_px
    # Convert from ROS Image message to an OpenCV image.
    occ_map_true = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # Set robot's initial pose to the map center.
    veh_pose_true_px[0] = occ_map_true.shape[0] // 2
    veh_pose_true_px[1] = occ_map_true.shape[1] // 2
    veh_pose_true_px[2] = pi/4

    # debug attempt rotated roi.
    generate_observation()

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