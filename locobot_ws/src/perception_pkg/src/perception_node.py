#!/usr/bin/env python3

"""
Node to handle ROS interface for ML model that does the actual observation generation from sensor data.
"""

import rospy
from sensor_msgs.msg import Image
import rospkg, yaml
import numpy as np
import cv2
from cv_bridge import CvBridge

############ GLOBAL VARIABLES ###################
bridge = CvBridge()
observation_pub = None
occ_map_pub = None
raw_map_pub = None
most_recent_measurement = None
#################################################


def read_params():
    """
    Read configuration params from the yaml.
    """
    global cfg_debug_mode, cfg_map_filepath, cfg_obstacle_balloon_radius_px, cfg_dt, topic_measurements, topic_observations, topic_occ_map, topic_raw_map
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
        topic_measurements = config["topics"]["measurements"]
        topic_observations = config["topics"]["observations"]
        topic_occ_map = config["topics"]["occ_map"]
        topic_raw_map = config["topics"]["raw_map"]


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


def read_coarse_map():
    """
    Read the coarse map image from the provided filepath.
    Save the map itself to use for visualizations.
    Process the image by converting it to an occupancy grid.
    I ripped this from my previous personal project since it's pretty general: https://github.com/kevin-robb/live_ekf_slam
    """
    # read map image and account for possible white = transparency that cv2 will think is black.
    # https://stackoverflow.com/questions/31656366/cv2-imread-and-cv2-imshow-return-all-zeros-and-black-image/62985765#62985765
    img = cv2.imread(cfg_map_filepath, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4: # we have an alpha channel
        a1 = ~img[:,:,3] # extract and invert that alpha
        img = cv2.add(cv2.merge([a1,a1,a1,a1]), img) # add up values (with clipping)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # strip alpha channels
    if cfg_debug_mode:
        print("map has shape {:}".format(img.shape))
        cv2.imshow('initial map', img); cv2.waitKey(0); cv2.destroyAllWindows()

    # convert from BGR to RGB and save the color map for any viz.
    raw_map = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # turn this into a grayscale img and then to a binary map.
    occ_map_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)[1]
    # normalize to range [0,1].
    occ_map_img = np.divide(occ_map_img, 255)
    if cfg_debug_mode:
        print("map has shape {:}".format(occ_map_img.shape))
        cv2.imshow("Thresholded Map", occ_map_img); cv2.waitKey(0); cv2.destroyAllWindows()
    
    # consider anything not completely white (1) as occluded (0).
    occ_map = np.floor(occ_map_img)

    # determine index pairs to select all neighbors when ballooning obstacles.
    nbrs = []
    for i in range(-cfg_obstacle_balloon_radius_px, cfg_obstacle_balloon_radius_px+1):
        for j in range(-cfg_obstacle_balloon_radius_px, cfg_obstacle_balloon_radius_px+1):
            nbrs.append((i, j))
    # remove 0,0 which is just the parent cell.
    nbrs.remove((0,0))
    # expand all occluded cells outwards.
    for i in range(len(occ_map)):
        for j in range(len(occ_map[0])):
            if occ_map_img[i][j] != 1: # occluded.
                # mark all neighbors as occluded.
                for chg in nbrs:
                    occ_map[max(0, min(i+chg[0], occ_map.shape[0]-1))][max(0, min(j+chg[1], occ_map.shape[1]-1))] = 0
    occ_map = np.float32(np.array(occ_map))
    if cfg_debug_mode:
        print("map has shape {:}".format(occ_map.shape))
        cv2.imshow("Ballooned Occ Map", occ_map); cv2.waitKey(0); cv2.destroyAllWindows()

    if cfg_debug_mode:
        # show value distribution in occ_map.
        freqs = [0, 0]
        for i in range(len(occ_map)):
            for j in range(len(occ_map[0])):
                if occ_map[i][j] == 0:
                    freqs[0] += 1
                else:
                    freqs[1] += 1
        print("Occ map value frequencies: "+str(freqs[1])+" free, "+str(freqs[0])+" occluded.")

    # turn them into Image messages to publish for other nodes.
    bridge = CvBridge()
    occ_map_pub.publish(bridge.cv2_to_imgmsg(occ_map, encoding="passthrough"))
    raw_map_pub.publish(bridge.cv2_to_imgmsg(raw_map, encoding="passthrough"))


def main():
    global observation_pub, occ_map_pub, raw_map_pub
    rospy.init_node('perception_node')

    read_params()

    # Subscribe to sensor images from RealSense.
    # TODO may want to check /locobot/camera/color/camera_info
    rospy.Subscriber(topic_measurements, Image, get_RS_image, queue_size=1)

    # Publish refined "observation".
    observation_pub = rospy.Publisher(topic_observations, Image, queue_size=1)

    # Publish the coarse map for other nodes to use.
    occ_map_pub = rospy.Publisher(topic_occ_map, Image, queue_size=1)
    raw_map_pub = rospy.Publisher(topic_raw_map, Image, queue_size=1)
    read_coarse_map()

    # Main update loop will be a timer that checks for a new measurement, and attempts to generate an observation.
    rospy.Timer(rospy.Duration(cfg_dt), get_observation_from_model)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass