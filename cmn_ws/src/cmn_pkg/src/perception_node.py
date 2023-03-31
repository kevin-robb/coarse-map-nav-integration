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

from scripts.cmn_utilities import clamp

############ GLOBAL VARIABLES ###################
bridge = CvBridge()
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
        g_dt = config["perception_node_dt"]
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


def read_coarse_map():
    """
    Read the coarse map image from the provided filepath.
    Save the map itself to use for visualizations.
    Process the image by converting it to an occupancy grid.
    I ripped this from my previous personal project since it's pretty general: https://github.com/kevin-robb/live_ekf_slam
    """
    # Read map image and account for possible white = transparency that cv2 will think is black.
    # https://stackoverflow.com/questions/31656366/cv2-imread-and-cv2-imshow-return-all-zeros-and-black-image/62985765#62985765
    img = cv2.imread(g_map_fpath, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4: # we have an alpha channel.
        a1 = ~img[:,:,3] # extract and invert that alpha.
        img = cv2.add(cv2.merge([a1,a1,a1,a1]), img) # add up values (with clipping).
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # strip alpha channels.

    rospy.loginfo("PER: Read raw coarse map image with shape {:}".format(img.shape))
    if g_show_map_images:
        cv2.imshow('initial map', img); cv2.waitKey(0); cv2.destroyAllWindows()

    # Downsize the image to the desired resolution.
    img = cv2.resize(img, (int(img.shape[0] * g_map_downscale_ratio), int(img.shape[1] * g_map_downscale_ratio)))
    rospy.loginfo("PER: Resized coarse map to shape {:}".format(img.shape))
    if g_show_map_images:
        cv2.imshow('resized map', img); cv2.waitKey(0); cv2.destroyAllWindows()

    # Convert from BGR to RGB and save the color map for any viz.
    raw_map = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Turn this into a grayscale img and then to a binary map.
    occ_map_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)[1]
    # Normalize to range [0,1].
    occ_map_img = np.divide(occ_map_img, 255)
    rospy.loginfo("PER: Thresholded/binarized map to shape {:}".format(img.shape))
    if g_show_map_images:
        cv2.imshow("Thresholded Map", occ_map_img); cv2.waitKey(0); cv2.destroyAllWindows()
    
    # Consider anything not completely white (1) as occluded (0).
    occ_map = np.floor(occ_map_img)

    # Expand occluded cells so path planning won't take us right next to obstacles.
    global g_obs_balloon_radius
    if g_obs_balloon_radius == 0:
        rospy.logwarn("For some reason this doesn't work if we skip the ballooning step.")
        g_obs_balloon_radius = 1
    # Determine index pairs to select all neighbors when ballooning obstacles.
    nbrs = []
    for i in range(-g_obs_balloon_radius, g_obs_balloon_radius+1):
        for j in range(-g_obs_balloon_radius, g_obs_balloon_radius+1):
            nbrs.append((i, j))
    # Remove 0,0 which is just the parent cell.
    nbrs.remove((0,0))
    # Expand all occluded cells outwards.
    for i in range(len(occ_map)):
        for j in range(len(occ_map[0])):
            if occ_map_img[i][j] != 1: # occluded.
                # Mark all neighbors as occluded.
                for chg in nbrs:
                    occ_map[clamp(i+chg[0], 0, occ_map.shape[0]-1)][clamp(j+chg[1], 0, occ_map.shape[1]-1)] = 0
    occ_map = np.float32(np.array(occ_map))
    if g_show_map_images:
        cv2.imshow("Ballooned Occ Map", occ_map); cv2.waitKey(0); cv2.destroyAllWindows()

    if g_show_map_images:
        # Show value distribution in occ_map.
        freqs = [0, 0]
        for i in range(len(occ_map)):
            for j in range(len(occ_map[0])):
                if occ_map[i][j] == 0:
                    freqs[0] += 1
                else:
                    freqs[1] += 1
        rospy.loginfo("PER: Occ map value frequencies: "+str(freqs[1])+" free, "+str(freqs[0])+" occluded.")

    # Turn them into Image messages to publish for other nodes.
    # Wait to make sure the other nodes are ready to receive it.
    rospy.sleep(1)
    try:
        occ_map_pub.publish(bridge.cv2_to_imgmsg(occ_map, encoding="passthrough"))
        raw_map_pub.publish(bridge.cv2_to_imgmsg(raw_map, encoding="passthrough"))
        rospy.loginfo("PER: Published processed (coarse) occupancy map with shape {:}.".format(occ_map.shape))
    except CvBridgeError as e:
        rospy.logerr("PER: Unable to publish processed coarse map. Error: " + e)


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
    read_coarse_map()

    # Main update loop will be a timer that checks for a new measurement, and attempts to generate an observation.
    # TODO find a way to just wait until the commanded motion has completed, and then generate an observation.
    rospy.Timer(rospy.Duration(g_dt), get_observation_from_model)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass