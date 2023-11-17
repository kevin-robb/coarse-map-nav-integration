#!/usr/bin/env python3

"""
Interface to parse sensor data from the locobot.
"""


import rospy, rospkg
import numpy as np
import yaml, os, cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan, Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from bresenham import bresenham

g_cv_bridge = CvBridge()

g_lidar_local_occ_meas = None # Last successful local occ meas from LiDAR data.
g_lidar_detects_robot_facing_wall:bool = False # Flag if the area in front of the robot is obstructed.
g_depth_local_occ_meas = None # Last successful local occ meas from RS depth data.
# Config params.
g_local_occ_size:int = None # Number of pixels on each side of the square local occupancy grid.
g_local_occ_resolution:float = None # Meters/pixel on the local occupancy grid.


def get_local_occ_from_lidar(msg:LaserScan):
    """
    Get a scan message from the robot's LiDAR.
    Convert it into a pseudo-local-occupancy measurement, akin to the model predictions.
    """
    # Use standard size for model predictions, i.e., 128x128 centered on the robot, with 0.01 m/px resolution.
    # (Size can be configured above, but it will always be centered on the robot.)
    local_occ_meas = np.ones((g_local_occ_size, g_local_occ_size))
    center_r = local_occ_meas.shape[0] // 2
    center_c = local_occ_meas.shape[1] // 2
    max_range_px = msg.range_max / g_local_occ_resolution
    for i in range(len(msg.ranges)):
        # Only use measurements within the valid range.
        if msg.ranges[i] < msg.range_min or msg.ranges[i] > msg.range_max:
            continue
        # Compute angle for this ray based on its index and the given angle range.
        # TODO shift angle based on current robot yaw so this aligns with the global map.
        angle = msg.angle_min + i * msg.angle_increment
        # Convert angle from meters to pixels.
        dist_px = msg.ranges[i] / g_local_occ_resolution
        # Add the data from this ray's detection to the local occ meas.
        # Signs are chosen s.t. the robot is facing EAST on the image.
        r_hit = center_r + int(dist_px * np.sin(angle))
        c_hit = center_c - int(dist_px * np.cos(angle))
        r_max_range = center_r + int(max_range_px * np.sin(angle))
        c_max_range = center_c - int(max_range_px * np.cos(angle))
        for cell in bresenham(r_hit, c_hit, r_max_range, c_max_range):
            # Mark all cells as occupied until leaving the bounds of the image.
            r = cell[0]; c = cell[1]
            if r >= 0 and c >= 0 and r < local_occ_meas.shape[0] and c < local_occ_meas.shape[1]:
                local_occ_meas[r, c] = 0
            else:
                break
    
    # Check if the area in front of the robot is occluded.
    # The motion planner checks this flag in realtime (not just once per CMN iteration) and stop a forward motion if it becomes true.
    global g_lidar_detects_robot_facing_wall
    # Check if the cell in front of the robot (i.e., right center cell) is occupied (i.e., == 0).
    front_cell_block = local_occ_meas[local_occ_meas.shape[0]//3:2*local_occ_meas.shape[0]//3, 2*local_occ_meas.shape[0]//3:]
    front_cell_mean = np.mean(front_cell_block)
    g_lidar_detects_robot_facing_wall = front_cell_mean <= 0.75

    # Save the local occ for CMN to use.
    global g_lidar_local_occ_meas
    g_lidar_local_occ_meas = local_occ_meas.copy()

    if __name__ == '__main__':
        cv2.namedWindow("LiDAR -> local occ meas (front = right)", cv2.WINDOW_NORMAL)
        cv2.imshow("LiDAR -> local occ meas (front = right)", g_lidar_local_occ_meas)
        # print("WALL DETECTED IN FRONT OF ROBOT?: {:}".format(g_lidar_detects_robot_facing_wall))
        
        if g_depth_local_occ_meas is not None:
            cv2.namedWindow("Depth -> local occ meas (front = right)", cv2.WINDOW_NORMAL)
            cv2.imshow("Depth -> local occ meas (front = right)", g_depth_local_occ_meas)

        cv2.waitKey(100)


def get_local_occ_from_depth(msg:Image):
    """
    Process a depth image to get a local occupancy measurement.
    """
    # Convert message to cv2 image type.
    depth_img = g_cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').copy()

    # Use standard size for model predictions, i.e., 128x128 centered on the robot, with 0.01 m/px resolution.
    # (Size can be configured above, but it will always be centered on the robot.)
    local_occ_meas = np.ones((g_local_occ_size, g_local_occ_size))
    center_r = local_occ_meas.shape[0] // 2
    center_c = local_occ_meas.shape[1] // 2
    max_range_px = g_local_occ_size # Ray from center can never be this long, so this is a safe upper bound.
    range_max = max_range_px * g_local_occ_resolution

    rs_range_min = 0.1 # Min feasible range of RS depth sensor, in meters.
    rs_range_max = 5.0 # Max feasible range of RS depth sensor, in meters.

    # Compress the depth information into a LiDAR-like plane.
    ratio_to_use = 1.0 # Ignore the bottom part of the image to avoid ground. Set to 1.0 to keep entire image.
    depth_img_top_half = depth_img[:int(ratio_to_use * depth_img.shape[0]), :]
    # Blank out non-detections with the mean.
    # depth_img_top_half[(depth_img_top_half < rs_range_min) | (depth_img_top_half > rs_range_max)] = np.mean(depth_img_top_half[(depth_img_top_half >= rs_range_min) & (depth_img_top_half <= rs_range_max)])
    depth_img_top_half[depth_img_top_half < rs_range_min] = np.mean(depth_img_top_half[depth_img_top_half >= rs_range_min])
    flat_depth_meas = np.mean(depth_img_top_half, axis=0)
    # Treat this as angle range based on RS depth FOV.
    depth_fov = np.deg2rad(90)
    half_depth_fov = depth_fov / 2
    num_rays = flat_depth_meas.shape[0]
    dtheta = depth_fov / num_rays
    angles = [-half_depth_fov + dtheta*i for i in range(num_rays)]

    # We seem to get erroneous depth readings on the edges of the FOV, so ignore these regions.
    ignore_edge_regions_area = np.deg2rad(0.0) # Degrees to ignore on each side of the FOV. Set 0 to use whole image.
    angle_upper_mag_to_keep = half_depth_fov - ignore_edge_regions_area
    
    for i in range(num_rays):
        angle = angles[i]
        # We seem to get erroneous depth readings on the edges of the FOV, so ignore these regions.
        if angle < -angle_upper_mag_to_keep or angle > angle_upper_mag_to_keep:
            continue

        depth = flat_depth_meas[i] * 0.001 # Convert from mm to meters.
        # Only use measurements within the valid range.
        if depth < rs_range_min or depth > range_max:
            continue
        # Convert measurement from meters to pixels.
        dist_px = depth / g_local_occ_resolution
        # Add the data from this ray's detection to the local occ meas.
        # Signs are chosen s.t. the robot is facing EAST on the image.
        r_hit = center_r + int(dist_px * np.sin(angle))
        c_hit = center_c + int(dist_px * np.cos(angle))
        r_max_range = center_r + int(max_range_px * np.sin(angle))
        c_max_range = center_c + int(max_range_px * np.cos(angle))
        for cell in bresenham(r_hit, c_hit, r_max_range, c_max_range):
            # Mark all cells as occupied until leaving the bounds of the image.
            r = cell[0]; c = cell[1]
            if r >= 0 and c >= 0 and r < local_occ_meas.shape[0] and c < local_occ_meas.shape[1]:
                local_occ_meas[r, c] = 0
            else:
                break
    
    # Save the local occ for CMN to use.
    global g_depth_local_occ_meas
    g_depth_local_occ_meas = local_occ_meas.copy()


def get_local_occ_from_pointcloud(msg:PointCloud2):
    """
    Process a pointcloud message containing depth data.
    """
    rospy.loginfo("Got a pointcloud!")
    gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
    for pt in gen:
        pass
        # print(pt)


def read_params():
    """
    Read configuration params from the yaml.
    """
    # Determine filepath.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('cmn_pkg')
    global g_yaml_path
    g_yaml_path = os.path.join(pkg_path, 'config/config.yaml')
    # Open the yaml and get the relevant params.
    with open(g_yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        # LiDAR params.
        global g_local_occ_size, g_local_occ_resolution
        g_local_occ_size = config["lidar"]["local_occ_size"]
        g_local_occ_resolution = config["lidar"]["local_occ_resolution"]


def main():
    rospy.init_node('interface_node')

    read_params()

    # Subscribe to LiDAR data.
    rospy.Subscriber("/locobot/scan", LaserScan, get_local_occ_from_lidar, queue_size=1)

    # Subscribe to depth data from RealSense.
    rospy.Subscriber("/locobot/camera/depth/image_rect_raw", Image, get_local_occ_from_depth, queue_size=1)

    # Subscribe to depth cloud processed from depth image.
    rospy.Subscriber("/locobot/camera/depth/points", PointCloud2, get_local_occ_from_pointcloud, queue_size=1)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass