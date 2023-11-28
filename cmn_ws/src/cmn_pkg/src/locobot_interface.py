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
from typing import Tuple

g_cv_bridge = CvBridge()

g_pointcloud_msg = None # Pointcloud message from RealSense depth data.

g_lidar_local_occ_meas = None # Last successful local occ meas from LiDAR data.
g_lidar_detects_robot_facing_wall:bool = False # Flag if the area in front of the robot is obstructed.
g_depth_local_occ_meas = None # Last successful local occ meas from RS depth image.
g_pointcloud_local_occ_meas = None # Last successful local occ meas from RS depth pointcloud.
# Config params.
g_local_occ_size:int = None # Number of pixels on each side of the square local occupancy grid.
g_local_occ_resolution:float = None # Meters/pixel on the local occupancy grid.


def show_images(event=None):
    """
    When running this as a standalone node, the result images are displayed.
    """
    if __name__ == '__main__':
        if g_lidar_local_occ_meas is not None:
            cv2.namedWindow("LiDAR -> local occ meas (front = right)", cv2.WINDOW_NORMAL)
            cv2.imshow("LiDAR -> local occ meas (front = right)", g_lidar_local_occ_meas)
            # print("WALL DETECTED IN FRONT OF ROBOT?: {:}".format(g_lidar_detects_robot_facing_wall))
        
        if g_depth_local_occ_meas is not None:
            cv2.namedWindow("Depth Img -> local occ meas (front = right)", cv2.WINDOW_NORMAL)
            cv2.imshow("Depth Img -> local occ meas (front = right)", g_depth_local_occ_meas)

        # if g_pointcloud_local_occ_meas is not None:
        global g_pointcloud_msg
        if g_pointcloud_msg is not None:
            # Save it so we don't process the same pointcloud more than once.
            pc = g_pointcloud_msg
            g_pointcloud_msg = None
            get_local_occ_from_pointcloud(pc)
            cv2.namedWindow("Pointcloud -> local occ meas (front = right)", cv2.WINDOW_NORMAL)
            cv2.imshow("Pointcloud -> local occ meas (front = right)", g_pointcloud_local_occ_meas)

        cv2.waitKey(100)


def get_local_occ_from_lidar(msg:LaserScan):
    """
    Get a scan message from the robot's LiDAR.
    Convert it into a pseudo-local-occupancy measurement, akin to the model predictions.
    @param msg - Scan from the 360 degree planar LiDAR.
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
        # NOTE we could potentially shift angle here based on current robot yaw so this aligns with the global map.
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


def get_local_occ_from_depth(msg:Image):
    """
    Process a depth image to get a local occupancy measurement.
    @param msg - Raw rectified depth image from the RealSense.
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
    @param msg - Pointcloud from the RealSense depth data.
    """
    gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))

    # Use standard size for model predictions, i.e., 128x128 centered on the robot, with 0.01 m/px resolution.
    # (Size can be configured above, but it will always be centered on the robot.)
    local_occ_meas = np.ones((g_local_occ_size, g_local_occ_size))
    center_r = local_occ_meas.shape[0] // 2
    center_c = local_occ_meas.shape[1] // 2
    max_range_px = g_local_occ_size # Ray from center can never be this long, so this is a safe upper bound.

    for pt in gen:
        # Skip points outside max sensor range.
        if pt[0] > 5 or pt[1] > 5 or pt[2] > 5:
            continue
        # print("Pt: ({:.3f}, {:.3f}, {:.3f})".format(pt[0], pt[1], pt[2]))
        # Flatten the point onto the local occupancy grid (ignore z).
        # Signs are chosen s.t. the robot is facing EAST on the image.
        # For RealSense, +x is to the right, +y is down, +z is forward.
        dc = pt[2] / g_local_occ_resolution
        dr = pt[0] / g_local_occ_resolution
        r_hit = center_r + dr
        c_hit = center_c + dc

        # Skip points out of bounds of the image.
        if r_hit < 0 or c_hit < 0 or r_hit >= g_local_occ_size or c_hit >= g_local_occ_size:
            continue
        # Skip points too close to the robot.
        if dc < 3:
            continue

        # We want to mark not only this cell as occupied, but also all cells behind it.
        angle = np.arctan2(dr, dc)
        r_max_range = center_r + max_range_px * np.sin(angle)
        c_max_range = center_c + max_range_px * np.cos(angle)
        for cell in bresenham(int(r_hit), int(c_hit), int(r_max_range), int(c_max_range)):
            # Mark all cells as occupied until leaving the bounds of the image.
            r = cell[0]; c = cell[1]
            if r >= 0 and c >= 0 and r < local_occ_meas.shape[0] and c < local_occ_meas.shape[1]:
                local_occ_meas[r, c] = 0
            else:
                break

    # Save the local occ for CMN to use.
    global g_pointcloud_local_occ_meas
    g_pointcloud_local_occ_meas = local_occ_meas.copy()


def get_pointcloud_msg(msg:PointCloud2):
    """
    Get a pointcloud message, and save it to use.
    """
    global g_pointcloud_msg
    g_pointcloud_msg = msg


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
    # rospy.Subscriber("/locobot/scan", LaserScan, get_local_occ_from_lidar, queue_size=1)

    # Subscribe to depth data from RealSense.
    # rospy.Subscriber("/locobot/camera/depth/image_rect_raw", Image, get_local_occ_from_depth, queue_size=1)

    # Subscribe to depth cloud processed from depth image.
    rospy.Subscriber("/locobot/camera/depth/points", PointCloud2, get_pointcloud_msg, queue_size=1)

    rospy.Timer(rospy.Duration(0.1), show_images)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass