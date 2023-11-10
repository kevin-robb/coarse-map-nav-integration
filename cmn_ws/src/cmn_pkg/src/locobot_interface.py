#!/usr/bin/env python3

"""
Interface to parse sensor data from the locobot.
"""


import rospy
import numpy as np
import os, cv2
from sensor_msgs.msg import LaserScan
from bresenham import bresenham

g_lidar_local_occ_meas = None # Last successful local occ meas from LiDAR data.
g_lidar_detects_robot_facing_wall:bool = False # Flag if the area in front of the robot is obstructed.

def get_lidar(msg:LaserScan):
    """
    Get a scan message from the robot's LiDAR.
    Convert it into a pseudo-local-occupancy measurement, akin to the model predictions.
    """
    # Use standard size for model predictions, i.e., 128x128 centered on the robot, with 0.01 m/px resolution.
    local_occ_meas = np.ones((128, 128))
    resolution = 0.01
    center_r = local_occ_meas.shape[0] // 2
    center_c = local_occ_meas.shape[1] // 2
    max_range_px = msg.range_max / resolution
    for i in range(len(msg.ranges)):
        # Only use measurements within the valid range.
        if msg.ranges[i] < msg.range_min or msg.ranges[i] > msg.range_max:
            continue
        # Compute angle for this ray based on its index and the given angle range.
        # TODO shift angle based on current robot yaw so this aligns with the global map.
        angle = msg.angle_min + i * msg.angle_increment
        # Convert angle from meters to pixels.
        dist_px = msg.ranges[i] / resolution
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
    global g_lidar_detects_robot_facing_wall
    # Check if the cell in front of the robot (i.e., right center cell) is occupied (i.e., == 0).
    front_cell_block = local_occ_meas[local_occ_meas.shape[0]//3:2*local_occ_meas.shape[0]//3, 2*local_occ_meas.shape[0]//3:]
    front_cell_mean = np.mean(front_cell_block)
    g_lidar_detects_robot_facing_wall = front_cell_mean <= 0.75
    # TODO make the motion planner check this in realtime (not just once per iteration) and stop a forward motion if it becomes true.

    if __name__ == '__main__':
        cv2.namedWindow("LiDAR -> local occ meas (front = right)", cv2.WINDOW_NORMAL)
        cv2.imshow("LiDAR -> local occ meas (front = right)", local_occ_meas)
        print("WALL DETECTED IN FRONT OF ROBOT?: {:}".format(g_lidar_detects_robot_facing_wall))
        cv2.waitKey(100)
    else:
        # Save the local occ for CMN to use.
        global g_lidar_local_occ_meas
        g_lidar_local_occ_meas = local_occ_meas.copy()



def main():
    rospy.init_node('interface_node')

    # Subscribe to LiDAR data.
    rospy.Subscriber("/locobot/scan", LaserScan, get_lidar, queue_size=1)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass