#!/usr/bin/env python3

"""
Interface to publish commands to the physical locobot and parse sensor data.
"""


import rospy
import numpy as np
import os, cv2
from sensor_msgs.msg import LaserScan


def get_lidar(msg:LaserScan):
    """
    Get a scan message from the robot's LiDAR.
    Convert it into a pseudo-local-occupancy measurement, akin to the model predictions.
    """
    # Use standard size for model predictions, i.e., 128x128 centered on the robot, with 0.01 m/px resolution.
    local_occ_meas = np.zeros((128, 128))
    center_r = local_occ_meas.shape[0] // 2
    center_c = local_occ_meas.shape[1] // 2
    for i in range(len(msg.ranges)):
        # Only use measurements within the valid range.
        if msg.ranges[i] < msg.range_min or msg.ranges[i] > msg.range_max:
            continue
        # Compute angle for this ray based on its index and the given angle range.
        # TODO shift angle based on current robot yaw so this aligns with the global map.
        angle = msg.angle_min + i * msg.angle_increment
        # Convert angle from meters to pixels.
        dist_px = msg.ranges[i] / 0.01
        # Add the data from this ray's detection to the local occ meas.
        r = center_r - int(dist_px * np.sin(angle))
        c = center_c + int(dist_px * np.cos(angle))
        if r >= 0 and c >= 0 and r < local_occ_meas.shape[0] and c < local_occ_meas.shape[1]:
            local_occ_meas[r, c] = 1

    cv2.namedWindow("LiDAR -> local occ meas", cv2.WINDOW_NORMAL)
    cv2.imshow("LiDAR -> local occ meas", local_occ_meas)
    cv2.waitKey(100)


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