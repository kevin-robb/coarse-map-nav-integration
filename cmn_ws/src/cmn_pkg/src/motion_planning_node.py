#!/usr/bin/env python3

"""
Node to handle ROS interface for getting localization estimate, global map, and performing path planning, navigation, and publishing a control command to the turtlebot.
"""

import rospy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import rospkg, yaml
import numpy as np
import cv2
from cv_bridge import CvBridge
from random import random
from math import pi

from scripts.cmn_utilities import clamp, ObservationGenerator
from scripts.astar import Astar
from scripts.pure_pursuit import PurePursuit

############ GLOBAL VARIABLES ###################
bridge = CvBridge()
obs_gen = ObservationGenerator()
occ_map = None # global occupancy grid map.
# goal pose in global map coords (row, col).
goal_pos_px_r = None
goal_pos_px_c = None
#################################################

##################### UTILITY FUNCTIONS #######################
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
        global g_topic_commands, g_topic_localization, g_topic_occ_map, g_topic_planned_path
        g_topic_occ_map = config["topics"]["occ_map"]
        g_topic_localization = config["topics"]["localization"]
        g_topic_commands = config["topics"]["commands"]
        g_topic_planned_path = config["topics"]["planned_path"]
        # Constraints.
        global g_max_fwd_cmd, g_max_ang_cmd
        g_max_fwd_cmd = config["constraints"]["fwd"]
        g_max_ang_cmd = config["constraints"]["ang"]
        # Goal point.
        global g_goal_col_rel, g_goal_row_rel
        g_goal_col_rel = config["goal"]["x_rel"]
        g_goal_row_rel = config["goal"]["y_rel"]

def publish_command(fwd, ang):
    """
    Clamp a command within valid values, and publish it to the vehicle/simulator.
    """
    fwd = clamp(fwd, 0, g_max_fwd_cmd)
    ang = clamp(ang, -g_max_ang_cmd, g_max_ang_cmd)
    # Create ROS message.
    # NOTE x-component = forward motion, z-component = angular motion. y-component = lateral motion, which is impossible for our system and is ignored.
    msg = Vector3(fwd, 0.0, ang)
    cmd_pub.publish(msg)

######################## CALLBACKS ########################
def get_localization_est(msg):
    """
    Get localization estimate from the particle filter.
    """
    # TODO process it and associate with a particular cell/orientation on the map.
    print("Got localization estimate {:}".format(msg))
    # Convert message into numpy array (x,y,yaw).
    pose_est = np.array([msg.x, msg.y, msg.z])

    # Plan a path from this estimated position to the goal.
    plan_path_to_goal(pose_est)
    
    # DEBUG send a simple motion command.    
    # publish_command(0.02, pi) # drive in a small circle.
    # publish_command(0.5, 0.0) # drive in a straight line.
    
def get_map(msg):
    """
    Get the global occupancy map to use for path planning.
    NOTE Map was already processed into an occupancy grid before being sent.
    """
    global occ_map, goal_pos_px_r, goal_pos_px_c
    # Convert from ROS Image message to an OpenCV image.
    occ_map = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    obs_gen.set_map(occ_map)
    Astar.occ_map = occ_map
    # Set the goal position in pixels.
    goal_pos_px_r = int(clamp(g_goal_row_rel * occ_map.shape[0], 0, occ_map.shape[0]-1))
    goal_pos_px_c = int(clamp(g_goal_col_rel * occ_map.shape[1], 0, occ_map.shape[1]-1))

################ PATH PLANNING FUNCTIONS #####################
def plan_path_to_goal(veh_pose_est):
    """
    Given a desired goal point, use A* to generate a path there,
    starting at the current localization estimate.
    @param veh_pose_est, 3x1 numpy array of localization estimate (x,y,yaw) in meters.
    """
    # Convert vehicle pose from meters to pixels.
    veh_r, veh_c = obs_gen.transform_map_m_to_px(veh_pose_est[0], veh_pose_est[1])

    # Generate (reverse) path with A*.
    path_px_rev = Astar.astar(veh_r, veh_c, goal_pos_px_r, goal_pos_px_c)
    if path_px_rev is None:
        rospy.logerr("No path found by A*.")
        return
    # Turn this path from px to meters and reverse it.
    path = []
    for i in range(len(path_px_rev)-1, -1, -1):
        path.append(obs_gen.transform_map_m_to_px(path_px_rev[i][0], path_px_rev[i][1]))
    # Set the path for pure pursuit, and generate a command.
    PurePursuit.path_meters = path
    fwd, ang = PurePursuit.compute_command(veh_pose_est)
    # Keep within constraints.
    fwd = clamp(fwd, 0, g_max_fwd_cmd)
    ang = clamp(ang, g_max_ang_cmd, g_max_ang_cmd)

    print("Planned path " + str(path_px_rev) + " and commands " + str(fwd) + ", " + str(ang))

    # Publish this motion command.
    publish_command(fwd, ang)

    # Publish the path in pixels for the plotter to display.
    path_pub.publish(Float32MultiArray(data=sum([path_px_rev[i][0] for i in range(len(path_px_rev))], [path_px_rev[i][1] for i in range(len(path_px_rev))])))


# TODO obstacle avoidance?


def main():
    global cmd_pub, path_pub
    rospy.init_node('motion_planning_node')

    read_params()

    # Subscribe to localization est.
    rospy.Subscriber(g_topic_localization, Vector3, get_localization_est, queue_size=1)
    # Subscribe to (or just read the map from) file.
    rospy.Subscriber(g_topic_occ_map, Image, get_map, queue_size=1)

    # Publish control commands.
    cmd_pub = rospy.Publisher(g_topic_commands, Vector3, queue_size=1)
    # there is a way to command a relative position/yaw motion:
    # python navigation/base_position_control.py --base_planner none --base_controller ilqr --smooth --close_loop --relative_position 1.,1.,1.57 --botname locobot

    # Publish planned path to the goal (for viz).
    path_pub = rospy.Publisher(g_topic_planned_path, Float32MultiArray, queue_size=1)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass