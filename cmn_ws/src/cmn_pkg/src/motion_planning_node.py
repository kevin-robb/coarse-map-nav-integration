#!/usr/bin/env python3

"""
Node to handle ROS interface for getting localization estimate, global map, and performing path planning, navigation, and publishing a control command to the turtlebot.
"""

import rospy, sys
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import rospkg, yaml
import numpy as np
import cv2
from cv_bridge import CvBridge
from random import random
from math import pi, radians
from time import time
from random import randint, random

from scripts.cmn_utilities import clamp, ObservationGenerator
from scripts.astar import Astar
from scripts.pure_pursuit import PurePursuit

############ GLOBAL VARIABLES ###################
bridge = CvBridge()
obs_gen = ObservationGenerator()
astar = Astar()
# goal pose in global map coords (row, col).
goal_pos_px = None
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
        global g_debug_mode, g_do_path_planning
        g_debug_mode = config["test"]["run_debug_mode"]
        g_do_path_planning = config["path_planning"]["do_path_planning"]
        PurePursuit.use_finite_lookahead_dist = g_do_path_planning
        # Rostopics.
        global g_topic_commands, g_topic_localization, g_topic_occ_map, g_topic_planned_path, g_topic_goal
        g_topic_occ_map = config["topics"]["occ_map"]
        g_topic_localization = config["topics"]["localization"]
        g_topic_goal = config["topics"]["goal"]
        g_topic_commands = config["topics"]["commands"]
        g_topic_planned_path = config["topics"]["planned_path"]
        # Constraints.
        global g_max_fwd_cmd, g_max_ang_cmd
        g_max_fwd_cmd = config["constraints"]["fwd"]
        g_max_ang_cmd = config["constraints"]["ang"]
        # In motion test mode, only this node will run, so it will handle the timer.
        global g_dt
        g_dt = config["dt"]
        # Params for discrete actions.
        global g_use_discrete_actions, g_discrete_forward_dist, g_discrete_forward_skip_probability, g_test_motion_type
        g_use_discrete_actions = config["actions"]["use_discrete_actions"]
        g_test_motion_type = "random_discrete" if g_use_discrete_actions else "circle"
        rospy.logwarn("Set g_test_motion_type: {:}".format(g_test_motion_type))
        g_discrete_forward_dist = abs(config["actions"]["discrete_forward_dist"])
        g_discrete_forward_skip_probability = config["actions"]["discrete_forward_skip_probability"]
        if g_discrete_forward_skip_probability < 0.0 or g_discrete_forward_skip_probability > 1.0:
            rospy.logwarn("MOT: Invalid value of discrete_forward_skip_probability. Must lie in range [0, 1]. Setting to 0.")
            g_discrete_forward_skip_probability = 0

def publish_command(fwd, ang):
    """
    Clamp a command within valid values, and publish it to the vehicle/simulator.
    """
    # Clamp to allowed velocity ranges.
    fwd = clamp(fwd, 0, g_max_fwd_cmd)
    ang = clamp(ang, -g_max_ang_cmd, g_max_ang_cmd)
    rospy.loginfo("MOT: Publishing a command ({:}, {:})".format(fwd, ang))
    # Create ROS message.
    msg = Twist(Vector3(fwd, 0, 0), Vector3(0, 0, ang))
    cmd_pub.publish(msg)

######################## CALLBACKS ########################
def test_timer_callback(event=None):
    """
    Only runs in test mode.
    Publish desired type of test motion every iteration.
    """
    fwd, ang = 0.0, 0.0
    if g_test_motion_type == "none":
        pass
    elif g_test_motion_type == "circle":
        fwd, ang = g_max_fwd_cmd, g_max_ang_cmd
    elif g_test_motion_type == "straight":
        fwd, ang = g_max_fwd_cmd, 0.0
    elif g_test_motion_type == "random_discrete":
        random_discrete_action()
        return
    else:
        rospy.logerr("MOT: test mode called with invalid test_motion_type: {:}".format(g_test_motion_type))
        # exit()
    # Send the motion to the robot.
    publish_command(fwd, ang)

def get_localization_est(msg:Vector3):
    """
    Get localization estimate from the particle filter.
    """
    # TODO process it and associate with a particular cell/orientation on the map.
    rospy.loginfo("MOT: Got localization estimate ({:.2f}, {:.2f}, {:.2f})".format(msg.x, msg.y, msg.z))
    # Convert message into numpy array (x,y,yaw).
    pose_est = np.array([msg.x, msg.y, msg.z])

    # Choose a motion command to send. Must send something to keep cycle going.
    fwd, ang = 0.0, 0.0
    if goal_pos_px is None:
        rospy.loginfo("MOT: No goal point, so commanding constant motion.")
        # Set a simple motion command, since we have no goal to plan towards.  
        test_timer_callback()
    else:
        rospy.loginfo("MOT: Goal point exists, so planning a path there.")
        # Plan a path from this estimated position to the goal.
        fwd, ang = plan_path_to_goal(pose_est)
        # Publish the motion command.
        publish_command(fwd, ang)

def get_goal_pos(msg:Vector3):
    """
    Get goal position in pixels.
    For now, this is obtained from the user clicking on the map in the sim viz.
    """
    rospy.loginfo("MOT: Got goal pos ({:}, {:})".format(int(msg.x), int(msg.y)))
    global goal_pos_px
    goal_pos_px = (int(msg.x), int(msg.y))
    
def get_map(msg:Image):
    """
    Get the global occupancy map to use for path planning.
    NOTE Map was already processed into an occupancy grid before being sent.
    """
    rospy.loginfo("MOT: Got occupancy map.")
    # Convert from ROS Image message to an OpenCV image.
    occ_map = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    obs_gen.set_map(occ_map)
    astar.map = obs_gen.map

################ PATH PLANNING FUNCTIONS #####################
def plan_path_to_goal(veh_pose_est):
    """
    Given a desired goal point, use A* to generate a path there,
    starting at the current localization estimate.
    @param veh_pose_est, 3x1 numpy array of localization estimate (x,y,yaw) in meters.
    """
    # Convert vehicle pose from meters to pixels.
    veh_r, veh_c = obs_gen.transform_map_m_to_px(veh_pose_est[0], veh_pose_est[1])

    if g_do_path_planning:
        # Generate (reverse) path with A*.
        path_px_rev = astar.run_astar(veh_r, veh_c, goal_pos_px[0], goal_pos_px[1])
    else:
        # Just use the goal point as the "path".
        path_px_rev = [goal_pos_px, (veh_r, veh_c)]
    if path_px_rev is None:
        rospy.logerr("MOT: No path found by A*. Publishing zeros for motion command.")
        return 0.0, 0.0
    # rospy.loginfo("MOT: Planned path from A*: " + str(path_px_rev))
    # Turn this path from px to meters and reverse it.
    path = []
    for i in range(len(path_px_rev)-1, -1, -1):
        path.append(obs_gen.transform_map_px_to_m(path_px_rev[i][0], path_px_rev[i][1]))
        # Check if the path contains any occluded cells.
        if obs_gen.map[path_px_rev[i][0], path_px_rev[i][1]] == 0:
            rospy.logwarn("MOT: Path contains an occluded cell.")

    # Set the path for pure pursuit, and generate a command.
    PurePursuit.path_meters = path
    fwd, ang = PurePursuit.compute_command(veh_pose_est)
    # Keep within constraints.
    fwd_clamped = clamp(fwd, 0, g_max_fwd_cmd)
    ang_clamped = clamp(ang, -g_max_ang_cmd, g_max_ang_cmd)
    if fwd != fwd_clamped or ang != ang_clamped:
        rospy.logwarn("MOT: Clamped pure pursuit output from ({:.2f}, {:.2f}) to ({:.2f}, {:.2f}).".format(fwd, ang, fwd_clamped, ang_clamped))

    # Publish the path in pixels for the plotter to display.
    path_as_list = [path_px_rev[i][0] for i in range(len(path_px_rev))] + [path_px_rev[i][1] for i in range(len(path_px_rev))]
    path_pub.publish(Float32MultiArray(data=path_as_list))

    # Return the motion command to be published.
    return fwd_clamped, ang_clamped


def discrete_turn(angle:float):
    """
    Turn the robot by a discrete amount, and then stop.
    @param angle - the angle to turn (radians). Positive for CCW, negative for CW.
    """
    # Determine the amount of time it will take to turn this amount at the max turn speed.
    time_to_turn = abs(angle / g_max_ang_cmd) # rad / (rad/s) = seconds.
    rospy.logwarn("MOT: Determined time_to_move: {:} seconds.".format(time_to_turn))
    start_time = time()
    # Get turn direction.
    turn_dir_sign = angle / abs(angle)
    # Send the command velocity for the duration.
    while time() - start_time < time_to_turn:
        rospy.logwarn("MOT: Elapsed time: {:} seconds.".format(time() - start_time))
        publish_command(0, g_max_ang_cmd * turn_dir_sign)
        rospy.sleep(0.001)
    # When the time has elapsed, send a command to stop.
    publish_command(0, 0)


def discrete_motion(dist:float):
    """
    Move the robot forwards by a discrete distance, and then stop.
    @param dist - distance in meters to move. Positive is forwards, negative for backwards.
    """
    # Determine the amount of time it will take to move the specified distance.
    time_to_move = abs(dist / g_max_fwd_cmd) # rad / (rad/s) = seconds.
    rospy.logwarn("MOT: Determined time_to_move: {:} seconds.".format(time_to_move))
    start_time = time()
    # Get direction of motion.
    motion_sign = dist / abs(dist)
    # Send the command velocity for the duration.
    while time() - start_time < time_to_move:
        rospy.logwarn("MOT: Elapsed time: {:} seconds.".format(time() - start_time))
        publish_command(g_max_fwd_cmd * motion_sign, 0)
        rospy.sleep(0.1)
    # When the time has elapsed, send a command to stop.
    publish_command(0, 0)


# Possible discrete motions that can be commanded.
discrete_motion_types = ["90_LEFT", "90_RIGHT", "FORWARD"]
def random_discrete_action():
    """
    Choose a random action from our list of possible discrete actions, and command it.
    """
    action = discrete_motion_types[randint(0,len(discrete_motion_types)-1)]
    if action == "90_LEFT":
        discrete_turn(radians(90))
    elif action == "90_RIGHT":
        discrete_turn(radians(-90))
    elif action == "FORWARD":
        if random() > g_discrete_forward_skip_probability:
            discrete_motion(g_discrete_forward_dist)



# TODO obstacle avoidance?


def main():
    global cmd_pub, path_pub
    rospy.init_node('motion_planning_node')

    read_params()

    # Read command line args.
    if len(sys.argv) > 1:
        global g_run_test_motion
        g_run_test_motion = sys.argv[1].lower() == "true"
    # if len(sys.argv) > 2:
    #     global g_test_motion_type
    #     g_test_motion_type = sys.argv[2]

    # Subscribe to localization est.
    rospy.Subscriber(g_topic_localization, Vector3, get_localization_est, queue_size=1)
    # Subscribe to goal position in pixels on the map.
    rospy.Subscriber(g_topic_goal, Vector3, get_goal_pos, queue_size=1)
    # Subscribe to (or just read the map from) file.
    rospy.Subscriber(g_topic_occ_map, Image, get_map, queue_size=1)

    # Publish control commands (velocities in m/s and rad/s).
    cmd_pub = rospy.Publisher(g_topic_commands, Twist, queue_size=1)
    # there is a way to command a relative position/yaw motion:
    # python navigation/base_position_control.py --base_planner none --base_controller ilqr --smooth --close_loop --relative_position 1.,1.,1.57 --botname locobot

    # Publish planned path to the goal (for viz).
    path_pub = rospy.Publisher(g_topic_planned_path, Float32MultiArray, queue_size=1)

    # In test mode, start a timer to publish commands to the robot.
    if g_run_test_motion:
        rospy.logwarn("MOT: Running motion_planning_node in test mode, on its own timer.")
        rospy.Timer(rospy.Duration(g_dt), test_timer_callback)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass