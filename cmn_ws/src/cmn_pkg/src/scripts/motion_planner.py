 #!/usr/bin/env python3

"""
Functions that will be useful in more than one node in this project.
"""

import rospkg, yaml, cv2, rospy
import numpy as np
from math import sin, cos, remainder, tau, ceil, radians
from random import random, randrange
from random import randint
from time import time
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String, Float32MultiArray

from scripts.cmn_utilities import clamp, MapFrameManager
from scripts.astar import Astar
from scripts.pure_pursuit import PurePursuit


class MotionPlanner:
    """
    Class to send commands to the robot.
    """
    # Publisher that will be defined by a ROS node and set.
    cmd_vel_pub = None

    # Instances of utility classes.
    astar = Astar() # For path planning.
    mfm = MapFrameManager() # For coordinate transforms between localization estimate and the map frame.
    # Goal cell in pixels on the map.
    goal_pos_px = None

    # Types of motion that can be commanded to the robot when in test mode.
    test_motion_types = ["NONE", "RANDOM", "CIRCLE", "STRAIGHT"]
    # Currently active test motion type. None if inactive.
    cur_test_motion_type = None

    # Most recently planned path. Can be accessed for viz. Will be None until a path is successfully planned.
    path_px_reversed = None


    def __init__(self):
        self.read_params()

    def read_params(self):
        """
        Read configuration params from the yaml.
        """
        # Determine filepath.
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('cmn_pkg')
        # Open the yaml and get the relevant params.
        with open(pkg_path+'/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            # Constraints.
            self.max_fwd_cmd = config["constraints"]["fwd"]
            self.max_ang_cmd = config["constraints"]["ang"]
            # Path planning.
            self.do_path_planning = config["path_planning"]["do_path_planning"]
            PurePursuit.use_finite_lookahead_dist = self.do_path_planning

    def set_vel_pub(self, pub):
        """
        Set our publisher for velocity commands, which are sent to the robot and used for updating viz.
        """
        self.cmd_vel_pub = pub

    def pub_velocity_cmd(self, fwd, ang):
        """
        Clamp a velocity command within valid values, and publish it to the vehicle.
        """
        # Clamp to allowed velocity ranges.
        fwd = clamp(fwd, 0, self.max_fwd_cmd)
        ang = clamp(ang, -self.max_ang_cmd, self.max_ang_cmd)
        rospy.loginfo("MP: Publishing a command ({:}, {:})".format(fwd, ang))
        # Create ROS message.
        msg = Twist(Vector3(fwd, 0, 0), Vector3(0, 0, ang))
        self.cmd_vel_pub.publish(msg)

    def set_test_motion_type(self, type:str):
        """
        Set the type of motion that will be constantly commanded to the robot.
        @param type - motion type to command. Must be in self.test_motion_types.
        """
        if type not in self.test_motion_types:
            rospy.logwarn("MP: Cannot set invalid test motion type {:}. Setting to NONE".format(type))
            self.cur_test_motion_type = "NONE"
        else:
            self.cur_test_motion_type = type

    def cmd_test_motion(self):
        """
        Only runs in test mode.
        Publish test motion every iteration.
        """
        fwd, ang = 0.0, 0.0
        if self.cur_test_motion_type == "NONE":
            fwd, ang = 0.0, 0.0 # No motion.
        elif self.cur_test_motion_type == "CIRCLE":
            fwd, ang = self.max_fwd_cmd, self.max_ang_cmd # Arc motion.
        elif self.cur_test_motion_type == "STRAIGHT":
            fwd, ang = self.max_fwd_cmd, 0.0 # Forward-only motion.
        elif self.cur_test_motion_type == "RANDOM":
            rospy.logwarn("MP: Test motion type RANDOM is not yet implemented. Sending zero velocity.")

        # Send the motion to the robot.
        self.pub_velocity_cmd(fwd, ang)

    def set_map(self, occ_map):
        """
        Set the occupancy grid map which will be used for path planning.
        """
        self.mfm.set_map(occ_map)
        # NOTE some processing happens in set_map, so use the result that was saved to keep A* and mfm in sync w/o duplicate operations.
        self.astar.map = self.mfm.map

    def set_goal_point(self, goal_cell):
        """
        Set the goal point in pixels on the map, which we will try to go to.
        """
        rospy.loginfo("MP: Got goal cell ({:}, {:})".format(int(goal_cell[0]), int(goal_cell[1])))
        self.goal_pos_px = goal_cell

    def plan_path_to_goal(self, veh_pose_est):
        """
        Use A* to generate a path to the current goal cell, starting at the current localization estimate.
        @param veh_pose_est, 3x1 numpy array of localization estimate (x,y,yaw) in meters.
        @return fwd, ang - velocities to command.
        """
        if self.goal_pos_px is None:
            rospy.logerr("MP: Cannot generate a path to the goal cell, since the goal has not been set. Commanding zero velocity.")
            return 0.0, 0.0
        
        # Convert vehicle pose from meters to pixels.
        veh_r, veh_c = self.mfm.transform_map_m_to_px(veh_pose_est[0], veh_pose_est[1])

        if self.do_path_planning:
            # Generate (reverse) path with A*.
            self.path_px_reversed = self.astar.run_astar(veh_r, veh_c, self.goal_pos_px[0], self.goal_pos_px[1])
            # Check if A* failed to find a path.
            if self.path_px_reversed is None:
                rospy.logerr("MOT: No path found by A*. Publishing zeros for motion command.")
                return 0.0, 0.0
        else:
            # Just use the goal point as the "path" and naively steer directly towards it.
            self.path_px_reversed = [self.goal_pos_px, (veh_r, veh_c)]
            
        # rospy.loginfo("MOT: Planned path from A*: " + str(self.path_px_reversed))
        # Turn this path from px to meters and un-reverse it.
        path = []
        for i in range(len(self.path_px_reversed)-1, -1, -1):
            path.append(self.mfm.transform_map_px_to_m(self.path_px_reversed[i][0], self.path_px_reversed[i][1]))
            # Check if the path contains any occluded cells.
            if self.mfm.map[self.path_px_reversed[i][0], self.path_px_reversed[i][1]] == 0:
                rospy.logwarn("MOT: Path contains an occluded cell.")

        # Set the path for pure pursuit, and generate a command.
        PurePursuit.path_meters = path
        fwd, ang = PurePursuit.compute_command(veh_pose_est)

        # Keep within constraints.
        fwd_clamped = clamp(fwd, 0, self.max_fwd_cmd)
        ang_clamped = clamp(ang, -self.max_ang_cmd, self.max_ang_cmd)
        if fwd != fwd_clamped or ang != ang_clamped:
            rospy.logwarn("MOT: Clamped pure pursuit output from ({:.2f}, {:.2f}) to ({:.2f}, {:.2f}).".format(fwd, ang, fwd_clamped, ang_clamped))

        # Return the commanded motion to be published.
        return fwd_clamped, ang_clamped




class DiscreteMotionPlanner(MotionPlanner):
    """
    Class to command discrete actions to the robot.
    """
    # Define allowable discrete actions.
    discrete_actions = ["90_LEFT", "90_RIGHT", "FORWARD"]
    # Publisher that will be defined by a ROS node and set.
    discrete_action_pub = None

    def __init__(self):
        self.read_params()

    def read_params(self):
        """
        Read configuration params from the yaml.
        """
        super().read_params()
        # Determine filepath.
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('cmn_pkg')
        # Open the yaml and get the relevant params.
        with open(pkg_path+'/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            # Params for discrete actions.
            # g_use_discrete_actions = config["actions"]["use_discrete_actions"]
            self.discrete_forward_dist = abs(config["actions"]["discrete_forward_dist"])
            self.discrete_forward_skip_probability = config["actions"]["discrete_forward_skip_probability"]
            if self.discrete_forward_skip_probability < 0.0 or self.discrete_forward_skip_probability > 1.0:
                rospy.logwarn("DMP: Invalid value of discrete_forward_skip_probability. Must lie in range [0, 1]. Setting to 0.")
                self.discrete_forward_skip_probability = 0

    def set_discrete_action_pub(self, pub):
        """
        Set our publisher for discrete actions.
        """
        self.discrete_action_pub = pub

    def cmd_discrete_action(self, action:str):
        """
        Command a discrete action.
        @param action - str representing a defined discrete action.
        @return fwd, ang distances moved, which will allow us to propagate our simulated robot pose.
        """
        if action not in self.discrete_actions:
            rospy.logwarn("DMP: Invalid discrete action {:} cannot be commanded.".format(action))
            return 0.0, 0.0

        if action == "90_LEFT":
            self.cmd_discrete_ang_motion(radians(90))
            return 0.0, radians(90)
        elif action == "90_RIGHT":
            self.cmd_discrete_ang_motion(radians(-90))
            return 0.0, radians(-90)
        elif action == "FORWARD":
            # Forward motions have a chance to not occur when commanded.
            if random() < self.discrete_forward_skip_probability:
                rospy.logwarn("DMP: Fwd motion requested, but skipping.")
                return 0.0, 0.0
            self.cmd_discrete_fwd_motion(self.discrete_forward_dist)
            return self.discrete_forward_dist, 0.0
    
    def cmd_random_discrete_action(self):
        """
        Choose a random action from our list of possible discrete actions, and command it.
        @return fwd, ang distances moved, which will allow us to propagate our simulated robot pose.
        """
        return self.cmd_discrete_action(self.discrete_actions[randint(0,len(self.discrete_actions)-1)])

    def cmd_discrete_ang_motion(self, angle:float):
        """
        Turn the robot in-place by a discrete amount, and then stop.
        @param angle - the angle to turn (radians). Positive for CCW, negative for CW.
        """
        # Determine the amount of time it will take to turn this amount at the max turn speed.
        time_to_turn = abs(angle / self.max_ang_cmd) # rad / (rad/s) = seconds.
        rospy.logwarn("DMP: Determined time_to_move: {:} seconds.".format(time_to_turn))
        start_time = time()
        # Get turn direction.
        turn_dir_sign = angle / abs(angle)
        # Send the command velocity for the duration.
        # NOTE may want to only send once and use sleep() to wait the whole duration.
        while time() - start_time < time_to_turn:
            rospy.logwarn("DMP: Elapsed time: {:} seconds.".format(time() - start_time))
            self.pub_velocity_cmd(0, self.max_ang_cmd * turn_dir_sign)
            rospy.sleep(0.001)
        # When the time has elapsed, send a command to stop.
        self.pub_velocity_cmd(0, 0)

    def cmd_discrete_fwd_motion(self, dist:float):
        """
        Move the robot forwards by a discrete distance, and then stop.
        @param dist - distance in meters to move. Positive is forwards, negative for backwards.
        """
        # Determine the amount of time it will take to move the specified distance.
        time_to_move = abs(dist / self.max_fwd_cmd) # rad / (rad/s) = seconds.
        rospy.logwarn("DMP: Determined time_to_move: {:} seconds.".format(time_to_move))
        start_time = time()
        # Get direction of motion.
        motion_sign = dist / abs(dist)
        # Send the command velocity for the duration.
        # NOTE may want to only send once and use sleep() to wait the whole duration.
        while time() - start_time < time_to_move:
            rospy.logwarn("DMP: Elapsed time: {:} seconds.".format(time() - start_time))
            self.pub_velocity_cmd(self.max_fwd_cmd * motion_sign, 0)
            rospy.sleep(0.1)
        # When the time has elapsed, send a command to stop.
        self.pub_velocity_cmd(0, 0)
