 #!/usr/bin/env python3

"""
Functions that will be useful in more than one node in this project.
"""

import rospkg, yaml, rospy
from math import radians, pi, sqrt
from random import random, randint
from time import time
from geometry_msgs.msg import Twist, Vector3

from scripts.map_handler import clamp, MapFrameManager
from scripts.astar import Astar
from scripts.pure_pursuit import PurePursuit
from scripts.basic_types import PoseMeters, PosePixels

"""
Ideas to try:
    - medial axis graph path planning
    - only allow motion along voronoi lines that stay as far from obstacles as possible
    - allow the vehicle to be in collision since the coarse map could be inaccurate or have extraneous info
    - perhaps a base heuristic of removing small connected components in map to remove furniture and stuff.
"""

class MotionPlanner:
    """
    Class to send commands to the robot.
    """
    verbose = False
    # Publisher that will be defined by a ROS node and set.
    cmd_vel_pub = None

    # Instances of utility classes.
    astar = Astar() # For path planning.
    pure_pursuit = PurePursuit() # For path following.
    mfm = None # For coordinate transforms between localization estimate and the map frame. Will be set after init by runner.
    # Goal cell in pixels on the map.
    goal_pos_px = None # PosePixels instance.

    # Types of motion that can be commanded to the robot when in test mode.
    test_motion_types = ["NONE", "RANDOM", "CIRCLE", "STRAIGHT"]
    # Currently active test motion type. None if inactive.
    cur_test_motion_type = None

    # Most recently planned path. Can be accessed for viz. Will be None until a path is successfully planned.
    path_px_reversed = None # List of PosePixels objects.


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
            self.verbose = config["verbose"]
            self.astar.verbose = self.verbose
            self.pure_pursuit.verbose = self.verbose
            # Constraints.
            self.max_fwd_cmd = config["constraints"]["fwd"]
            self.max_ang_cmd = config["constraints"]["ang"]
            # Path planning.
            self.do_path_planning = config["path_planning"]["do_path_planning"]
            self.pure_pursuit.use_finite_lookahead_dist = self.do_path_planning

    def set_vel_pub(self, pub):
        """
        Set our publisher for velocity commands, which are sent to the robot and used for updating viz.
        """
        self.cmd_vel_pub = pub

    def set_odom(self, odom):
        """
        Update our motion progress based on a new odometry measurement.
        """
        self.odom = odom

    def pub_velocity_cmd(self, fwd, ang):
        """
        Clamp a velocity command within valid values, and publish it to the vehicle.
        """
        # Clamp to allowed velocity ranges.
        fwd = clamp(fwd, 0, self.max_fwd_cmd)
        ang = clamp(ang, -self.max_ang_cmd, self.max_ang_cmd)
        if self.verbose:
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
        Publish one of the test motions every iteration.
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

    def set_map_frame_manager(self, mfm:MapFrameManager):
        """
        Set our reference to the map frame manager, which allows us to use the map and coordinate transform functions to help with path planning.
        @param mfg MapFrameManager instance that has already been initialized with a map.
        """
        self.mfm = mfm
        # Save the map in A* to use as well.
        self.astar.map = self.mfm.map

    def set_goal_point_random(self):
        """
        Select a random free cell on the map to use as the goal.
        """
        self.goal_pos_px = self.mfm.choose_random_free_cell()

    def set_goal_point(self, goal_cell:PosePixels):
        """
        Set the goal point in pixels on the map, which we will try to go to.
        """
        if self.verbose:
            rospy.loginfo("MP: Got goal cell ({:}, {:})".format(int(goal_cell.r), int(goal_cell.c)))
        self.goal_pos_px = goal_cell

    def is_goal_reached(self) -> bool:
        """
        Compute difference between current and goal poses.
        NOTE must use estimated pose to match goal pose's coordinate system.
        """
        # TODO
        return False

    def plan_path_to_goal(self, veh_pose_est:PoseMeters):
        """
        Use A* to generate a path to the current goal cell, starting at the current localization estimate.
        @param veh_pose_est - PoseMeters instance of localization estimate (x,y,yaw).
        @return fwd, ang - velocities to command.
        """
        if self.goal_pos_px is None:
            rospy.logerr("MP: Cannot generate a path to the goal cell, since the goal has not been set. Commanding zero velocity.")
            return 0.0, 0.0
        
        # Convert vehicle pose from meters to pixels.
        veh_pose_est_px = self.mfm.transform_pose_m_to_px(veh_pose_est)

        if self.do_path_planning:
            # Generate (reverse) path with A*.
            self.path_px_reversed = self.astar.run_astar(veh_pose_est_px, self.goal_pos_px)
            # Check if A* failed to find a path.
            if self.path_px_reversed is None:
                rospy.logerr("MOT: No path found by A*. Publishing zeros for motion command.")
                return 0.0, 0.0
        else:
            # Just use the goal point as the "path" and naively steer directly towards it.
            self.path_px_reversed = [self.goal_pos_px, veh_pose_est_px]

        if self.verbose:
            rospy.loginfo("MOT: Planned path from A*: " + ",".join([str(pose) for pose in self.path_px_reversed]))
        # Turn this path from PosePixels to PoseMeters and un-reverse it.
        path = []
        for i in range(len(self.path_px_reversed)-1, -1, -1):
            path.append(self.mfm.transform_pose_px_to_m(self.path_px_reversed[i]))
            # Check if the path contains any occluded cells.
            if self.mfm.map[self.path_px_reversed[i].r, self.path_px_reversed[i].c] == 0:
                if self.verbose:
                    rospy.logwarn("MOT: Path contains an occluded cell.")

        # Set the path for pure pursuit, and generate a command.
        fwd, ang = self.pure_pursuit.compute_command(veh_pose_est, path)

        # Keep within constraints.
        fwd_clamped = clamp(fwd, 0, self.max_fwd_cmd)
        ang_clamped = clamp(ang, -self.max_ang_cmd, self.max_ang_cmd)
        if self.verbose and (fwd != fwd_clamped or ang != ang_clamped):
            rospy.logwarn("MOT: Clamped pure pursuit output from ({:.2f}, {:.2f}) to ({:.2f}, {:.2f}).".format(fwd, ang, fwd_clamped, ang_clamped))

        # Return the commanded motion to be published.
        return fwd_clamped, ang_clamped



class MotionTracker:
    """
    Utility to keep track of how far the robot has pivoted since being initialized.
    """
    # Odom on previous iteration.
    last_yaw = None
    # Progress tracker.
    ang_motion = 0.0

    def __init__(self):
        """
        Do not initialize until we have a starting odom.
        """
        pass

    def reset(self):
        """
        Wipe everything to be able to use the class again.
        """
        self.last_yaw = None
        self.ang_motion = 0.0

    def update_for_pivot(self, new_yaw):
        """
        Given a new angle from odom, update our pivot tracker.
        """
        dtheta = 0.0
        if self.last_yaw is not None:
            dtheta = new_yaw - self.last_yaw
            # Handle pi = -pi boundary.
            if abs(dtheta) > pi:
                dtheta = 2*pi - abs(new_yaw) - abs(self.last_yaw)
                # Fix sign difference.
                if self.last_yaw < 0 and new_yaw > 0:
                    dtheta = -dtheta # clockwise.
        self.last_yaw = new_yaw
        self.ang_motion += dtheta
        return self.ang_motion


class DiscreteMotionPlanner(MotionPlanner):
    """
    Class to command discrete actions to the robot.
    """
    # Define allowable discrete actions.
    discrete_actions = ["90_LEFT", "90_RIGHT", "FORWARD"]
    # Create MotionTracker instance.
    motion_tracker = MotionTracker()

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

    def cmd_discrete_action(self, action:str):
        """
        Command a discrete action.
        @param action - str representing a defined discrete action.
        @return fwd, ang distances moved, which will allow us to propagate our simulated robot pose.
        """
        if action == "90_LEFT":
            # NOTE under-command the angle slightly since we tend to over-turn.
            self.cmd_discrete_ang_motion(radians(82))
            return 0.0, radians(90)
        elif action == "90_RIGHT":
            self.cmd_discrete_ang_motion(radians(-82))
            return 0.0, radians(-90)
        elif action == "FORWARD":
            # Forward motions have a chance to not occur when commanded.
            if random() < self.discrete_forward_skip_probability:
                rospy.logwarn("DMP: Fwd motion requested, but skipping.")
                return 0.0, 0.0
            self.cmd_discrete_fwd_motion(self.discrete_forward_dist)
            return self.discrete_forward_dist, 0.0
        else:
            rospy.logwarn("DMP: Invalid discrete action {:} cannot be commanded.".format(action))
            return 0.0, 0.0
    
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
        if self.verbose:
            rospy.loginfo("DMP: Commanding a discrete pivot of {:} radians.".format(angle))
        # Get turn direction.
        turn_dir_sign = angle / abs(angle)
        # Keep waiting until motion has completed.
        self.motion_tracker.reset()
        while abs(self.motion_tracker.update_for_pivot(self.odom[2])) < abs(angle):
            # Command the max possible turn speed, in the desired direction.
            # NOTE since there is no "ramping down" in the speed, we may over-turn slightly.
            self.pub_velocity_cmd(0, self.max_ang_cmd * turn_dir_sign)
            rospy.sleep(0.001)
        # When the motion has finished, send a command to stop.
        self.pub_velocity_cmd(0, 0)
        # Insert a small pause to help differentiate adjacent discrete motions.
        rospy.sleep(0.5)

    def cmd_discrete_fwd_motion(self, dist:float):
        """
        Move the robot forwards by a discrete distance, and then stop.
        @param dist - distance in meters to move. Positive is forwards, negative for backwards.
        """
        if self.verbose:
            rospy.loginfo("DMP: Commanding a discrete forward motion of {:} meters.".format(dist))
        # Get direction of motion.
        motion_sign = dist / abs(dist)
        # Save the starting odom.
        init_odom = self.odom
        # Keep waiting until motion has completed.
        while sqrt((self.odom[0]-init_odom[0])**2 + (self.odom[1]-init_odom[1])**2) < dist:
            # Command the max possible move speed, in the desired direction.
            # NOTE since there is no "ramping down" in the speed, we may move slightly further than intended.
            self.pub_velocity_cmd(self.max_fwd_cmd * motion_sign, 0)
            rospy.sleep(0.001)
        # When the motion has finished, send a command to stop.
        self.pub_velocity_cmd(0, 0)
        # Insert a small pause to help differentiate adjacent discrete motions.
        rospy.sleep(0.5)

