 #!/usr/bin/env python3

"""
Functions that will be useful in more than one node in this project.
"""

import rospkg, yaml, rospy
from math import radians, pi, sqrt, remainder, tau
from random import random, randint
from time import time
from geometry_msgs.msg import Twist, Vector3

from scripts.map_handler import clamp, MapFrameManager
from scripts.astar import Astar
from scripts.pure_pursuit import PurePursuit
from scripts.basic_types import PoseMeters, PosePixels, yaw_to_cardinal_dir, cardinal_dir_to_yaw

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
    move_goal_after_reaching:bool = False # If true, choose a new goal cell when the goal is reached.
    # Publisher that will be defined by a ROS node and set.
    cmd_vel_pub = None

    obstacle_in_front_of_robot:bool = False # Flag that will be set when the LiDAR detects an obstacle in front of the robot, so we can immediately halt motion.

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

    # Current robot odometry.
    odom = PoseMeters(0,0,0)


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
            self.min_lin_vel = config["constraints"]["min_lin_vel"]
            self.max_lin_vel = config["constraints"]["max_lin_vel"]
            self.min_ang_vel = config["constraints"]["min_ang_vel"]
            self.max_ang_vel = config["constraints"]["max_ang_vel"]
            # Path planning.
            self.do_path_planning = config["path_planning"]["do_path_planning"]
            self.pure_pursuit.use_finite_lookahead_dist = self.do_path_planning
            # Other configs.
            self.move_goal_after_reaching = config["move_goal_after_reaching"]

    def set_vel_pub(self, pub):
        """
        Set our publisher for velocity commands, which are sent to the robot and used for updating viz.
        """
        self.cmd_vel_pub = pub

    def set_odom(self, odom_pose:PoseMeters):
        """
        Update our motion progress based on a new odometry measurement.
        @param odom_pose - pose containing x,y in meters, yaw in radians.
        """
        self.odom = odom_pose

    def pub_velocity_cmd(self, fwd, ang):
        """
        Clamp a velocity command within valid values, and publish it to the vehicle.
        """
        # Clamp to allowed velocity ranges.
        fwd = clamp(fwd, 0, self.max_lin_vel)
        ang = clamp(ang, -self.max_ang_vel, self.max_ang_vel)
        if self.verbose:
            rospy.loginfo("MP: Publishing a command ({:}, {:})".format(fwd, ang))
        
        if self.cmd_vel_pub is not None: # May be undefined when using sim.
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
            fwd, ang = self.max_lin_vel, self.max_ang_vel # Arc motion.
        elif self.cur_test_motion_type == "STRAIGHT":
            fwd, ang = self.max_lin_vel, 0.0 # Forward-only motion.
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
        self.astar.map = self.mfm.map_with_border.copy()

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

        # Check if we have already arrived at the goal.
        if veh_pose_est_px.distance(self.goal_pos_px) < 2:
            # We are within 2 pixels of the goal, so declare it reached.
            # NOTE It is fine to check this in pixels instead of meters, since pixels is our lowest resolution for path planning.
            return None, None

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
            if self.mfm.map_with_border[self.path_px_reversed[i].r, self.path_px_reversed[i].c] == 0:
                if self.verbose:
                    rospy.logwarn("MOT: Path contains an occluded cell.")

        # Set the path for pure pursuit, and generate a command.
        fwd, ang = self.pure_pursuit.compute_command(veh_pose_est, path)

        # Keep within constraints.
        fwd_clamped = clamp(fwd, 0, self.max_lin_vel)
        ang_clamped = clamp(ang, -self.max_ang_vel, self.max_ang_vel)
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
    

class PController:
    """
    PID wihout the I and D, to perform some extremely basic motion control.
    """
    # Value on previous iteration.
    last_v:float = None
    # P-coefficient for the filter.
    p = 0.1

    def __init__(self, init_value:float=0.0, p:float=0.1):
        """
        For our case (linear velocity control), we will be stationary at the start of each discrete motion.
        @param init_value - Initial filter value.
        @param p - P-coefficient for the filter.
        """
        self.last_v = init_value
        self.p = p

    def update(self, target_v:float):
        """
        Update the filter with a new target value.
        """
        self.last_v = target_v * self.p + self.last_v * (1 - self.p)
        return self.last_v

class DiscreteMotionPlanner(MotionPlanner):
    """
    Class to command discrete actions to the robot.
    """
    # Define allowable discrete actions.
    discrete_actions = ["turn_left", "turn_right", "move_forward"]
    # Create MotionTracker instance.
    motion_tracker = MotionTracker()
    # Flag to keep publishing commands until the robot odometry indicates the motion has finished.
    # Since the sim does not publish robot odom or subscribe to these velocity commands, this is a simpler solution.
    wait_for_motion_to_complete:bool = True
    # Flag to command all pivots relative to global frame, instead of relative to current yaw.
    command_pivots_globally:bool = True
    # When commanding a discrete motion, wait until these conditions are satisfied.
    lin_goal_reach_deviation:float = None
    ang_goal_reach_deviation:float = None

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
            self.discrete_forward_dist = abs(config["actions"]["discrete_forward_dist"])
            self.lin_goal_reach_deviation = abs(config["goal_reach_deviation"]["linear"])
            self.ang_goal_reach_deviation = radians(abs(config["goal_reach_deviation"]["angular"]))

    def cmd_discrete_action(self, action:str):
        """
        Command a discrete action.
        @param action - str representing a defined discrete action.
        @return fwd, ang distances moved, which will allow us to propagate our simulated robot pose.
        """
        # Convert string to twist values.
        fwd = self.discrete_forward_dist if action == "move_forward" else 0.0
        ang = radians(-90.0 if action == "turn_right" else (90.0 if action == "turn_left" else 0.0))
        # Only command the motion and wait for it to finish if we're using a physical robot.
        if self.wait_for_motion_to_complete:
            rospy.loginfo("DMP: Commanding discrete action {:}.".format(action))
            if action in ["turn_left", "turn_right"]:
                if self.command_pivots_globally:
                    self.cmd_discrete_ang_motion_global(ang)
                else:
                    self.cmd_discrete_ang_motion_relative(ang)
            elif action == "move_forward":
                self.cmd_discrete_fwd_motion(fwd)
            else:
                rospy.logwarn("DMP: Invalid discrete action {:} cannot be commanded.".format(action))
                fwd, ang = 0.0, 0.0
            # When the motion has finished, send a command to stop.
            self.pub_velocity_cmd(0, 0)
            # Insert a small pause to help differentiate adjacent discrete motions.
            rospy.sleep(0.5)
        return fwd, ang
    
    def cmd_random_discrete_action(self):
        """
        Choose a random action from our list of possible discrete actions, and command it.
        @return fwd, ang distances moved, which will allow us to propagate our simulated robot pose.
        """
        return self.cmd_discrete_action(self.discrete_actions[randint(0,len(self.discrete_actions)-1)])

    def cmd_discrete_ang_motion_global(self, angle:float):
        """
        Turn the robot in-place by a discrete amount, and then stop.
        Do our best to stay axis-locked to a cardinal direction, so modify turn angle to achieve this.
        @param angle - the angle to turn (radians). Positive for CCW, negative for CW.
        """
        # Get desired final orientation after the pivot.
        final_dir:str = yaw_to_cardinal_dir(self.odom.yaw + angle)
        # Command the robot to turn to this direction.
        self.cmd_pivot_to_face_direction(final_dir)

    def cmd_pivot_to_face_direction(self, final_dir:str):
        """
        Command a pivot to turn the robot in-place to align with the specified cardinal direction.
        @param final_dir - desired cardinal direction, one of "east", "west", "north", "south".
        """
        # Convert this to yaw.
        final_yaw:float = cardinal_dir_to_yaw[final_dir]
        # Compute actual amount we will turn.
        actual_amount_to_turn = remainder(final_yaw - self.odom.yaw, tau)

        if self.verbose:
            rospy.loginfo("DMP: Commanding a discrete pivot from {:} to {:}. starting yaw: {:.3f}, goal yaw: {:.3f}".format(self.odom.get_direction(), final_dir, self.odom.yaw, final_yaw))

        # Get desired turn direction.
        turn_dir_sign = actual_amount_to_turn / abs(actual_amount_to_turn)
        # Keep waiting until motion has completed.
        remaining_turn_rads = abs(actual_amount_to_turn)
        while remaining_turn_rads > self.ang_goal_reach_deviation:
            if actual_amount_to_turn > 0.5:
                # Command the max possible turn speed, in the desired direction.
                # NOTE if we don't "ramp down" the speed, we may over-turn slightly.
                abs_ang_vel_to_cmd = remaining_turn_rads / abs(actual_amount_to_turn) * self.max_ang_vel
            else:
                # If the total amount to turn is small, this method will way over-turn. So, just command minimum speed.
                abs_ang_vel_to_cmd = 0

            abs_ang_vel_to_cmd = max(abs_ang_vel_to_cmd, self.min_ang_vel) # Enforce a minimum speed.
            self.pub_velocity_cmd(0, abs_ang_vel_to_cmd * turn_dir_sign)
            rospy.sleep(0.001)
            # Compute new remaining radians to turn.
            rads_to_go = remainder(final_yaw - self.odom.yaw, tau)
            remaining_turn_rads = abs(rads_to_go)
            # Safety check for direction to goal, in case we pivoted quickly past it and didn't get a measurement in the stopping region.
            if rads_to_go * turn_dir_sign < 0:
                break
            # rospy.loginfo("DMP: remaining_turn_rads is {:.3f}, with current yaw {:.3f}".format(remaining_turn_rads, self.odom.yaw))

    def cmd_discrete_ang_motion_relative(self, angle:float):
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
        # NOTE may still need to reduce 'angle' by a factor of say, 0.8, to prevent over-turning.
        remaining_turn_rads = abs(angle)
        while remaining_turn_rads > self.ang_goal_reach_deviation:
            if angle > 0.5:
                # Command the max possible turn speed, in the desired direction.
                # NOTE if we don't "ramp down" the speed, we may over-turn slightly.
                abs_ang_vel_to_cmd = remaining_turn_rads / abs(angle) * self.max_ang_vel
            else:
                # If the total amount to turn is small, this method will way over-turn. So, just command minimum speed.
                abs_ang_vel_to_cmd = 0
            
            abs_ang_vel_to_cmd = max(abs_ang_vel_to_cmd, self.min_ang_vel) # Enforce a minimum speed.
            self.pub_velocity_cmd(0, abs_ang_vel_to_cmd * turn_dir_sign)
            rospy.sleep(0.001)
            # Compute new remaining radians to turn.
            remaining_turn_rads = abs(angle) - abs(self.motion_tracker.update_for_pivot(self.odom.yaw))

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
        # Init the p-controller, starting at 0 velocity.
        pid = PController(0.0, 0.0001)
        ramp_threshold = 0.5 * dist # Remaining distance threshold at which we will change the set point from max to min speed.
        # Keep waiting until motion has completed.
        remaining_motion = dist - sqrt((self.odom.x-init_odom.x)**2 + (self.odom.y-init_odom.y)**2)
        while remaining_motion > self.lin_goal_reach_deviation and not self.obstacle_in_front_of_robot:
            if remaining_motion > ramp_threshold:
                # Ramp up during the first part of the motion.
                v = pid.update(self.max_lin_vel)
            else:
                # Ramp down in the last part of the motion.
                v = pid.update(self.min_lin_vel)
            # Command this speed in the desired direction.
            self.pub_velocity_cmd(v * motion_sign, -0.008) # Include a small angular component to combat the robot's natural leftward drift.
            # self.pub_velocity_cmd(v * motion_sign, 0)
            rospy.sleep(0.001)
            # Compute new remaining distance to travel. NOTE we do not take absolute value, so if we pass the point we will still stop.
            remaining_motion = dist - sqrt((self.odom.x-init_odom.x)**2 + (self.odom.y-init_odom.y)**2)
        
        self.pub_velocity_cmd(0.0, 0.0) # Stop the robot.
        
        if self.obstacle_in_front_of_robot:
            rospy.logwarn("DMP: Stopping forward motion due to obstacle.")

        # Now that the forward motion has completed, it's possible there was some angular deviation. So, turn slightly to correct this.
        # self.cmd_pivot_to_face_direction(self.odom.get_direction())

        # Tends to drift to the left, so correct this.
        # self.cmd_discrete_ang_motion_relative(-0.02)
