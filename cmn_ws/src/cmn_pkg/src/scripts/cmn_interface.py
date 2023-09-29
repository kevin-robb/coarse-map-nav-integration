 #!/usr/bin/env python3

"""
Wrapper for the original CMN Habitat code from Chengguang Xu to work with my custom simulator or a physical robot.
"""

import rospy, rospkg
import os, sys
import numpy as np
from math import degrees
from skimage.transform import resize, rotate

from enum import IntEnum

from scripts.basic_types import PoseMeters
from scripts.map_handler import Simulator, MapFrameManager
from scripts.motion_planner import DiscreteMotionPlanner, MotionPlanner
from scripts.particle_filter import ParticleFilter
from scripts.visualizer import Visualizer

# from scripts.cmn.main_run_cmn import CoarseMapNav
from scripts.cmn_ported import CoarseMapNavDiscrete
from scripts.cmn.utils.Parser import YamlParser
from scripts.cmn.Env.habitat_env import quat_from_angle_axis


class CoarseMapNavInterface():
    # Overarching run modes for the project.
    enable_sim:bool = False
    enable_viz:bool = False
    use_discrete_space:bool = False
    enable_localization:bool = True # Debugging flag; if false, uses ground truth pose from sim instead of estimating pose.

    # Other modules which will be initialized if needed.
    cmn_node:CoarseMapNavDiscrete = None
    # MapFrameManager will allow us to read in the map and do coordinate transforms between px and meters.
    map_frame_manager = None # will be MapFrameManager or Simulator(MapFrameManager).
    # ParticleFilter is for continuous state-space localization.
    particle_filter:ParticleFilter = None
    # MotionPlanner will be used to plan paths and command continuous or discrete motions.
    motion_planner = None # Will be MotionPlanner or DiscreteMotionPlanner(MotionPlanner).
    # Visualizer is only possible when running the sim on host PC.
    visualizer:Visualizer = None # Will be initialized only if enabled.

    current_agent_pose:PoseMeters = PoseMeters(0,0,0) # Current pose of the agent, (x, y, yaw) in meters & radians.


    def __init__(self, enable_sim:bool, use_discrete_space:bool, enable_viz:bool, cmd_vel_pub, enable_localization:bool=True, enable_ml_model:bool=False):
        """
        Initialize all modules needed for this project.
        @param enable_sim Flag to use the simulator to generate ground truth observations.
        @param use_discrete_space Flag to use the discrete version of this project rather than continuous.
        @param enable_viz Flag to show a live visualization of the simulation running.
        @param cmd_vel_pub rospy publisher for Twist velocities, which the motion planner will use to command motion.
        @param enable_localization (optional, default True) Debug flag to allow disabling localization from running, using ground truth pose for planning.
        @param enable_ml_model (optional, default False) Debug flag to allow disabling the ML model from being loaded. Allows running on a computer with no GPU.
        """
        self.enable_sim = enable_sim
        self.use_discrete_space = use_discrete_space
        self.enable_viz = enable_viz
        self.enable_localization = enable_localization and enable_sim

        # Init the map manager / simulator.
        if enable_sim:
            self.map_frame_manager = Simulator(use_discrete_space)
        else:
            self.map_frame_manager = MapFrameManager(use_discrete_space)
        # We will give reference to map manager to all other modules so they can use the map and perform coordinate transforms.

        # Init the motion planner.
        if use_discrete_space:
            self.motion_planner = DiscreteMotionPlanner()
        else:
            self.motion_planner = MotionPlanner()
        self.motion_planner.set_vel_pub(cmd_vel_pub)
        self.motion_planner.set_map_frame_manager(self.map_frame_manager)
        # Discrete motion commands internally publish velocity commands for the robot and wait for the motion to be complete, which cannot be run without a robot (i.e., in the sim).
        self.motion_planner.wait_for_motion_to_complete = not enable_sim
        # Select a random goal point.
        self.motion_planner.set_goal_point_random()

        # Init the localization module.
        self.particle_filter = ParticleFilter()
        self.particle_filter.set_map_frame_manager(self.map_frame_manager)

        # Create Coarse Map Navigator (CMN), disabling the ML model if flag is set.
        self.cmn_node = CoarseMapNavDiscrete(self.map_frame_manager, self.motion_planner.goal_pos_px.as_tuple(), not enable_ml_model)

        # Init the visualizer only if it's enabled.
        if enable_viz:
            self.visualizer = Visualizer()
            self.visualizer.set_map_frame_manager(self.map_frame_manager)
            self.visualizer.goal_cell = self.motion_planner.goal_pos_px


    def run(self, pano_rgb=None, dt:float=None):
        """
        Run one iteration.
        @param pano_rgb Numpy array containing four color images concatenated horizontally, (front, right, back, left).
        @param dt - Timer period in seconds representing how often commands are sent to the robot. Only used for particle filter propagation.
        """
        current_local_map = None
        if self.enable_sim:
            current_local_map, rect = self.map_frame_manager.get_true_observation()
            # Save this observation for the viz.
            if self.enable_viz:
                self.visualizer.set_observation(current_local_map, rect)

        if not self.use_discrete_space:
            # Run the continuous version of the project.
            if current_local_map is None:
                current_local_map = self.compute_observation_continuous(pano_rgb)
            self.run_particle_filter(current_local_map)
            self.command_motion_continuous(dt)
        else:
            # TODO CMN requires the ground truth yaw?
            if self.enable_sim:
                agent_yaw = self.map_frame_manager.veh_pose_true.yaw
            else:
                agent_yaw = self.current_agent_pose.yaw # This is just whatever we initialized it to...

            # Run discrete CMN.
            action_str = self.cmn_node.run_one_iter(agent_yaw, pano_rgb, current_local_map)
            # Save the data it computed for the visualizer.
            if self.enable_viz:
                self.visualizer.set_observation(self.cmn_node.current_local_map)
            
            # Command the decided action to the robot/sim.
            fwd, ang = self.motion_planner.cmd_discrete_action(action_str)
            # fwd, ang = self.motion_planner.cmd_random_discrete_action()
            # In the simulator, propagate the true vehicle pose by this discrete action.
            if self.enable_sim:
                self.map_frame_manager.propagate_with_dist(fwd, ang)


    def set_new_odom(self, x, y, yaw):
        """
        Get a new odometry message from the robot.
        @param x, y - Position in meters.
        @param yaw - Orientation in radians.
        """
        self.motion_planner.set_odom((x, y, yaw))


    def compute_observation_continuous(self, pano_rgb=None):
        """
        Use the new panoramic RGB measurement to generate an observation grid.
        Use the agent's orientation to axis-align this observation. Note this version of the function allows continuous orientation space.
        @param pano_rgb Numpy array containing four color images concatenated horizontally, (front, right, back, left).
        @return new observation / "current local map" from the ML model.
        """
        # Predict the local occupancy from panoramic RGB images.
        map_obs = self.cmn_node.predict_local_occupancy(pano_rgb)

        # Rotate the egocentric local occupancy to face NORTH.
        # Robot yaw is represented in radians with 0 being right (east), increasing CCW.
        # So, to rotate it to face north, need to rotate by opposite of yaw, plus an additional 90 degrees.
        # NOTE even though the function doc says to provide the amount to rotate CCW, it seems like chengguang's code gives the negative of this.
        map_obs = rotate(map_obs, -degrees(self.current_agent_pose.yaw) + 90.0)
        self.cmn_node.current_local_map = map_obs
        return map_obs


    def run_particle_filter(self, current_local_map):
        """
        Run an iteration of our active localization method to update our current estimated robot pose.
        @param current_local_map - new observation from the ML model (or ground truth from sim).
        """
        if not self.enable_localization:
            # Use the ground-truth agent pose.
            self.current_agent_pose = self.map_frame_manager.veh_pose_true
            return

        # Use the particle filter to get a localization estimate from this observation.
        self.current_agent_pose = self.particle_filter.update_with_observation(current_local_map)
        if self.enable_viz:
            # Convert particle set to pixels for viz.
            self.visualizer.particle_set = self.particle_filter.get_particle_set_px()
            self.visualizer.veh_pose_estimate = self.map_frame_manager.transform_pose_m_to_px(self.current_agent_pose)
            # If using the simulator, also save the ground truth pose for viz.
            if self.enable_sim:
                self.visualizer.veh_pose_true = self.map_frame_manager.transform_pose_m_to_px(self.map_frame_manager.veh_pose_true)
        # Run the PF resampling step.
        self.particle_filter.resample()


    def command_motion_continuous(self, dt:float=None):
        """
        Choose a motion to command. Use this commanded motion to propagate our beliefs forward.
        @param dt - Timer period in seconds representing how often commands are sent to the robot. Only used for particle filter propagation.
        """
        fwd, ang = self.motion_planner.plan_path_to_goal(self.current_agent_pose)
        # If the goal was reached, plan_path_to_goal returns None, None.
        if fwd is None and ang is None:
            rospy.loginfo("Goal is reached, so ending the run loop.")
            exit()
        self.motion_planner.pub_velocity_cmd(fwd, ang)
        if self.enable_sim:
            # Apply motion to the ground truth vehicle pose.
            self.map_frame_manager.propagate_with_vel(fwd, ang)

        if self.enable_localization:
            # Propagate all particles by the commanded motion.
            # TODO compute dt
            self.particle_filter.propagate_particles(fwd * dt, ang * dt)

        # Save the planned path for the viz.
        if self.enable_viz:
            self.visualizer.planned_path = self.motion_planner.path_px_reversed





