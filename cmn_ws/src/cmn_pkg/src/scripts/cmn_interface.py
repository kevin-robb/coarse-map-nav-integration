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
from scripts.cmn.main_run_cmn import CoarseMapNav
from scripts.map_handler import Simulator, MapFrameManager
from scripts.motion_planner import DiscreteMotionPlanner, MotionPlanner
from scripts.particle_filter import ParticleFilter
from scripts.visualizer import Visualizer

from scripts.cmn.utils.Parser import YamlParser
from scripts.cmn.Env.habitat_env import quat_from_angle_axis


class RunMode(IntEnum):
    DISCRETE_HABITAT = 0
    DISCRETE_SIM = 1
    DISCRETE_TURTLEBOT = 2
    CONTINUOUS_SIM = 3
    CONTINUOUS_TURTLEBOT = 4


class CoarseMapNavInterface():
    # Overarching run modes for the project.
    enable_sim:bool = False
    enable_viz:bool = False
    use_discrete_space:bool = False
    enable_localization:bool = True # Debugging flag; if false, uses ground truth pose from sim instead of estimating pose.

    # Other modules which will be initialized if needed.
    cmn_node:CoarseMapNav = None
    # MapFrameManager will allow us to read in the map and do coordinate transforms between px and meters.
    map_frame_manager = None # will be MapFrameManager or Simulator(MapFrameManager).
    # ParticleFilter is for continuous state-space localization.
    particle_filter:ParticleFilter = None
    # MotionPlanner will be used to plan paths and command continuous or discrete motions.
    motion_planner = None # Will be MotionPlanner or DiscreteMotionPlanner(MotionPlanner).
    # Visualizer is only possible when running the sim on host PC.
    visualizer:Visualizer = None # Will be initialized only if enabled.

    current_observation = None
    current_agent_pose:PoseMeters = None # Current pose of the agent, (x, y, yaw) in meters & radians.


    def __init__(self, enable_sim:bool, use_discrete_space:bool, enable_viz:bool, cmd_vel_pub, yaml_path:str, enable_localization:bool=True):
        """
        Initialize all modules needed for this project.
        @param enable_sim Flag to use the simulator to generate ground truth observations.
        @param use_discrete_space Flag to use the discrete version of this project rather than continuous.
        @param enable_viz Flag to show a live visualization of the simulation running.
        @param cmd_vel_pub rospy publisher for Twist velocities, which the motion planner will use to command motion.
        @param yaml_path Path to config.yaml.
        @param enable_localization (optional) Debug flag to allow disabling localization from running, using ground truth pose for planning.
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

        # Load configurations
        configurations = YamlParser(yaml_path).data["cmn"]
        # Get filepath to cmn directory.
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('cmn_pkg')
        cmn_path = os.path.join(pkg_path, "src/scripts/cmn")
        # Create Coarse Map Navigator (CMN), using the already-processed coarse map.
        self.cmn_node = CoarseMapNav(configurations, cmn_path, self.map_frame_manager.inv_occ_map)

        # Init the visualizer only if it's enabled.
        if enable_viz:
            self.visualizer = Visualizer()
            self.visualizer.set_map_frame_manager(self.map_frame_manager)
            self.visualizer.goal_cell = self.motion_planner.goal_pos_px


    def set_new_odom(self, x, y, yaw):
        """
        Get a new odometry message from the robot.
        @param x, y - Position in meters.
        @param yaw - Orientation in radians.
        """
        self.motion_planner.set_odom((x, y, yaw))


    def compute_new_observation(self, pano_rgb=None):
        """
        Use the new panoramic RGB measurement to generate an observation grid.
        Use the agent's orientation to axis-align this observation.
        @param pano_rgb Numpy array containing four color images concatenated horizontally, (front, right, back, left).
        NOTE: pano_rgb will be None if we're using the simulator instead of getting sensor data.
        """
        if self.enable_sim:
            observation, rect = self.map_frame_manager.get_true_observation()
        else:
            # Predict the local occupancy from panoramic RGB images.
            observation = self.cmn_node.predict_local_occupancy(pano_rgb)
            rect = None # Bounding box for the observation on the map. Used for sim/viz only.

            # Robot yaw is represented in radians with 0 being right (east), increasing CCW.
            agent_yaw = self.current_agent_pose.yaw

            # Rotate the egocentric local occupancy to face NORTH.
            # So, to rotate it to face north, need to rotate by opposite of yaw, plus an additional 90 degrees.
            # NOTE even though the function doc says to provide the amount to rotate CCW, it seems like chengguang's code gives the negative of this.
            map_obs = rotate(map_obs, -degrees(agent_yaw) + 90.0)
            self.cmn_node.current_local_map = map_obs

        self.current_observation = observation

        # Save this observation for the viz.
        if self.enable_viz:
            self.visualizer.set_observation(observation, rect)


    def run_localization(self):
        """
        Run an iteration of our active localization method to update our current estimated robot pose.
        """
        if not self.enable_localization:
            # Use the ground-truth agent pose.
            self.current_agent_pose = self.map_frame_manager.veh_pose_true
            return

        if not self.use_discrete_space:
            # Use the particle filter to get a localization estimate from this observation.
            self.current_agent_pose = self.particle_filter.update_with_observation(self.current_observation)
            if self.enable_viz:
                # Convert particle set to pixels for viz.
                self.visualizer.particle_set = self.particle_filter.get_particle_set_px()
                self.visualizer.veh_pose_estimate = self.map_frame_manager.transform_pose_m_to_px(self.current_agent_pose)
                # If using the simulator, also save the ground truth pose for viz.
                if self.enable_sim:
                    self.visualizer.veh_pose_true = self.map_frame_manager.transform_pose_m_to_px(self.map_frame_manager.veh_pose_true)
            # Run the PF resampling step.
            self.particle_filter.resample()
        else:
            # Measurement update stage.
            self.observation_prob_map = self.cmn_node.measurement_update_func(self.cmn_node.current_local_map)

            # Full Bayesian update with weights.
            log_belief = np.log(self.cmn_node.observation_prob_map + 1e-8) + np.log(self.cmn_node.predictive_belief_map + 1e-8)
            belief = np.exp(log_belief)
            normalized_belief = belief / belief.sum()

            # Record the belief for visualization.
            self.cmn_node.last_agent_belief_map = self.cmn_node.agent_belief_map.copy()
            self.cmn_node.updated_belief_map = normalized_belief.copy()
            self.cmn_node.agent_belief_map = normalized_belief.copy()


    def choose_motion_to_command(self, dt:float=None):
        """
        Choose a motion to command. Use this commanded motion to propagate our beliefs forward.
        @param dt - Timer period in seconds representing how often commands are sent to the robot. Only used for particle filter propagation.
        """
        if not self.use_discrete_space:
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

        else: # discrete
            # Convert robot state to format expected by CMN.
            state = [self.current_agent_pose.x, self.current_agent_pose.y, 0] + quat_from_angle_axis(self.current_agent_pose.yaw, np.array([0, 1, 0]))  # pitch, yaw, roll
            # Select one action using CMN.
            action = self.cmn_node.cmn_planner(state[3:])
            # action = np.random.choice(['move_forward', 'turn_left', 'turn_right'], 1)[0]

            # add noise
            if action == "move_forward":
                # randomly sample a p from a uniform distribution between [0, 1]
                self.noise_trans_prob = np.random.rand()

            # Predictive update stage
            self.predictive_belief_map = self.cmn_node.predictive_update_func(action, self.current_agent_pose.get_direction())



            # TODO for now, just command random discrete action.
            fwd, ang = self.motion_planner.cmd_random_discrete_action()
            # fwd, ang = g_motion_planner.cmd_discrete_action("90_LEFT")
            # fwd, ang = g_motion_planner.cmd_discrete_action("FORWARD")

            # In the simulator, propagate the true vehicle pose by this discrete action.
            if self.enable_sim:
                self.map_frame_manager.propagate_with_dist(fwd, ang)


