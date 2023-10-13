 #!/usr/bin/env python3

"""
Wrapper for the original CMN Habitat code from Chengguang Xu to work with my custom simulator or a physical robot.
"""

import rospy
import numpy as np
from math import degrees
# from skimage.transform import rotate
import cv2, os

from scripts.basic_types import PoseMeters, PosePixels
from scripts.map_handler import Simulator, MapFrameManager
from scripts.motion_planner import DiscreteMotionPlanner, MotionPlanner
from scripts.particle_filter import ParticleFilter
from scripts.visualizer import Visualizer

from scripts.cmn.cmn_ported import CoarseMapNavDiscrete

class CmnConfig():
    # Flag to track whether we are using discrete or continuous state/action space.
    # Should be one of ["continuous", "discrete", "discrete_random"].
    run_mode:str = "continuous"
    # Flag to use the simulator to generate ground truth observations.
    enable_sim:bool = False
    # Flag to show a live visualization of the simulation running.
    enable_viz:bool = False
    # Debug flag to allow disabling the ML model from loading/running, since it can't run on all machines.
    enable_ml_model:bool = False
    # Debug flag to allow disabling localization from running, using ground truth pose for planning.
    enable_localization:bool = True

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

    current_agent_pose:PoseMeters = PoseMeters(0,0,0) # Current estimated pose of the agent, (x, y, yaw) in meters & radians. For discrete case, assume yaw is ground truth (known).
    pano_rgb = None # If we perform an action of turning in-place, we don't need to retake the measurement. Instead, we shift it to be centered on the new orientation, and save it here.

    # Params for saving training/eval data during the run.
    save_training_data:bool = False # Flag to save data when running on robot for later training/evaluation.
    training_data_dirpath:str = None # Location of directory to save data to.
    iteration:int = 0 # Current iteration number. Used for filenames when saving data.


    def __init__(self, config:CmnConfig, cmd_vel_pub):
        """
        Initialize all modules needed for this project.
        @param config Relevant params/flags for how we should run.
        @param cmd_vel_pub rospy publisher for Twist velocities, which the motion planner will use to command motion.
        """
        self.enable_sim = config.enable_sim
        self.use_discrete_space = "discrete" in config.run_mode
        self.enable_viz = config.enable_viz
        self.enable_localization = config.enable_localization and config.enable_sim

        # Init the map manager / simulator.
        if self.enable_sim:
            self.map_frame_manager = Simulator(self.use_discrete_space)
        else:
            self.map_frame_manager = MapFrameManager(self.use_discrete_space)
        # We will give reference to map manager to all other modules so they can use the map and perform coordinate transforms.

        # Init the motion planner.
        if self.use_discrete_space:
            self.motion_planner = DiscreteMotionPlanner()
        else:
            self.motion_planner = MotionPlanner()
        self.motion_planner.set_vel_pub(cmd_vel_pub)
        self.motion_planner.set_map_frame_manager(self.map_frame_manager)
        # Discrete motion commands internally publish velocity commands for the robot and wait for the motion to be complete, which cannot be run without a robot (i.e., in the sim).
        self.motion_planner.wait_for_motion_to_complete = not self.enable_sim
        # Select a random goal point.
        self.motion_planner.set_goal_point_random()

        if not self.use_discrete_space:
            # Init the localization module.
            self.particle_filter = ParticleFilter()
            self.particle_filter.set_map_frame_manager(self.map_frame_manager)

        if self.use_discrete_space or not self.enable_sim:
            # Create Coarse Map Navigator (CMN) node.
            # NOTE For continuous, only need it to process sensor data into local occupancy map.
            self.cmn_node = CoarseMapNavDiscrete(self.map_frame_manager, self.motion_planner.goal_pos_px.as_tuple(), not self.enable_ml_model, "random" in config.run_mode)

        # Init the visualizer only if it's enabled.
        if self.enable_viz:
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
                if self.cmn_node is not None:
                    self.cmn_node.visualizer.current_ground_truth_local_map = current_local_map
                # Also save the ground truth pose for viz.
                self.visualizer.veh_pose_true = self.map_frame_manager.transform_pose_m_to_px(self.map_frame_manager.veh_pose_true)

        if not self.use_discrete_space:
            # Run the continuous version of the project.
            if current_local_map is None:
                current_local_map = self.compute_observation_continuous(pano_rgb)
                # Save this observation for the viz.
                if self.enable_viz:
                    self.visualizer.set_observation(current_local_map)
            self.run_particle_filter(current_local_map)
            self.command_motion_continuous(dt)
        else:
            # TODO CMN requires the ground truth yaw?
            if self.enable_sim:
                agent_yaw = self.map_frame_manager.veh_pose_true.yaw
            else:
                agent_yaw = self.current_agent_pose.yaw # This is just whatever we initialized it to...
                # TODO ensure initialized yaw is correct, and then use robot odom propagation so we always know the ground truth cardinal direction.

            # Run discrete CMN.
            action_str = self.cmn_node.run_one_iter(agent_yaw, pano_rgb, current_local_map)
            
            if pano_rgb is not None:
                # If the action we've chosen to take is a rotation in-place, we don't need to retake the pano RGB measurement next iteration, but can just shift it instead.
                # NOTE pano_rgb = [front, right, back, left].
                width_each_img = pano_rgb.shape[1] // 4
                if action_str == "move_forward":
                    # Will need to retake measurement.
                    self.pano_rgb = None
                elif action_str == "turn_right":
                    # Shift images to the left by one, so "right" becomes "front".
                    self.pano_rgb = np.roll(pano_rgb, shift=-width_each_img, axis=1)
                    # self.pano_rgb = np.concatenate([pano_rgb[:, width_each_img:, :], pano_rgb[:, :width_each_img, :]], axis=1)
                elif action_str == "turn_left":
                    # Shift images to the right by one, so "left" becomes "front".
                    self.pano_rgb = np.roll(pano_rgb, shift=width_each_img, axis=1)
                    # self.pano_rgb = np.concatenate([pano_rgb[:, 3*width_each_img:, :], pano_rgb[:, :3*width_each_img, :]], axis=1)
                    
            # Save the data it computed for the visualizer.
            if self.enable_viz:
                if not self.enable_sim:
                    # The observation was not set yet, since we don't have ground truth. So, use predicted for viz instead.
                    self.visualizer.set_observation(self.cmn_node.current_local_map)

                if self.cmn_node is not None:
                    # Set data for the CMN visualization.
                    self.cmn_node.visualizer.pano_rgb = pano_rgb
                    self.cmn_node.visualizer.current_predicted_local_map = self.cmn_node.current_local_map
                    self.cmn_node.visualizer.predictive_belief_map = self.cmn_node.predictive_belief_map
                    self.cmn_node.visualizer.observation_prob_map = self.cmn_node.observation_prob_map
                    self.cmn_node.visualizer.agent_belief_map = self.cmn_node.updated_belief_map

            # If localization is running, get the veh pose estimate to use.
            if self.cmn_node.agent_pose_estimate_px is not None:
                # Save the localization estimate (and save in the visualizer).
                localization_result_px = PosePixels(self.cmn_node.agent_pose_estimate_px[0], self.cmn_node.agent_pose_estimate_px[1], agent_yaw)
                self.current_agent_pose = self.map_frame_manager.transform_pose_px_to_m(localization_result_px)
                if self.enable_viz:
                    self.visualizer.veh_pose_estimate = localization_result_px

            # Command the decided action to the robot/sim.
            fwd, ang = self.motion_planner.cmd_discrete_action(action_str)
            # fwd, ang = self.motion_planner.cmd_random_discrete_action()
            # In the simulator, propagate the true vehicle pose by this discrete action.
            if self.enable_sim:
                self.map_frame_manager.propagate_with_dist(fwd, ang)

        # Save all desired data for later training/evaluation.
        if self.save_training_data:
            self.iteration += 1
            if pano_rgb is not None:
                cv2.imwrite(os.path.join(self.training_data_dirpath, "pano_rgb_{:03}.png".format(self.iteration)), pano_rgb)
            if current_local_map is not None:
                # local_map_bgr = cv2.cvtColor(current_local_map, cv2.COLOR_GRAY2BGR)
                # cv2.imshow('local_map_to_save', local_map_bgr); cv2.waitKey(0)
                cv2.imwrite(os.path.join(self.training_data_dirpath, "local_map_{:03}.png".format(self.iteration)), current_local_map)


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
        # map_obs = rotate(map_obs, -degrees(self.current_agent_pose.yaw) + 90.0)
        # TODO find a way to do this continuous rotation without using skimage.rotate; may be able to use the other existing submodule.
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





