#!/usr/bin/env python3

"""
Functions ported from Chengguang Xu's original CoarseMapNav class.
"""

import yaml, rospy, os, sys
import numpy as np
import cv2

# Add parent dirs to the path so this can be imported by runner scripts.
sys.path.append(os.path.abspath(os.path.join(__file__, "..")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

# CMN related
from scripts.cmn.topo_map import TopoMap, compute_similarity_iou, up_scale_grid, compute_similarity_mse

from scripts.cmn.cmn_visualizer import CoarseMapNavVisualizer

# Image process related
from PIL import Image
# from skimage.transform import rotate

# Pytorch related
from scripts.cmn.model.local_occupancy_predictor import LocalOccNet
import torch
from torchvision.transforms import Compose, Normalize, PILToTensor

from scripts.map_handler import MapFrameManager
from scripts.basic_types import yaw_to_cardinal_dir, PosePixels, rotate_image_to_north, cardinal_dir_to_yaw
from scripts.astar import Astar


def compute_norm_heuristic_vec(loc_1, loc_2):
    arr_1 = np.array(loc_1)
    arr_2 = np.array(loc_2)
    heu_vec = arr_2 - arr_1
    return heu_vec / np.linalg.norm(heu_vec)


class CoarseMapNavDiscrete:
    """
    Original functions from Chengguang Xu, modified to suit the format/architecture of this project.
    """
    # Instances of utility classes.
    mfm:MapFrameManager = None # For coordinate transforms between localization estimate and the map frame. Will be set after init by runner.
    visualizer:CoarseMapNavVisualizer = CoarseMapNavVisualizer() # Visualizer for all the original CMN discrete stuff.
    astar:Astar = Astar() # For path planning.
    send_random_commands:bool = False # Flag to send random discrete actions instead of planning.
    
    enable_sim:bool = False # Flag to know if we're in the simulator vs physical robot.
    fuse_lidar_with_rgb:bool = False # Flag to fuse lidar local occ meas with the predicted (if lidar data exists).
    assume_yaw_is_known:bool = True # If false, CMN will estimate cardinal direction in addition to position.
    # When estimating yaw, it will be easiest to map indexes to orientations.
    ind_to_orientation = ["north", "east", "south", "west"]

    # Coarse map itself.
    coarse_map_arr = None # 2D numpy array of coarse map. Free=1, Occupied=0.
    goal_cell:PosePixels = None # Goal cell on the coarse map.
    agent_pose_estimate_px:PosePixels = None # Current localization estimate of the robot pose on the coarse map in pixels. Its yaw is whatever was passed into run_one_iter.

    # ML model for generating observations.
    local_occ_net_config = None # Dictionary of params for local occ net.
    device = None # torch.device
    model = None
    transformation = None
    
    coarse_map_graph = None # Graph based on coarse map. Only used to get local maps for measurement update step.

    # Define the variables to store the beliefs. All have same dimensions as coarse map.
    predictive_belief_map = None  # predictive prob map. Values range from 0 to 1.
    observation_prob_map = None  # measurement prob map.
    agent_belief_map = None  # current agent pose belief on the map

    current_local_map = None # Current predicted local occupancy map, rotated to align with global coarse map.
    noise_trans_prob = None # Randomly sampled each iteration in range [0,1]. Chance that we don't modify our estimates after commanding a forward motion. Accounts for cell-based representation that relies on scale.
    is_facing_a_wall_in_pred_local_occ:bool = False # Flag to store whether the predicted local occupancy has the robot facing a wall. If so, do not allow forward motion.


    def __init__(self, mfm:MapFrameManager, skip_load_model:bool=False, send_random_commands:bool=False, assume_yaw_is_known:bool=True):
        """
        Initialize the CMN instance.
        @param mfm - Reference to MapFrameManager which has already loaded in the coarse map and processed it by adding a border.
        @param skip_load_model (optional, default False) Flag to skip loading the observation model. Useful to run on a computer w/o nvidia gpu.
        @param send_random_commands (optional, default False) Flag to send random discrete actions instead of planning. Useful for basic demo.
        """
        # If this was initialized with None arguments, it is just being used by a runner and should skip normal setup.
        if mfm is None:
            return

        # Save reference to map frame manager.
        self.mfm = mfm
        # Load the coarse occupancy map: 1.0 for occupied cell and 0.0 for empty cell
        self.coarse_map_arr = self.mfm.map_with_border.astype(int) # free = 1, occ = 0.
        self.astar.map = self.mfm.map_with_border.copy() # Set the map for A* as well.
        self.visualizer.coarse_map = self.mfm.map_with_border.copy() # Set the map for visualizer as well.
        # Setup filepaths using mfm's pkg path.
        cmn_path = os.path.join(mfm.pkg_path, "src/scripts/cmn")

        self.send_random_commands = send_random_commands

        # Read config params from yaml.
        with open(self.mfm.pkg_path+'/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            # Save as dictionary, same format as original CMN.
            device_str = config["model"]["device"]     
            self.local_occ_net_config = config["model"]["local_occ_net"]       

        # Load in the ML model, if enabled.
        if not skip_load_model:
            path_to_model = os.path.join(cmn_path, "model/trained_local_occupancy_predictor_model.pt")
            self.load_ml_model(path_to_model, device_str)

        # Initialize the coarse map graph for extracting local maps.
        self.coarse_map_graph = TopoMap(self.coarse_map_arr, self.mfm.obs_height_px, self.mfm.obs_width_px)

        # Initialize the coarse map grid beliefs for global localization with Bayesian filtering.
        # Initialize the belief as a uniform distribution over all empty spaces
        init_belief = self.coarse_map_arr.copy() # free = 1, occ = 0.
        self.agent_belief_map = init_belief / init_belief.sum()

        self.assume_yaw_is_known = assume_yaw_is_known
        # If we will have to estimate yaw as well, expand the belief map to four layers, one per orientation.
        if not self.assume_yaw_is_known:
            self.agent_belief_map = np.repeat(self.agent_belief_map[:, :, np.newaxis], 4, axis=2)
            # Now this is r by c by 4.

    def set_goal_cell(self, goal_cell:PosePixels):
        """
        Set a new goal cell for CMN, A*, and the visualizer.
        @param goal_cell
        """
        self.goal_cell = goal_cell
        if self.coarse_map_arr[self.goal_cell.r, self.goal_cell.c] != 1.0:
            # Cell is not free.
            rospy.logwarn("CMN: Goal cell given in CMN set_goal_cell() is not free!")
        # Set this goal location in utility subclasses as well.
        self.astar.goal_cell = goal_cell
        self.visualizer.goal_cell = goal_cell


    def load_ml_model(self, path_to_model:str, device_str:str="cpu"):
        """
        Load in the local occupancy predictor ML model.
        @param path_to_model - Filepath the saved model.pt file.
        @param device_str (optional) - Device to use to load model. e.g., "cpu", "cuda:0".
        """
        # Load the trained local occupancy predictor
        self.device = torch.device(device_str)
        # Create the local occupancy network
        model = LocalOccNet(self.local_occ_net_config)
        # Load the trained model
        model.load_state_dict(torch.load(path_to_model, map_location="cpu"))
        # Disable the dropout
        model.eval()
        self.model = model.to(self.device)

        # Define the transformation to transform the observation suitable to the
        # local occupancy model
        # Convert PIL.Image to tensor
        # Convert uint8 to float
        # Convert [0, 255] to [0, 1]
        # Normalize the observation
        # Reshape the tensor by adding the batch dimension
        # Put the tensor to GPU
        self.transformation = Compose([
            PILToTensor(),
            lambda x: x.float(),
            lambda x: x / 255.0,
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            lambda x: x.unsqueeze(dim=0),
            lambda x: x.to(self.device)
        ])


    def update_beliefs(self, action:str, agent_yaw:float, facing_a_wall:bool=False):
        """
        Run one iteration of CMN.
        @param action - Chosen action for this iteration.
        @param agent_yaw - Current robot yaw in radians, with 0=east, increasing CCW. In range [-pi, pi].
        @param facing_a_wall - (optional) If we are facing a wall, we cannot move forwards. So, don't update the belief if we decide to move_forward. This shouldn't be commanded, except in random actions mode.
        """
        # If this is the first iteration, just set up the belief maps and return.
        if self.predictive_belief_map is None:
            self.predictive_belief_map = self.agent_belief_map.copy() # predictive probability
            self.observation_prob_map = np.zeros_like(self.agent_belief_map)  # observation probability

        else:
            # When we command a forward motion, the actual robot will always be commanded to move.
            # However, we don't know if this motion is enough to correspond to motion between cells on the coarse map.
            # i.e., the coarse map scale may be so large that it takes several forward motions to achieve a different cell.
            # So, there is a probability here that the forward motion is not carried out in the cell representation.
            if action == "move_forward":
                if not self.enable_sim:
                    # randomly sample a p from a uniform distribution between [0, 1]
                    self.noise_trans_prob = np.random.rand()
                else:
                    # For the simulator, robot will always move when commanded, and the scale for map vs local occ is guaranteed to match, 
                    # so this just makes the estimate diverge from truth after finding it. So, set the probability of transitioning to the next state to 1.
                    self.noise_trans_prob = 1

            # Check if the action is able to happen. i.e., if this commanded action will be ignored because of a wall, don't move the predictive belief.
            if action != "move_forward" or not facing_a_wall:
                # Run the predictive update stage.
                if self.assume_yaw_is_known:
                    self.predictive_update_func(action, yaw_to_cardinal_dir(agent_yaw))
                    # Update the current localization estimate as well.
                    self.agent_pose_estimate_px.apply_action(action)
                else:
                    if action == "move_forward":
                        for i, dir in enumerate(self.ind_to_orientation):
                            self.predictive_update_func(action, dir, i)
                    else:
                        # Apply a rotation by cycling the layers of the predictive update map.
                        shift = 1 if action == "turn_right" else -1
                        self.predictive_belief_map = np.roll(self.predictive_belief_map, shift=shift, axis=2)


            # Run the measurement update stage.
            if self.assume_yaw_is_known:
                measurement_prob_map = self.measurement_update_func()
                # Normalize it to [0, 1], and save to member var.
                self.observation_prob_map = measurement_prob_map / (np.max(measurement_prob_map) + 1e-8)
            else:
                for i in range(4):
                    self.observation_prob_map[:,:,i] = self.measurement_update_func()
                    # Rotate the local occupancy by 90 degrees for the next orientation estimate.
                    self.current_local_map = np.rot90(self.current_local_map, k=1)

            # Determine the weights for predicted vs observation belief maps.
            # Belief will be computed as this * pred + (1 - this) * obs.
            rel_weight_pred_vs_obs:float = 0.5
            if action == "move_forward" and facing_a_wall:
                # If this happens, it means our localization estimate is probably wrong, since we would never plan to move into a wall.
                # So, blank it out on the predictive belief map.
                # self.predictive_belief_map[self.agent_pose_estimate_px.r, self.agent_pose_estimate_px.c] = 0
                # So, lower the weight of the existing/predicted belief, and increase the belief from current observation.
                # rel_weight_pred_vs_obs = 0.1
                pass

            # Full Bayesian update with weights for both update stages.
            log_belief = np.log((1 - rel_weight_pred_vs_obs) * self.observation_prob_map + 1e-8) + np.log(rel_weight_pred_vs_obs * self.predictive_belief_map + 1e-8)
            belief = np.exp(log_belief)
            normalized_belief = belief / belief.sum()
            # Record this new belief map.
            self.agent_belief_map = normalized_belief.copy()

        # Update the visualization.
        self.visualizer.predictive_belief_map = self.predictive_belief_map
        self.visualizer.observation_prob_map = self.observation_prob_map
        self.visualizer.agent_belief_map = self.agent_belief_map


    def predictive_update_func(self, agent_act:str, agent_dir:str, dir_ind:int=None):
        """
        Update the grid-based beliefs using the roll function in python. 
        @param agent_act: Action commanded to the robot.
        @param agent_dir: String representation of a cardinal direction for the robot orientation.
        @param dir_ind (optional): If we're estimating yaw, this is the index for the channel in the belief map.
        """
        trans_dir_dict = {
            'east': {'shift': 1, 'axis': 1},
            'west': {'shift': -1, 'axis': 1},
            'north': {'shift': -1, 'axis': 0},
            'south': {'shift': 1, 'axis': 0}
        }
        shift = trans_dir_dict[agent_dir]['shift']
        axis = trans_dir_dict[agent_dir]['axis']

        # Apply the movement
        if agent_act == "move_forward":
            # ======= Find all cells beside walls =======
            # The coarse map has value 1 for free space and 0 for occupied space, so we can use it as a base for empty cells.
            movable_locations = np.roll(self.coarse_map_arr, shift=-shift, axis=axis)
            # mask all wall locations
            movable_locations = np.multiply(self.coarse_map_arr, movable_locations)
            # Cells near the walls has value = 1.0 other has value = 2.0
            movable_locations = movable_locations + self.coarse_map_arr
            # Find all cells beside the wallself.predictive_belief_maps in the moving direction
            space_to_wall_cells = np.where(movable_locations == 1.0, 1.0, 0.0)

            # ======= Update the belief =======
            # Obtain the current belief
            if self.assume_yaw_is_known:
                current_belief = self.agent_belief_map.copy()
            else:
                current_belief = self.agent_belief_map[:,:,dir_ind].copy()

            # Compute the move prob and stay prob
            noise_trans_move_prob = self.noise_trans_prob
            noise_trans_stay_prob = 1 - self.noise_trans_prob

            # Update the belief: probability for staying in the same cell
            pred_stay_belief = np.where(space_to_wall_cells == 1.0,
                                        current_belief,
                                        current_belief * noise_trans_stay_prob)

            # Update the belief: probability for moving to the next cell
            pred_move_belief = np.roll(current_belief, shift=shift, axis=axis)
            pred_move_belief = np.multiply(pred_move_belief, self.coarse_map_arr)
            pred_move_belief = pred_move_belief * noise_trans_move_prob

            # Update the belief: combine the two
            pred_belief = pred_stay_belief + pred_move_belief
        else: # Pivot.
            if self.assume_yaw_is_known:
                # No change, since yaw is not estimated.
                pred_belief = self.agent_belief_map.copy()
            else:
                # Apply a rotation by cycling the layers of the predictive update map.
                pred_belief = self.agent_belief_map[dir_ind].copy()

        # Update the belief map.
        if self.assume_yaw_is_known:
            self.predictive_belief_map = pred_belief
        else:
            self.predictive_belief_map[:,:,dir_ind] = pred_belief


    def measurement_update_func(self) -> np.ndarray:
        """
        Use the current local map from generated observation to update our beliefs.
        @return r by c array of probabilities.
        """
        # Define the measurement probability map
        measurement_prob_map = np.zeros_like(self.coarse_map_arr).astype(float)

        # Compute the measurement probability for all cells on the map
        for m in self.coarse_map_graph.local_maps:
            # compute the similarity
            candidate_loc = m['loc']
            # NOTE The model outputs 128x128, so we need to upscale the grid to the same resolution.
            candidate_map = up_scale_grid(m['map_arr'])

            # compute the similarity between predicted map and ground truth map
            # score = compute_similarity_iou(self.current_local_map, candidate_map)
            score = compute_similarity_mse(self.current_local_map, candidate_map)

            # set the observation probability based on similarity
            # still: -1 is dealing with the mismatch
            measurement_prob_map[candidate_loc[0], candidate_loc[1]] = score

        return measurement_prob_map #/ (np.max(measurement_prob_map) + 1e-8)


    def predict_local_occupancy(self, pano_rgb, agent_yaw:float=None, gt_observation=None, lidar_local_occ_meas=None):
        """
        Use the model to predict local occupancy map.
        NOTE: Must provide either pano_rgb (sensor data to run model to generate observation) or gt_observation (ground-truth from sim).
        @param pano_rgb - Panorama of four RGB images concatenated together.
        @param agent_yaw - Agent orientation in radians.
        @param gt_observation - Ground-truth observation from simulator. If provided, use it instead of running the model.
        @param lidar_local_occ_meas - Local occ from LiDAR data. Will be fused with prediction if config is set. Unused otherwise.
        """
        if gt_observation is None:
            if self.model is None:
                rospy.logerr("Cannot predict_local_occupancy() because the model was not loaded!")
                return

            # Process the observation to tensor
            # Convert observation from ndarray to PIL.Image
            obs = Image.fromarray(pano_rgb)
            # Convert observation from PIL.Image to tensor
            pano_rgb_obs_tensor = self.transformation(obs)

            with torch.no_grad():
                # Predict the local occupancy
                pred_local_occ = self.model(pano_rgb_obs_tensor)
                # Reshape the predicted local occupancy
                pred_local_occ = pred_local_occ.cpu().squeeze(dim=0).squeeze(dim=0).numpy()
                # This is now a 128x128 numpy array, with values in range 0 (free) -- 1 (occupied).
                # Model is trained to produce a 1.28 x 1.28 meter predicted local occupancy, so each pixel is 1 cm.

            # Invert the predicted local map so 0 = black = occupied and 1 = white = free.
            pred_local_occ = 1 - pred_local_occ

            # crop_prediction:bool = False
            # if crop_prediction:
            #     # Crop the prediction to only the center region, for which it seems more accurate.
            #     new_width:int = 100
            #     buffer:int = (128 - new_width) // 2
            #     pred_local_occ_cropped = pred_local_occ[buffer:-buffer, buffer:-buffer]
            #     # Resize back up to 128x128.
            #     pred_local_occ = cv2.resize(pred_local_occ_cropped, (128, 128), 0, 0, cv2.INTER_LINEAR)

        else:
            # Scale observation up to 128x128 to match the output from the model.
            pred_local_occ = up_scale_grid(gt_observation)


        # If the agent yaw was not provided, this is just being used by the runner to test the model.
        if agent_yaw is not None or not self.assume_yaw_is_known:
            # Fuse with lidar data if enabled.
            if self.fuse_lidar_with_rgb and lidar_local_occ_meas is not None:
                # LiDAR local occ has robot facing EAST.
                lidar_occ_facing_NORTH = rotate_image_to_north(lidar_local_occ_meas, 0)
                # Combine via elementwise averaging.
                pred_local_occ = np.mean(np.array([pred_local_occ, lidar_occ_facing_NORTH]), axis=0)

            # Before rotating the local map, check if the cell in front of the robot (i.e., top center cell) is occupied (i.e., == 0).
            top_center_cell_block = pred_local_occ[:pred_local_occ.shape[0]//3, pred_local_occ.shape[0]//3:2*pred_local_occ.shape[0]//3]
            top_center_cell_mean = np.mean(top_center_cell_block)
            # print("top_center_cell_mean is {:}".format(top_center_cell_mean))
            self.is_facing_a_wall_in_pred_local_occ = top_center_cell_mean <= 0.75

            if self.assume_yaw_is_known:
                # Rotate the egocentric local occupancy to face NORTH
                pred_local_occ = rotate_image_to_north(pred_local_occ, agent_yaw)
            else:
                # Leave the local map relative to robot, with robot orientation UP.
                pass

        self.current_local_map = pred_local_occ

        # Update the viz.
        self.visualizer.pano_rgb = pano_rgb
        self.visualizer.current_predicted_local_map = self.current_local_map
        # If this is ground-truth, assign it to that for viz as well.
        if gt_observation is not None:
            self.visualizer.current_ground_truth_local_map = self.current_local_map


    def cmn_localizer(self, agent_yaw:float):
        """
        Perform localization (discrete bayesian filter) using belief map. Save result to member variable agent_pose_estimate_px.
        @param agent_yaw - Orientation of agent in radians; will be saved as part of localization estimate.
        """
        if self.assume_yaw_is_known:
            # Find all the locations with max estimated probability
            candidates = np.where(self.agent_belief_map == self.agent_belief_map.max())
            candidates = [[r, c] for r, c in zip(candidates[0].tolist(), candidates[1].tolist())]
            # If the current estimate is still among the max likelihood candidates, keep using it.
            current_est_still_good:bool = self.agent_pose_estimate_px is not None and [self.agent_pose_estimate_px.r, self.agent_pose_estimate_px.c] in candidates
        else:
            # Find all the locations with max estimated probability
            candidates = np.where(self.agent_belief_map == self.agent_belief_map.max())
            candidates = [[r, c, i] for r, c, i in zip(candidates[0].tolist(), candidates[1].tolist(), candidates[2].tolist())]
            # If the current estimate is still among the max likelihood candidates, keep using it.
            current_est_still_good:bool = False
            if self.agent_pose_estimate_px is not None:
                cur_est_yaw_index = self.ind_to_orientation.index(self.agent_pose_estimate_px.get_direction())
                current_est_still_good = [self.agent_pose_estimate_px.r, self.agent_pose_estimate_px.c, cur_est_yaw_index] in candidates

        if not current_est_still_good:
            # Randomly sample one as the estimate
            rnd_idx = np.random.randint(low=0, high=len(candidates))
            local_map_loc = tuple(candidates[rnd_idx])
            # Save this localization result for the planner to use.
            if self.assume_yaw_is_known:
                self.agent_pose_estimate_px = PosePixels(local_map_loc[0], local_map_loc[1], agent_yaw)
            else:
                agent_yaw = cardinal_dir_to_yaw[self.ind_to_orientation[local_map_loc[2]]]
                self.agent_pose_estimate_px = PosePixels(local_map_loc[0], local_map_loc[1], agent_yaw)

        # Save this for the viz.
        self.visualizer.current_localization_estimate = self.agent_pose_estimate_px


    def choose_next_action(self, agent_yaw:float, true_agent_pose:PosePixels=None) -> str:
        """
        Use the current localization estimate to choose the next discrete action to take.
        @param agent_yaw - Current orientation of the robot in radians.
        @param true_agent_pose (optional) - Ground truth agent pose from sim. If provided, used for A* path planning instead of the estimate.
        @return next action to take. Returns "goal_reached" if the estimated robot pose matches the goal cell.
        """
        # Run localization.
        self.cmn_localizer(agent_yaw)

        # Check if the agent has reached the goal location.
        # print("agent pose estimate is {:}, while goal cell is {:}".format(self.agent_pose_estimate_px, self.goal_cell))
        if self.agent_pose_estimate_px.r == self.goal_cell.r and self.agent_pose_estimate_px.c == self.goal_cell.c:
            if self.agent_belief_map.max() > 0.9:
                # We're at the goal cell and the belief map has converged.
                return "goal_reached"
            else:
                # We aren't confident enough in this estimate, so reset the belief map and keep going.
                # The coarse map has value 1 for free cells and 0 for occupied, so this sets equal probability to all free cells.
                self.agent_belief_map = self.coarse_map_arr / self.coarse_map_arr.sum()

        # Check if planning is enabled.
        if self.send_random_commands:
            return np.random.choice(['move_forward', 'turn_left', 'turn_right'], 1)[0]
        
        # If we aren't confident enough in any particular region, just explore a bit first.
        # This is necessary because allowing path planning while the localization estimate is jumping around leads to a lot of turning in place and no convergence.
        # print("self.agent_belief_map.max() is {:}".format(self.agent_belief_map.max()))
        if self.agent_belief_map.max() < 0.05:
            # Explore some more before planning.
            rospy.logwarn("CMN: Localization has not converged enough, so exploring rather than planning a path to the goal.")
            if self.is_facing_a_wall_in_pred_local_occ:
                # return "turn_left"
                return np.random.choice(['turn_left', 'turn_right'], 1)[0]
            else:
                return "move_forward"

        # Use the vehicle pose estimate to plan a path with A*.
        if true_agent_pose is not None:
            action = self.astar.get_next_discrete_action(true_agent_pose)
        else:
            action = self.astar.get_next_discrete_action(self.agent_pose_estimate_px)
        # Save the path for viz.
        self.visualizer.planned_path_to_goal = self.astar.last_path_px_reversed
        return action


