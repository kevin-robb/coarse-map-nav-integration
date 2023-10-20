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
from scripts.cmn.tree_search import TreeNode, BFTree
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
from scripts.basic_types import yaw_to_cardinal_dir, PosePixels
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
    send_random_commands:bool = False # Flag to send random discrete actions instead of planning.
    # Configs from yaml:
    forward_step_k:int = None # Tree search param.
    forward_step_meters:float = None # Tree search param.
    local_occ_net_config = None # Dictionary of params for local occ net.

    # Coarse map itself.
    coarse_map_arr = None # 2D numpy array of coarse map. Free=0, Occupied=1.

    # ML model for generating observations.
    device = None # torch.device
    model = None
    transformation = None

    # Define the graph based on the coarse map for shortest path planning
    coarse_map_graph = None

    # Define the variables to store the beliefs
    predictive_belief_map = None  # predictive prob
    observation_prob_map = None  # measurement prob
    updated_belief_map = None  # updated prob
    agent_belief_map = None  # current agent pose belief on the map

    current_local_map = None
    empty_cell_map = None
    goal_map_loc = None
    goal_map_idx = None
    noise_trans_prob = None # Randomly sampled each iteration in range [0,1]. Chance that we don't modify our estimates after commanding a forward motion. Accounts for cell-based representation that relies on scale.

    agent_pose_estimate_px:PosePixels = None # Current localization estimate of the robot pose on the coarse map in pixels. Its yaw is whatever was passed into run_one_iter.

    visualizer:CoarseMapNavVisualizer = CoarseMapNavVisualizer() # Visualizer for all the original CMN discrete stuff.
    use_astar:bool = True # Flag to use A* instead of original CMN planner.
    astar = Astar() # For path planning.


    def __init__(self, mfm:MapFrameManager, goal_cell, skip_load_model:bool=False, send_random_commands:bool=False):
        """
        Initialize the CMN instance.
        @param mfm - Reference to MapFrameManager which has already loaded in the coarse map and processed it by adding a border.
        @param goal_cell - Tuple of goal cell (r,c) on coarse map. If provided, do some more setup with it.
        @param skip_load_model (optional, default False) Flag to skip loading the observation model. Useful to run on a computer w/o nvidia gpu.
        @param send_random_commands (optional, default False) Flag to send random discrete actions instead of planning. Useful for basic demo.
        """
        # If this was initialized with None arguments, it is just being used by a runner and should skip normal setup.
        if mfm is None:
            return

        # Save reference to map frame manager.
        self.mfm = mfm
        # Load the coarse occupancy map: 1.0 for occupied cell and 0.0 for empty cell
        self.coarse_map_arr = self.mfm.inv_map_with_border.copy()
        self.astar.map = self.mfm.map_with_border.copy() # Set the map for A* as well.
        self.visualizer.coarse_map = self.mfm.map_with_border.copy() # Set the map for visualizer as well.
        # Setup filepaths using mfm's pkg path.
        cmn_path = os.path.join(mfm.pkg_path, "src/scripts/cmn")

        self.send_random_commands = send_random_commands

        # Read config params from yaml.
        with open(self.mfm.pkg_path+'/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            # Save as dictionary, same format as original CMN.
            self.forward_step_k = config["path_planning"]["tree_search"]["forward_step_k"]
            self.forward_step_meters = config["path_planning"]["tree_search"]["forward_meter"]
            device_str = config["model"]["device"]     
            self.local_occ_net_config = config["model"]["local_occ_net"]       

        # Load in the ML model, if enabled.
        if not skip_load_model:
            path_to_model = os.path.join(cmn_path, "model/trained_local_occupancy_predictor_model.pt")
            self.load_ml_model(path_to_model, device_str)

        # Create environment (now done in sim).
        # Randomly choose starting position and goal cell (done in sim).
        # TODO set starting position in this class?

        # Setup the goal cell.
        self.goal_map_loc = goal_cell
        self.astar.goal_cell = PosePixels(goal_cell[0], goal_cell[1])
        self.visualizer.goal_cell = PosePixels(goal_cell[0], goal_cell[1])
        if self.coarse_map_arr[self.goal_map_loc[0], self.goal_map_loc[1]] != 0.0:
            # Cell is not free.
            print("Warning: Goal cell given in CMN init() is not free!")

        # Initialize the coarse map graph for path planning
        # Build a graph based on the coarse map for shortest path planning
        self.coarse_map_graph = TopoMap(self.coarse_map_arr, self.mfm.obs_height_px, self.mfm.obs_width_px)
        self.goal_map_idx = self.coarse_map_graph.sampled_locations.index(tuple(self.goal_map_loc))

        # Initialize the coarse map grid beliefs for global localization with Bayesian filtering.
        # Initialize the belief as a uniform distribution over all empty spaces
        init_belief = 1.0 - self.coarse_map_arr
        self.agent_belief_map = init_belief / init_belief.sum()
        # record the space map:
        self.empty_cell_map = 1 - self.coarse_map_arr
        # init other beliefs
        self.predictive_belief_map = np.zeros_like(self.coarse_map_arr)  # predictive probability
        self.observation_prob_map = np.zeros_like(self.coarse_map_arr)  # observation probability
        self.updated_belief_map = np.zeros_like(self.coarse_map_arr)  # updated probability


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
        # When we command a forward motion, the actual robot will always be commanded to move.
        # However, we don't know if this motion is enough to correspond to motion between cells on the coarse map.
        # i.e., the coarse map scale may be so large that it takes several forward motions to achieve a different cell.
        # So, there is a probability here that the forward motion is not carried out in the cell representation.
        if action == "move_forward":
            # randomly sample a p from a uniform distribution between [0, 1]
            self.noise_trans_prob = np.random.rand()

        # Check if the action is able to happen. i.e., if this commanded action will be ignored because of a wall, don't move the predictive belief.
        if action != "move_forward" or not facing_a_wall:
            # Run the predictive update stage.
            self.predictive_update_func(action, yaw_to_cardinal_dir(agent_yaw))

        # Run the measurement update stage.
        self.measurement_update_func()

        # Full Bayesian update with weights for both update stages.
        log_belief = np.log(self.observation_prob_map + 1e-8) + np.log(self.predictive_belief_map + 1e-8)
        belief = np.exp(log_belief)
        normalized_belief = belief / belief.sum()
        # Record this new belief map.
        self.updated_belief_map = normalized_belief.copy()
        self.agent_belief_map = normalized_belief.copy()

        # Update the visualization.
        self.visualizer.current_predicted_local_map = self.current_local_map
        self.visualizer.predictive_belief_map = self.predictive_belief_map
        self.visualizer.observation_prob_map = self.observation_prob_map
        self.visualizer.agent_belief_map = self.updated_belief_map


    def predictive_update_func(self, agent_act:str, agent_dir:str):
        """
        Update the grid-based beliefs using the roll function in python
        @param agent_act: Action commanded to the robot.
        @param agent_dir: String representation of a cardinal direction for the robot orientation.
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
            movable_locations = np.roll(self.empty_cell_map, shift=-shift, axis=axis)
            # mask all wall locations
            movable_locations = np.multiply(self.empty_cell_map, movable_locations)
            # Cells near the walls has value = 1.0 other has value = 2.0
            movable_locations = movable_locations + self.empty_cell_map
            # Find all cells beside the walls in the moving direction
            space_to_wall_cells = np.where(movable_locations == 1.0, 1.0, 0.0)

            # ======= Update the belief =======
            # Obtain the current belief
            current_belief = self.agent_belief_map.copy()

            # Compute the move prob and stay prob
            noise_trans_move_prob = self.noise_trans_prob
            noise_trans_stay_prob = 1 - self.noise_trans_prob

            # Update the belief: probability for staying in the same cell
            pred_stay_belief = np.where(space_to_wall_cells == 1.0,
                                        current_belief,
                                        current_belief * noise_trans_stay_prob)

            # Update the belief: probability for moving to the next cell
            pred_move_belief = np.roll(current_belief, shift=shift, axis=axis)
            pred_move_belief = np.multiply(pred_move_belief, self.empty_cell_map)
            pred_move_belief = pred_move_belief * noise_trans_move_prob

            # Update the belief: combine the two
            pred_belief = pred_stay_belief + pred_move_belief
        else:
            pred_belief = self.agent_belief_map.copy()

        # Update the belief map.
        self.predictive_belief_map = pred_belief


    def measurement_update_func(self):
        """
        Use the current local map from generated observation to update our beliefs.
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
            score = compute_similarity_mse(1 - self.current_local_map, candidate_map)

            # set the observation probability based on similarity
            # still: -1 is dealing with the mismatch
            measurement_prob_map[candidate_loc[0], candidate_loc[1]] = score

            # print(self.current_local_map)
            # print(candidate_map)
            # map_diff = np.abs(self.current_local_map - candidate_map)
            # print("map diff for loc {:}, score: {:.3f}".format(candidate_loc, score))
            # cv2.imshow("map_diff", map_diff)
            # key = cv2.waitKey(0)
            # if key == 113:
            #     cv2.destroyAllWindows()
            #     exit()


        # Normalize it to [0, 1], and save to member var.
        self.observation_prob_map = measurement_prob_map / (np.max(measurement_prob_map) + 1e-8)

        # print(measurement_prob_map)
        # print(np.min(self.observation_prob_map), np.max(self.observation_prob_map))
        # self.observation_prob_map *= 255
        # print(self.observation_prob_map.shape)
        # cv2.imshow("obs map", measurement_prob_map.astype(float)); cv2.waitKey(0)


    def predict_local_occupancy(self, pano_rgb, agent_yaw:float, gt_observation=None):
        """
        Use the model to predict local occupancy map.
        NOTE: Must provide either pano_rgb (sensor data to run model to generate observation) or gt_observation (ground-truth from sim).
        @param pano_rgb - Panorama of four RGB images concatenated together.
        @param agent_yaw - Agent orientation in radians.
        @param gt_observation - Ground-truth observation from simulator. If provided, use it instead of running the model.
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
        else:
            # Scale observation up to 128x128 to match the output from the model.
            pred_local_occ = up_scale_grid(gt_observation)

        # Get cardinal direction corresponding to agent orientation.
        agent_dir_str = yaw_to_cardinal_dir(agent_yaw)
        # Rotate the egocentric local occupancy to face NORTH
        if agent_dir_str == "east":
            pred_local_occ = np.rot90(pred_local_occ, k=-1)
        elif agent_dir_str == "north":
            pass
        elif agent_dir_str == "west":
            pred_local_occ = np.rot90(pred_local_occ, k=1)
        elif agent_dir_str == "south":
            pred_local_occ = np.rot90(pred_local_occ, k=2)
        else:
            raise Exception("Invalid agent direction")
        self.current_local_map = pred_local_occ

        # Update the viz.
        self.visualizer.pano_rgb = pano_rgb
        # If this is ground-truth, assign it to that for viz as well.
        if gt_observation is not None:
            self.visualizer.current_ground_truth_local_map = self.current_local_map


    def cmn_localizer(self, agent_yaw:float):
        """
        Perform localization (discrete bayesian filter) using belief map.
        For localization result, return 3-tuple of:
        @param agent_yaw - Orientation of agent in radians; will be saved as part of localization estimate.
        @return int: index of agent cell in local map.
        @return Tuple[int]: cell location in local map, (r, c).
        @return local occupancy grid map.
        """
        # Find all the locations with max estimated probability
        candidates = np.where(self.agent_belief_map == self.agent_belief_map.max())
        candidates = [[r, c] for r, c in zip(candidates[0].tolist(), candidates[1].tolist())]

        # Randomly sample one as the estimate
        rnd_idx = np.random.randint(low=0, high=len(candidates))
        local_map_loc = tuple(candidates[rnd_idx])

        # Save this localization result for the viz and planner to use.
        self.agent_pose_estimate_px = PosePixels(local_map_loc[0], local_map_loc[1], agent_yaw)
        self.visualizer.current_localization_estimate = self.agent_pose_estimate_px

        # Find its index and the 3 x 3 local occupancy grid
        local_map_idx = self.coarse_map_graph.sampled_locations.index((local_map_loc[0], local_map_loc[1]))

        # Render the local occupancy
        local_map_occ = self.coarse_map_graph.local_maps[local_map_idx]['map_arr']

        return local_map_idx, local_map_loc, local_map_occ


    def choose_next_action(self, agent_yaw:float, true_agent_pose:PosePixels=None) -> str:
        """
        Use the current localization estimate to choose the next discrete action to take.
        @param agent_yaw - Current orientation of the robot in radians.
        @param true_agent_pose (optional) - Ground truth agent pose from sim. If provided, used for A* path planning instead of the estimate.
        @return next action to take.
        """
        # Run localization.
        agent_map_idx, agent_map_loc, agent_local_map = self.cmn_localizer(agent_yaw)

        # Check if the agent has reached the goal location.
        if self.agent_pose_estimate_px.r == self.goal_map_loc[0] and self.agent_pose_estimate_px.c == self.goal_map_loc[1]:
            if self.agent_belief_map.max() > 0.9:
                # We're at the goal cell and the belief map has converged.
                return "stop"
            else:
                # We aren't confident enough in this estimate, so reset the belief map and keep going.
                self.agent_belief_map = self.empty_cell_map / self.empty_cell_map.sum()

        # Check if planning is enabled.
        if self.send_random_commands:
            return np.random.choice(['move_forward', 'turn_left', 'turn_right'], 1)[0]

        # Check which planning method to use.
        if self.use_astar:
            # Use the vehicle pose estimate to plan a path with A*.
            if true_agent_pose is not None:
                action = self.astar.get_next_discrete_action(true_agent_pose)
            else:
                action = self.astar.get_next_discrete_action(self.agent_pose_estimate_px)
            # Save the path for viz.
            self.visualizer.planned_path_to_goal = self.astar.last_path_px_reversed
            return action
        else:
            # Plan a path using Dijkstra's algorithm
            path = self.coarse_map_graph.dijkstra_path(agent_map_idx, self.goal_map_idx)
            if len(path) <= 1:
                return np.random.choice(['move_forward', 'turn_left', 'turn_right'], 1)[0]

            # Compute the heuristic vector
            loc_1 = agent_map_loc
            loc_2 = self.coarse_map_graph.sampled_locations[path[1]]
            heu_vec = compute_norm_heuristic_vec([loc_1[1], loc_1[0]],
                                                    [loc_2[1], loc_2[0]])

            # Do a k-step breadth first search based on the heuristic vector
            root_node = TreeNode([0, 0, 0], agent_yaw)
            breadth_first_tree = BFTree(root_node,
                                        depth=self.forward_step_k,
                                        agent_forward_step=self.forward_step_meters)
            best_child_node = breadth_first_tree.find_the_best_child(heu_vec)

            # retrieve the tree to get the action
            parent = best_child_node.parent
            while parent.parent is not None:
                best_child_node = best_child_node.parent
                parent = best_child_node.parent

            # based on the results select the best action
            return best_child_node.name

