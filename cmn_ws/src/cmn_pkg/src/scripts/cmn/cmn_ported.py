#!/usr/bin/env python3

"""
Functions ported from Chengguang Xu's original CoarseMapNav class.
"""

import yaml, rospy, os
import numpy as np

# CMN related
from scripts.cmn.model.local_occupancy_predictor import LocalOccNet
from scripts.cmn.tree_search import TreeNode, BFTree
from scripts.cmn.topo_map import TopoMap, compute_similarity_iou, up_scale_grid, compute_similarity_mse

from scripts.cmn.cmn_visualizer import CoarseMapNavVisualizer

# Pytorch related
import torch
from torchvision.transforms import Compose, Normalize, PILToTensor

# Image process related
from PIL import Image
from skimage.transform import rotate

from scripts.map_handler import MapFrameManager
from scripts.basic_types import yaw_to_cardinal_dir


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
    device_str:str = None # Name of device to use for ML model.
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
    last_agent_belief_map = None  # last agent pose belief on the map

    current_local_map = None
    empty_cell_map = None
    goal_map_loc = None
    goal_map_idx = None
    noise_trans_prob = None # Randomly sampled each iteration in range [0,1]. Chance that we don't modify our estimates after commanding a forward motion. Accounts for cell-based representation that relies on scale.

    agent_pose_estimate_px = None # Current localization estimate of the robot pose on the coarse map in pixels.

    visualizer:CoarseMapNavVisualizer = None # Visualizer for all the original CMN discrete stuff.


    def __init__(self, mfm:MapFrameManager, goal_cell, skip_load_model:bool=False, send_random_commands:bool=False):
        """
        Initialize the CMN instance.
        @param mfm - Reference to MapFrameManager which has already loaded in the coarse map and processed it by adding a border.
        @param goal_cell - Tuple of goal cell (r,c) on coarse map. If provided, do some more setup with it.
        @param skip_load_model (optional, default False) Flag to skip loading the observation model. Useful to run on a computer w/o nvidia gpu.
        @param send_random_commands (optional, default False) Flag to send random discrete actions instead of planning. Useful for basic demo.
        """
        # Save reference to map frame manager.
        self.mfm = mfm
        # Load the coarse occupancy map: 1.0 for occupied cell and 0.0 for empty cell
        self.coarse_map_arr = self.mfm.inv_map_with_border
        # Setup filepaths using mfm's pkg path.
        cmn_path = os.path.join(mfm.pkg_path, "src/scripts/cmn")

        self.send_random_commands = send_random_commands

        # Read config params from yaml.
        with open(self.mfm.pkg_path+'/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            # Save as dictionary, same format as original CMN.
            self.forward_step_k = config["path_planning"]["tree_search"]["forward_step_k"]
            self.forward_step_meters = config["path_planning"]["tree_search"]["forward_meter"]
            self.device_str = config["model"]["device"]     
            self.local_occ_net_config = config["model"]["local_occ_net"]       

        # Load in the ML model, if enabled.
        if not skip_load_model:
            # Load the trained local occupancy predictor
            self.device = torch.device(self.device_str)
            # Create the local occupancy network
            model = LocalOccNet(self.local_occ_net_config)
            # Load the trained model
            path_to_model = os.path.join(cmn_path, "model/trained_local_occupancy_predictor_model.pt")
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


        # Create environment (now done in sim).
        # Randomly choose starting position and goal cell (done in sim).
        # TODO set starting position in this class?

        # Setup the goal cell.
        self.goal_map_loc = goal_cell
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
        self.last_agent_belief_map = init_belief / init_belief.sum()
        # record the space map:
        self.empty_cell_map = 1 - self.coarse_map_arr
        # init other beliefs
        self.predictive_belief_map = np.zeros_like(self.coarse_map_arr)  # predictive probability
        self.observation_prob_map = np.zeros_like(self.coarse_map_arr)  # observation probability
        self.updated_belief_map = np.zeros_like(self.coarse_map_arr)  # updated probability

        # Init the visualizer.
        self.visualizer = CoarseMapNavVisualizer((mfm.obs_height_px, mfm.obs_width_px))


    def run_one_iter(self, agent_yaw:float, pano_rgb=None, gt_observation=None) -> str:
        """
        Run one iteration of CMN.
        @param agent_dir_str - Current direction the robot is facing. Should be one of ["north", "west", "south", "east"].
        NOTE: Must provide either pano_rgb (sensor data to run model to generate observation) or observation (ground-truth from sim).
        @param pano_rgb - Dictionary of four RGB images concatenated into a panorama.
        @param gt_observation - (optional) 2D numpy array containing (ground-truth) observation.
        @return str: chosen action, so that our motion planner can command this to the robot.
        """
        if not ((pano_rgb is None) ^ (gt_observation is None)):
            print("Error: Must provide exactly one of pano_rgb and gt_observation.")
            exit(0)

        # Get cardinal direction corresponding to agent orientation.
        agent_dir_str = yaw_to_cardinal_dir(agent_yaw)

        if self.send_random_commands:
            action = np.random.choice(['move_forward', 'turn_left', 'turn_right'], 1)[0]
        else:
            # Select one action using CMN
            action = self.cmn_planner(agent_yaw)

        # Obtain the next sensor measurement --> local observation map (self.current_local_map).
        # TODO cmn_update_beliefs with pano_rgb
        if pano_rgb is not None:
            # Predict the local occupancy from panoramic RGB images.
            map_obs = self.predict_local_occupancy(pano_rgb)

            # Rotate the egocentric local occupancy to face NORTH.
            # Robot yaw is represented in radians with 0 being right (east), increasing CCW.
            # So, to rotate it to face north, need to rotate by opposite of yaw, plus an additional 90 degrees.
            # NOTE even though the function doc says to provide the amount to rotate CCW, it seems like chengguang's code gives the negative of this.
            # map_obs = rotate(map_obs, -degrees(agent_yaw) + 90.0)

            # Rotate the egocentric local occupancy to face NORTH
            if agent_dir_str == "east":
                map_obs = rotate(map_obs, -90)
            elif agent_dir_str == "north":
                pass
            elif agent_dir_str == "west":
                map_obs = rotate(map_obs, 90)
            elif agent_dir_str == "south":
                map_obs = rotate(map_obs, 180)
            else:
                raise Exception("Invalid agent direction")
            self.current_local_map = map_obs
        else:
            self.current_local_map = gt_observation

        # When we command a forward motion, the actual robot will always be commanded to move.
        # However, we don't know if this motion is enough to correspond to motion between cells on the coarse map.
        # i.e., the coarse map scale may be so large that it takes several forward motions to achieve a different cell.
        # So, there is a probability here that the forward motion is not carried out in the cell representation.
        if action == "move_forward":
            # randomly sample a p from a uniform distribution between [0, 1]
            self.noise_trans_prob = np.random.rand()

        # Run the predictive update stage.
        self.predictive_update_func(action, agent_dir_str)

        # Run the measurement update stage.
        self.measurement_update_func()

        # Full Bayesian update with weights for both update stages.
        log_belief = np.log(self.observation_prob_map + 1e-8) + np.log(self.predictive_belief_map + 1e-8)
        belief = np.exp(log_belief)
        normalized_belief = belief / belief.sum()

        # Set data for the visualization.
        self.visualizer.pano_rgb = pano_rgb
        self.visualizer.current_predicted_local_map = self.current_local_map
        # Record the belief for visualization
        self.visualizer.last_agent_belief_map = self.agent_belief_map.copy()
        self.visualizer.updated_belief_map = normalized_belief.copy()
        self.visualizer.agent_belief_map = normalized_belief.copy()

        # Return the chosen action so that our motion planner can command this to the robot.
        return action

        

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
        measurement_prob_map = np.zeros_like(self.coarse_map_arr)

        # Compute the measurement probability for all cells on the map
        for m in self.coarse_map_graph.local_maps:
            # compute the similarity
            candidate_loc = m['loc']
            # TODO why was this being upscaled to 128x128? And how was that being compared to current_local_map, which should be small, like 3x3 or 5x5?
            candidate_map = m['map_arr']
            # candidate_map = up_scale_grid(m['map_arr'])

            # compute the similarity between predicted map and ground truth map
            # score = compute_similarity_iou(self.current_local_map, candidate_map)
            score = compute_similarity_mse(self.current_local_map, candidate_map)

            # set the observation probability based on similarity
            # still: -1 is dealing with the mismatch
            measurement_prob_map[candidate_loc[0], candidate_loc[1]] = score

        # Normalize it to [0, 1], and save to member var.
        self.observation_prob_map = measurement_prob_map / (np.max(measurement_prob_map) + 1e-8)


    def predict_local_occupancy(self, pano_rgb_obs):
        if self.model is not None:
            # Process the observation to tensor
            # Convert observation from ndarray to PIL.Image
            obs = Image.fromarray(pano_rgb_obs)
            # Convert observation from PIL.Image to tensor
            pano_rgb_obs_tensor = self.transformation(obs)

            with torch.no_grad():
                # Predict the local occupancy
                pred_local_occ = self.model(pano_rgb_obs_tensor)
                # Reshape the predicted local occupancy
                pred_local_occ = pred_local_occ.cpu().squeeze(dim=0).squeeze(dim=0).numpy()

            return pred_local_occ
        else:
            # The model was not loaded in. So, print a warning and return a blank observation.
            print("Cannot predict_local_occupancy() because the model was not loaded!")
            return np.zeros((self.mfm.obs_height_px, self.mfm.obs_width_px))


    def cmn_localizer(self):
        """
        Perform localization (discrete bayesian filter) using belief map.
        For localization result, return 3-tuple of:
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

        # Find its index and the 3 x 3 local occupancy grid
        local_map_idx = self.coarse_map_graph.sampled_locations.index((local_map_loc[0], local_map_loc[1]))

        # Render the local occupancy
        local_map_occ = self.coarse_map_graph.local_maps[local_map_idx]['map_arr']

        return local_map_idx, local_map_loc, local_map_occ


    def cmn_planner(self, agent_yaw:float):
        """
        Perform localization and path planning to decide on the next action to take.
        @param agent_yaw - Current orientation of the robot in radians.
        @return next action to take.
        """
        # Render the current map pose estimate using the latest belief
        agent_map_idx, agent_map_loc, agent_local_map = self.cmn_localizer()
        # Save this localization result for the viz to use.
        self.agent_pose_estimate_px = agent_map_loc

        # Check if the agent reaches the goal location
        if agent_map_loc == self.goal_map_loc:
            if self.agent_belief_map.max() > 0.9:
                return "stop"
            else:
                self.agent_belief_map = self.empty_cell_map / self.empty_cell_map.sum()

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
    
