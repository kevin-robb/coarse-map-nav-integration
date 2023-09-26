#!/usr/bin/env python3

"""
Functions ported from Chengguang Xu's original CoarseMapNav class.
"""

import rospkg, yaml, cv2, rospy, os, sys
import numpy as np
from math import sin, cos, remainder, tau, ceil, pi, degrees
from random import random, randrange
from cv_bridge import CvBridge, CvBridgeError

# Allow this to be imported from a higher directory.
parent_package_dir = os.path.abspath(os.path.join(__file__, ".."))
sys.path.append(parent_package_dir)
parent_package_dir = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(parent_package_dir)

# CMN related
from scripts.cmn.utils.Parser import YamlParser
from scripts.cmn.Model.local_occupancy_predictor import LocalOccNet
from scripts.cmn.utils.AnyTree import TreeNode, BFTree
from scripts.cmn.utils.Map import TopoMap, compute_similarity_iou, up_scale_grid, compute_similarity_mse
from scripts.cmn.main_run_cmn import CoarseMapNav
from scripts.cmn.Env.habitat_env import quat_from_angle_axis

# Pytorch related
import torch
from torchvision.transforms import Compose, Normalize, PILToTensor
import torch.nn as nn
import torchvision.models as models

# Image process related
from PIL import Image
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from scripts.map_handler import clamp, MapFrameManager
from scripts.rotated_rectangle_crop_opencv.rotated_rect_crop import crop_rotated_rectangle
from scripts.basic_types import PoseMeters, PosePixels

#### GLOBAL VARIABLES ####


def compute_norm_heuristic_vec(loc_1, loc_2):
    arr_1 = np.array(loc_1)
    arr_2 = np.array(loc_2)
    heu_vec = arr_2 - arr_1
    return heu_vec / np.linalg.norm(heu_vec)


class CoarseMapNavWrapper(CoarseMapNav):
    """
    Original functions from Chengguang Xu, modified to suit the format/architecture of this project.
    """
    # Instances of utility classes.
    mfm:MapFrameManager = None # For coordinate transforms between localization estimate and the map frame. Will be set after init by runner.
    
    configs = None # CMN configs dictionary.

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
    noise_trans_prob = None



    def __init__(self, mfm:MapFrameManager, goal_cell, skip_load_model:bool=False):
        """
        Initialize the CMN instance.
        @param mfm - Reference to MapFrameManager which has already loaded in the coarse map and processed it by adding a border.
        @param goal_cell - Tuple of goal cell (r,c) on coarse map. If provided, do some more setup with it.
        @param skip_load_model (optional, default False) Flag to skip loading the observation model. Useful to run on a computer w/o nvidia gpu.
        """
        # Save reference to map frame manager.
        self.mfm = mfm
        # Load the coarse occupancy map: 1.0 for occupied cell and 0.0 for empty cell
        self.coarse_map_arr = self.mfm.inv_map_with_border
        # Setup filepaths using mfm's pkg path.
        cmn_path = os.path.join(mfm.pkg_path, "src/scripts/cmn")

        # Read config params from yaml.
        with open(self.mfm.pkg_path+'/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            # Save as dictionary, same format as original CMN.
            self.configs = config["cmn"]
            

        # Load in the ML model, if enabled.
        if not skip_load_model:
            # Load the trained local occupancy predictor
            self.device = torch.device(self.configs["device"])
            # Create the local occupancy network
            model = LocalOccNet(self.configs['local_occ_net'])
            # Load the trained model
            path_to_model = os.path.join(cmn_path, "Model/trained_local_occupancy_predictor_model.pt")
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


        # Setup the goal cell.
        self.goal_map_loc = goal_cell
        if self.coarse_map_arr[self.goal_map_loc[0], self.goal_map_loc[1]] != 0.0:
            # Cell is not free.
            print("Warning: Goal cell given in CMN init() is not free!")

        # Initialize the coarse map graph for path planning
        # Build a graph based on the coarse map for shortest path planning
        self.coarse_map_graph = TopoMap(self.coarse_map_arr, self.configs['cmn_cfg']['local_map_size'])
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



        # # ======= Create visualization figures =======
        # self.fig, self.grid = None, None
        # # For observations
        # self.ax_pano_rgb, self.art_pano_rgb = None, None
        # self.ax_local_occ_gt, self.art_local_occ_gt = None, None
        # self.ax_local_occ_pred, self.art_local_occ_pred = None, None
        # self.ax_top_down_view, self.art_top_down_view = None, None
        # # For beliefs
        # self.ax_pred_update_bel, self.art_pred_update_bel = None, None
        # self.ax_obs_update_bel, self.art_obs_update_bel = None, None
        # self.ax_belief, self.art_belief = None, None
        # # Make the plots
        # self.make_plot_figures()


    def run_one_iter(self, current_agent_pose:PoseMeters, pano_rgb=None, gt_observation=None):
        """
        Run one iteration of CMN.
        @param current_agent_pose - Estimate of the current robot pose.
        NOTE: Must provide either pano_rgb (sensor data to run model to generate observation) or observation (ground-truth from sim).
        @param pano_rgb - Dictionary of four RGB images concatenated into a panorama.
        @param gt_observation - (optional) 2D numpy array containing (ground-truth) observation.
        """

        # Convert robot state to format expected by CMN.
        state = [current_agent_pose.x, current_agent_pose.y, 0] + quat_from_angle_axis(current_agent_pose.yaw, np.array([0, 1, 0]))  # pitch, yaw, roll
    
        # Select one action using CMN
        action = self.cmn_planner(state[3:])
        # action = np.random.choice(['move_forward', 'turn_left', 'turn_right'], 1)[0]

        # Obtain the next sensor measurement.
        # TODO cmn_update_beliefs with pano_rgb
        if pano_rgb is not None:
            # Predict the local occupancy from panoramic RGB images.
            observation = self.predict_local_occupancy(pano_rgb)

            # Robot yaw is represented in radians with 0 being right (east), increasing CCW.
            agent_yaw = current_agent_pose.yaw

            # Rotate the egocentric local occupancy to face NORTH.
            # So, to rotate it to face north, need to rotate by opposite of yaw, plus an additional 90 degrees.
            # NOTE even though the function doc says to provide the amount to rotate CCW, it seems like chengguang's code gives the negative of this.
            map_obs = rotate(map_obs, -degrees(agent_yaw) + 90.0)
            self.current_local_map = map_obs
        elif gt_observation is None:
            print("Must provide either pano_rgb or gt_observation arguments.")
            exit(0)
        else:
            observation = gt_observation





    def run_meas_update_step(self):
        """
        Run the measurement update step upon receiving a new observation.
        """

    def run_pred_update_step(self):
        """
        Run the prediction update step.
        """



    def render_local_map_info(self, loc):
        """ Render the local map from the 2-D coarse map """
        # Convert the real world coordinates to map coordinates
        row, col = self.env.map_2d_to_grid_func(loc, self.coarse_map_arr.shape)
        return row, col

    # ======= CMN core functions =======

    def predictive_update_func(self, agent_act, agent_dir):
        # Update the grid-based beliefs using the roll function in python
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

        return pred_belief


    def measurement_update_func(self, obs):
        # Define the measurement probability map
        measurement_prob_map = np.zeros_like(self.coarse_map_arr)

        # Compute the measurement probability for all cells on the map
        for m in self.coarse_map_graph.local_maps:
            # compute the similarity
            candidate_loc = m['loc']
            candidate_map = up_scale_grid(m['map_arr'])

            # compute the similarity between predicted map and ground truth map
            # score = compute_similarity_iou(obs, candidate_map)
            score = compute_similarity_mse(obs, candidate_map)

            # set the observation probability based on similarity
            # still: -1 is dealing with the mismatch
            measurement_prob_map[candidate_loc[0], candidate_loc[1]] = score

        # Normalize it to [0, 1]
        measurement_prob_map = measurement_prob_map / (np.max(measurement_prob_map) + 1e-8)

        return measurement_prob_map


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
            return np.zeros((self.configs['cmn_cfg']['local_map_size'], self.configs['cmn_cfg']['local_map_size']))


    def cmn_localizer(self):
        # Final the locations with max estimated probability
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


    def cmn_planner(self, agent_orientation):
        # Render the current map pose estimate using the latest belief
        agent_map_idx, agent_map_loc, agent_local_map = self.cmn_localizer()

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
        heu_vec = self.compute_norm_heuristic_vec([loc_1[1], loc_1[0]],
                                                  [loc_2[1], loc_2[0]])

        # Do a k-step breadth first search based on the heuristic vector
        relative_location = [0, 0, 0] + agent_orientation
        root_node = TreeNode(relative_location)
        breadth_first_tree = BFTree(root_node,
                                    depth=self.configs['cmn_cfg']['forward_step_k'],
                                    agent_forward_step=self.configs['cmn_cfg']['forward_meter'])
        best_child_node = breadth_first_tree.find_the_best_child(heu_vec)

        # retrieve the tree to get the action
        parent = best_child_node.parent
        while parent.parent is not None:
            best_child_node = best_child_node.parent
            parent = best_child_node.parent

        # based on the results select the best action
        return best_child_node.name