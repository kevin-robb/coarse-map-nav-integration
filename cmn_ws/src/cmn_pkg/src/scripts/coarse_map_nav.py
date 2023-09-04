"""
Original code for the Coarse Map Navigation project was created by Chengguang Xu to interface with Habitat simulator.
This modified version was created by Kevin Robb to work with physical robot in ROS Noetic.
"""

# Pytorch related
import torch
from torchvision.transforms import Compose, Normalize, PILToTensor
# Image process related
from PIL import Image
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
# Others
import numpy as np
import rospy, rospkg, yaml
from math import radians, degrees

from cmn_demo.main_run_cmn import CoarseMapNav
from map_handler import MapFrameManager
from basic_types import PoseMeters, PosePixels


class CoarseMapNavInterface(CoarseMapNav):
    """
    Modified version of CMN to remove habitat reliance and interface with the rest of my implementation.
    """
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


    def __init__(self):
        # Load params from config.yaml.
        self.read_params()
        # Load the trained local occupancy predictor
        self.model = self.load_model()

        # Define the transformation to transform the observation suitable to the local occupancy model.
        # Convert PIL.Image to tensor. Convert uint8 to float. Convert [0, 255] to [0, 1].
        # Normalize the observation. Reshape the tensor by adding the batch dimension.
        # Put the tensor to GPU.
        self.transformation = Compose([
            PILToTensor(),
            lambda x: x.float(),
            lambda x: x / 255.0,
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            lambda x: x.unsqueeze(dim=0),
            lambda x: x.to(self.device)
        ])

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


    def read_params(self):
        # Determine filepath.
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('cmn_pkg')
        # Open the yaml and get the relevant params.
        with open(pkg_path+'/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            # TODO have runner node tell us if discrete or continous run mode; no longer a yaml param.
            # self.use_discrete_state_space = True if config["run_mode"] == "discrete" else False
            # ML model params
            self.device = torch.device(config["model"]["device"])
    
    def set_map_frame_manager(self, mfm:MapFrameManager):
        """
        Set our reference to the map frame manager, which allows us to use the map and coordinate transform functions.
        @param mfg MapFrameManager instance that has already been initialized with a map.
        """
        self.mfm = mfm
        # NOTE the MapFrameManager uses the convention 1.0 == Free, 0.0 == Occluded.
        # We want the opposite here, with 1.0 == Occupied, and 0.0 == Free, so we have to invert it.
        self.coarse_map_arr = np.ones(mfm.map.shape) - mfm.map

        # Initialize the coarse map graph for path planning
        self.init_coarse_map_graph()
        # Find the index of the goal point in the graph.
        self.goal_map_idx = self.coarse_map_graph.sampled_locations.index(tuple(self.goal_map_loc))
        # Initialize the coarse map grid beliefs for global localization
        self.init_coarse_map_grid_belief()


    def update_beliefs(self, agent_pose:PoseMeters, pano_rgb, action:str):
        """
        Run the update step.
        @param agent_pose TODO where does this come from? It's just part of the "observation" in CMN code...
        @param pano_rgb Panoramic image created by concatenating four RGB measurements horizontally.
        @param action The action to take.
        """
        # Predict the local occupancy from panoramic RGB images
        map_obs = self.predict_local_occupancy(pano_rgb)

        # # Given a particular robot pose (estimate), get the expected observation.
        # obs_img_expected, _ = self.mfm.extract_observation_region(agent_pose)
        # # This always has the robot at center-left, facing right. Rotate this to face NORTH.
        # obs_img_expected = rotate(map_obs, -90)

        # Use the orientation of the agent to rotate the egocentric local occupancy to face NORTH.
        self.current_local_map = rotate(map_obs, degrees(agent_pose.yaw))

        # add noise
        if action == "move_forward":
            # randomly sample a p from a uniform distribution between [0, 1]
            self.noise_trans_prob = np.random.rand()

        # Predictive update stage
        self.predictive_belief_map = self.predictive_update_func(action, agent_pose.get_direction())

        # Measurement update stage
        self.observation_prob_map = self.measurement_update_func(self.current_local_map)

        # Full Bayesian update with weights
        log_belief = np.log(self.observation_prob_map + 1e-8) + np.log(self.predictive_belief_map + 1e-8)
        belief = np.exp(log_belief)
        normalized_belief = belief / belief.sum()

        # Record the belief for visualization
        # self.last_agent_belief_map = self.agent_belief_map.copy()
        # self.updated_belief_map = normalized_belief.copy()
        # self.agent_belief_map = normalized_belief.copy()


    def run_one_iteration(self, pano_rgb):
        """
        Run one iteration of CMN. This involves choosing an action, commanding it, and propagating our belief models forward.
        """
        # Select one action using CMN
        # TODO need to get robot's orientation as a quaternion.
        # action = self.cmn_planner(q)
        # action = np.random.choice(['move_forward', 'turn_left', 'turn_right'], 1)[0]

        # Interact with the Habitat to obtain the next observation
        # next_observation, done, timeout = self.env.step(action)
        # TODO command the motion using discrete motion planner.
        # TODO check if the goal is reached.

        # Update the beliefs
        # self.update_beliefs(agent_pose, pano_rgb, action)

        # Update the observation
        # TODO ???
        # observation = next_observation
