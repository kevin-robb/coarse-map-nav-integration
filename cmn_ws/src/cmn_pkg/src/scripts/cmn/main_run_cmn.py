import os, sys
# Allow this to be imported from a higher directory.
parent_package_dir = os.path.abspath(os.path.join(__file__, ".."))
sys.path.append(parent_package_dir)
parent_package_dir = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(parent_package_dir)

# Habitat
from Env.habitat_env import House
# CMN related
from utils.Parser import YamlParser
from Model.local_occupancy_predictor import LocalOccNet
from utils.AnyTree import TreeNode, BFTree
from utils.Map import TopoMap, compute_similarity_iou, up_scale_grid, compute_similarity_mse
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


class CoarseMapNav(object):
    """ Implement Coarse Map Navigator """
    def __init__(self, configs):
        # ======= Extract configurations =======
        self.configs = configs

        # ======= Environment configurations =======
        self.env_configs = self.configs['environment']
        self.env_name = self.configs['scene_name']
        self.env = None

        # ======= CMN Configs =======
        # Load the trained local occupancy predictor
        self.device = torch.device(self.configs['device'])
        self.model = self.load_model()

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

        # Define the graph based on the coarse map for shortest path planning
        self.coarse_map_graph = None

        # Load the coarse occupancy map: 1.0 for occupied cell and 0.0 for empty cell
        self.coarse_map_arr = self.load_coarse_2d_map()

        # Define the variables to store the beliefs
        self.predictive_belief_map = None  # predictive prob
        self.observation_prob_map = None  # measurement prob
        self.updated_belief_map = None  # updated prob

        self.agent_belief_map = None  # current agent pose belief on the map
        self.last_agent_belief_map = None  # last agent pose belief on the map

        self.current_local_map = None
        self.empty_cell_map = None
        self.goal_map_loc = None
        self.goal_map_idx = None
        self.noise_trans_prob = None

        # ======= Create visualization figures =======
        self.fig, self.grid = None, None
        # For observations
        self.ax_pano_rgb, self.art_pano_rgb = None, None
        self.ax_local_occ_gt, self.art_local_occ_gt = None, None
        self.ax_local_occ_pred, self.art_local_occ_pred = None, None
        self.ax_top_down_view, self.art_top_down_view = None, None
        # For beliefs
        self.ax_pred_update_bel, self.art_pred_update_bel = None, None
        self.ax_obs_update_bel, self.art_obs_update_bel = None, None
        self.ax_belief, self.art_belief = None, None
        # Make the plots
        self.make_plot_figures()

    # ======= Visualization functions =======
    @staticmethod
    def normalize_belief_for_visualization(belief):
        v_min = belief.min()
        v_max = belief.max()
        belief = (belief - v_min) / (v_max - v_min + 1e-8)
        return np.clip(belief, a_min=0, a_max=1)

    def show_observation(self, env_name, obs, time_step, if_init):
        # Convert the local occupancy to be binary
        local_map_size = 2 * self.configs['environment']['map_cfg']['local_map_size']
        gt_local_map_rgb = resize(obs['top_down'], (local_map_size, local_map_size, 3), anti_aliasing=True)
        gt_local_map = np.where(rgb2gray(gt_local_map_rgb) < 0.9, 0.0, 1.0)

        # Concatenate the RGB observations: clockwise
        # 4 x H x W x 3 --> H x 4W x 3
        pano_rgb = np.concatenate([obs['color_sensor_front'][:, :, 0:3],
                                   obs['color_sensor_right'][:, :, 0:3],
                                   obs['color_sensor_back'][:, :, 0:3],
                                   obs['color_sensor_left'][:, :, 0:3]], axis=1)

        # Predict the local occupancy using the local occupancy predictor
        pred_local_map = self.predict_local_occupancy(pano_rgb)

        # Set the super title of the visualization
        self.fig.suptitle(f"{env_name} : {time_step}")
        # Plot observations
        predictive_belief = self.normalize_belief_for_visualization(self.predictive_belief_map)
        observation_belief = self.normalize_belief_for_visualization(self.observation_prob_map)
        belief = self.normalize_belief_for_visualization(self.agent_belief_map)
        if if_init:
            self.art_pano_rgb = self.ax_pano_rgb.imshow(pano_rgb)
            self.art_local_occ_gt = self.ax_local_occ_gt.imshow(gt_local_map)
            self.art_local_occ_pred = self.ax_local_occ_pred.imshow(pred_local_map)
            self.art_top_down_view = self.ax_top_down_view.imshow(self.env.top_down_visualization)
            self.art_pred_update_bel = self.ax_pred_update_bel.imshow(belief)
            self.art_obs_update_bel = self.ax_obs_update_bel.imshow(belief)
            self.art_belief = self.ax_belief.imshow(belief)
        else:
            self.art_pano_rgb.set_data(pano_rgb)
            self.art_local_occ_gt.set_data(gt_local_map)
            self.art_local_occ_pred.set_data(pred_local_map)
            self.art_top_down_view.set_data(self.env.top_down_visualization)
            self.art_pred_update_bel.set_data(predictive_belief)
            self.art_obs_update_bel.set_data(observation_belief)
            self.art_belief.set_data(belief)
        # Draw observation
        self.fig.canvas.draw()
        plt.pause(1 / self.configs['visualize_freq'])

    # ======= CMN auxiliary functions =======
    def make_plot_figures(self):
        # Create the figure
        self.fig = plt.figure(figsize=(24, 8))
        self.grid = GridSpec(3, 3, figure=self.fig)

        # Add subplots for observations and local occupancy GT / Pred
        self.ax_pano_rgb = self.fig.add_subplot(self.grid[0, :])
        self.ax_local_occ_gt = self.fig.add_subplot(self.grid[1, 0])
        self.ax_local_occ_pred = self.fig.add_subplot(self.grid[1, 1])
        self.ax_top_down_view = self.fig.add_subplot(self.grid[1, 2])
        # Add subplots for beliefs
        self.ax_pred_update_bel = self.fig.add_subplot(self.grid[2, 0])
        self.ax_obs_update_bel = self.fig.add_subplot(self.grid[2, 1])
        self.ax_belief = self.fig.add_subplot(self.grid[2, 2])

        # Set titles and remove axis
        self.ax_pano_rgb.set_title("Panoramic RGB observation")
        self.ax_pano_rgb.axis("off")
        self.ax_local_occ_gt.set_title("GT local occ")
        self.ax_local_occ_gt.axis("off")
        self.ax_local_occ_pred.set_title("Pred local occ")
        self.ax_local_occ_pred.axis("off")
        self.ax_top_down_view.set_title("Top down view")
        self.ax_top_down_view.axis("off")
        self.ax_pred_update_bel.set_title("Predictive belief")
        self.ax_pred_update_bel.axis("off")
        self.ax_obs_update_bel.set_title("Obs belief")
        self.ax_obs_update_bel.axis("off")
        self.ax_belief.set_title("Belief")
        self.ax_belief.axis("off")

    def make_env(self):
        # Load the environment configurations
        configs = self.env_configs

        # Generate the path to the scene file (Replica and Gibson are two popular datasets for Habitat simulator)
        configs['scene_cfg']['scene_file'] = f"{self.configs['scene_dir']}/{self.configs['scene_name']}.glb"

        # Make the environment
        house = House(scene=configs['scene_cfg']['scene_file'],  # scene file
                      rnd_seed=configs['scene_cfg']['random_seed'],  # random seed
                      allow_sliding=configs['scene_cfg']['allow_sliding'],  # If True, agent will slide along the walls
                      max_episode_length=configs['scene_cfg']['max_episode_length'],  # Maximal episode length
                      goal_reach_eps=configs['scene_cfg']['goal_reach_eps'],  # goal reaching threshold
                      # observation configuration
                      enable_rgb=True if "color_sensor" in configs['sensor_cfg']["use_sensors"] else False,
                      enable_depth=True if "depth_sensor" in configs['sensor_cfg']["use_sensors"] else False,
                      enable_rgb_depth=True if "color_depth_sensor" in configs['sensor_cfg']["use_sensors"] else False,
                      enable_semantic=True if "semantic_sensor" in configs['sensor_cfg']["use_sensors"] else False,
                      enable_panorama=configs['sensor_cfg']['enable_panorama'],
                      sensor_height=configs['sensor_cfg']['sensor_height'],
                      depth_clip_vmax=configs['sensor_cfg']['clip_depth_max'],
                      obs_width=configs['sensor_cfg']['obs_width'],
                      obs_height=configs['sensor_cfg']['obs_height'],
                      enable_obs_noise=configs['sensor_cfg']['enable_noisy_observation'],
                      noise_intensity=configs['sensor_cfg']['noise_intensity'],
                      # map configuration
                      top_down_type=configs['map_cfg']['top_down_type'],
                      map_meters_per_pixel=configs['map_cfg']['meters_per_pixel'],
                      local_map_size=configs['map_cfg']['local_map_size'],
                      enable_local_map=configs['map_cfg']['enable_local_map'],
                      enable_ego_local_map=configs['map_cfg']['enable_ego_local_map'],
                      map_show_agent=configs['map_cfg']['show_agent'],
                      map_show_goal=configs['map_cfg']['show_goal'],
                      # agent configuration
                      move_forward_amount=configs['agent_cfg']['move_forward'],
                      turn_left_amount=configs['agent_cfg']['turn_left'],
                      turn_right_amount=configs['agent_cfg']['turn_right'],
                      enable_act_noise=configs['agent_cfg']['enable_noisy_actuation']
                      )
        self.env = house

    def load_model(self):
        # Create the local occupancy network
        model = LocalOccNet(self.configs['local_occ_net'])
        # Load the trained model
        model.load_state_dict(torch.load("Model/trained_local_occupancy_predictor_model.pt",
                                         map_location="cpu"))
        # Disable the dropout
        model.eval()
        return model.to(self.device)

    def load_coarse_2d_map(self):
        # Load the coarse 2-D map as the numpy array
        arr = np.load(f"CoarseMaps/{self.env_name}_map_binary_arr_mpp_{self.configs['cmn_cfg']['mpp']}.npy")
        row, col = arr.shape

        # Add one boundary
        arr_extended = np.ones((row + 2, col + 2))
        arr_extended[1:row+1, 1:col+1] = arr

        return arr_extended

    def render_agent_direction(self, quaternion):
        """ Render the ground truth agent direction using odometry sensor data"""
        agent_map_angle = self.env.quaternion_to_angle(quaternion)
        rot_control = int(np.round(agent_map_angle / (np.pi / 2)))
        if rot_control == 1:
            agent_dir = "east"
        elif rot_control == -1:
            agent_dir = "west"
        elif rot_control == 0:
            agent_dir = "south"
        else:
            agent_dir = "north"
        return agent_dir

    def render_local_map_info(self, loc):
        """ Render the local map from the 2-D coarse map """
        # Convert the real world coordinates to map coordinates
        row, col = self.env.map_2d_to_grid_func(loc, self.coarse_map_arr.shape)
        return row, col

    def process_observation(self, obs):
        # Convert observation from ndarray to PIL.Image
        obs = Image.fromarray(obs)
        # Convert observation from PIL.Image to tensor
        obs = self.transformation(obs)
        return obs

    # ======= CMN core functions =======
    def init_coarse_map_graph(self):
        """ Graph-structured coarse map for shortest path planning """
        # Build a graph based on the coarse map for shortest path planning
        self.coarse_map_graph = TopoMap(self.coarse_map_arr, self.configs['cmn_cfg']['local_map_size'])

    def init_coarse_map_grid_belief(self):
        """ Grid-based beliefs for Bayesian filtering """
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

    @staticmethod
    def compute_norm_heuristic_vec(loc_1, loc_2):
        arr_1 = np.array(loc_1)
        arr_2 = np.array(loc_2)
        heu_vec = arr_2 - arr_1
        return heu_vec / np.linalg.norm(heu_vec)

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
        # Process the observation to tensor
        pano_rgb_obs_tensor = self.process_observation(pano_rgb_obs)

        with torch.no_grad():
            # Predict the local occupancy
            pred_local_occ = self.model(pano_rgb_obs_tensor)
            # Reshape the predicted local occupancy
            pred_local_occ = pred_local_occ.cpu().squeeze(dim=0).squeeze(dim=0).numpy()

        return pred_local_occ

    def cmn_close(self):
        # Close the house environment
        self.env.close()
        # Clear the visualization
        # Clear the subplots
        self.ax_pano_rgb.clear()
        self.ax_local_occ_gt.clear()
        self.ax_local_occ_pred.clear()
        # Clear the artists
        self.art_pano_rgb = None
        self.art_local_occ_gt, self.art_local_occ_pred = None, None

    def cmn_reset(self):
        # Sample valid start and goal locations in the real world
        # Note that, the corresponding goal location should be in the empty cell
        while True:
            # Reset the environment
            observation = self.env.reset()
            # Sample a valid goal location in the real world that falls in the empty cell on the map
            self.goal_map_loc = self.render_local_map_info(self.env.goal_loc[0:3])
            if self.coarse_map_arr[self.goal_map_loc[0], self.goal_map_loc[1]] == 0.0:
                break

        # Initialize the coarse map graph for path planning
        self.init_coarse_map_graph()
        self.goal_map_idx = self.coarse_map_graph.sampled_locations.index(tuple(self.goal_map_loc))

        # Initialize the coarse map grid beliefs for global localization
        self.init_coarse_map_grid_belief()

        return observation

    def cmn_update_beliefs(self, obs, act):
        # Compute the orientation of the agent, return is east, north, west, and south
        # |state| = 7, the last four elements are the quaternion
        agent_dir = self.render_agent_direction(obs['state'][3:])

        # Predict the local occupancy from panoramic RGB images
        pano_rgb = np.concatenate([obs['color_sensor_front'][:, :, 0:3],
                                   obs['color_sensor_right'][:, :, 0:3],
                                   obs['color_sensor_back'][:, :, 0:3],
                                   obs['color_sensor_left'][:, :, 0:3]], axis=1)
        map_obs = self.predict_local_occupancy(pano_rgb)

        # Rotate the egocentric local occupancy to face NORTH
        if agent_dir == "east":
            map_obs = rotate(map_obs, -90)
        elif agent_dir == "north":
            pass
        elif agent_dir == "west":
            map_obs = rotate(map_obs, 90)
        elif agent_dir == "south":
            map_obs = rotate(map_obs, 180)
        else:
            raise Exception("Invalid agent direction")
        self.current_local_map = map_obs

        # add noise
        if act == "move_forward":
            # randomly sample a p from a uniform distribution between [0, 1]
            self.noise_trans_prob = np.random.rand()

        # Predictive update stage
        self.predictive_belief_map = self.predictive_update_func(act, agent_dir)

        # Measurement update stage
        self.observation_prob_map = self.measurement_update_func(self.current_local_map)

        # Full Bayesian update with weights
        log_belief = np.log(self.observation_prob_map + 1e-8) + np.log(self.predictive_belief_map + 1e-8)
        belief = np.exp(log_belief)
        normalized_belief = belief / belief.sum()

        # Record the belief for visualization
        self.last_agent_belief_map = self.agent_belief_map.copy()
        self.updated_belief_map = normalized_belief.copy()
        self.agent_belief_map = normalized_belief.copy()

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

    def run(self):
        # Make the environment
        self.make_env()

        # Record the number of success
        success_count = 0

        # Loop for N episodes
        for ep_idx in range(self.configs['episode_num']):
            # Reset CMN for each new episode
            observation = self.cmn_reset()

            # Visualization only
            self.show_observation(self.env_name, observation, time_step=0, if_init=True)

            # Start one navigation episode
            for t in range(self.env_configs['scene_cfg']['max_episode_length']):
                # Select one action using CMN
                action = self.cmn_planner(observation['state'][3:])
                # action = np.random.choice(['move_forward', 'turn_left', 'turn_right'], 1)[0]

                # Interact with the Habitat to obtain the next observation
                next_observation, done, timeout = self.env.step(action)

                # Update the beliefs
                self.cmn_update_beliefs(observation, action)

                # Update the observation
                observation = next_observation

                # Show observation
                self.show_observation(self.env_name, observation, time_step=t, if_init=False)

                # Reset if the goal is reached or timeouts
                if done or timeout:
                    if done:
                        # if the goal is True, the goal is reached
                        success_count += 1
                        print(f"Episode {ep_idx+1} = SUCCESS :) at time step {t}")
                        break
                    else:
                        # otherwise, it fails
                        print(f"Episode = {ep_idx + 1} = FAIL :(")

        # Print success rate
        print(f"The mean success rate in {self.env_name} = {success_count / self.configs['episode_num']}")
        # Close the CMN
        self.cmn_close()


if __name__ == "__main__":
    # Load configurations
    configurations = YamlParser("Config/run_cmn.yaml").data

    # Create Coarse Map Navigator (CMN)
    cmn = CoarseMapNav(configurations)

    # Run CMN
    cmn.run()
