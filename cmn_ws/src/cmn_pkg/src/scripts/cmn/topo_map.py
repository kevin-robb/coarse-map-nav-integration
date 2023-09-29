import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skimage.color import rgb2gray


class TopoMap(object):
    def __init__(self, map_arr, local_occ_height:int, local_occ_width:int):
        """
        @param map_arr
        @param local_occ_height, local_occ_width - Size of current local map "observation" in pixels.
        """
        # Binary map shape
        self.map_row, self.map_col = map_arr.shape

        # Set the local map size
        # TODO verify order of height, width is correct if using non-square observation region.
        self.local_occ_size = [local_occ_height, local_occ_width]

        # Covert the map to binary: 0 for empty cell and 1 for occupied cell
        self.map_binary_arr = map_arr

        # Get all empty cells: idx, location, and local occupancy 3 x 3
        self.local_maps = self.get_valid_locations()

        # Make the graph
        self.global_map_graph, self.global_map_dict, self.sampled_locations = self.make_graph()


    def get_valid_locations(self):
        # obtain all empty cells
        loc_coords = np.where(self.map_binary_arr == 0.0)
        space_locs = [(r, c) for r, c in zip(loc_coords[0], loc_coords[1])]

        # Crop 3 x 3 occupancy grids
        cropped_local_maps = []
        # Loop over all candidates for the center pixel.
        min_dist_to_edge = (self.local_occ_size[0] - 1) // 2
        for idx, space in enumerate(space_locs):
            # Only accept local maps that are completely within bounds.
            if (min_dist_to_edge < space[0] < (self.map_row - min_dist_to_edge - 1)) and (min_dist_to_edge < space[1] < (self.map_col - min_dist_to_edge - 1)):
                # crop the local map
                from_r = space[0] - min_dist_to_edge
                to_r = space[0] + min_dist_to_edge + 1
                from_c = space[1] - min_dist_to_edge
                to_c = space[1] + min_dist_to_edge + 1
                local_map = self.map_binary_arr[from_r:to_r, from_c:to_c]

                # save the local maps
                cropped_local_maps.append({'id': idx,  # local map index
                                           'loc': space,  # local map location
                                           'map_arr': local_map  # local map array
                                           })
        return cropped_local_maps

    # function is used to make the global graph
    def make_graph(self):
        # boundary
        min_r, max_r = 0, self.map_binary_arr.shape[0] - 1
        min_c, max_c = 0, self.map_binary_arr.shape[1] - 1

        # find the neighbors
        def find_neighbors(center_loc, act_stride=1):
            # actions
            actions = [[-act_stride, 0], [act_stride, 0], [0, -act_stride], [0, act_stride]]
            # compute the neighbors
            neighbor_idx = []
            for act in actions:
                # compute the neighbor location and clip the invalid value
                neighbor_loc_r = center_loc[0] + act[0]
                neighbor_loc_c = center_loc[1] + act[1]

                # check the validation
                if neighbor_loc_c > max_c or neighbor_loc_c < min_c or neighbor_loc_r > max_r or neighbor_loc_r < min_r:
                    continue
                neighbor_loc = (neighbor_loc_r, neighbor_loc_c)

                if neighbor_loc in sampled_cells:
                    neighbor_idx.append(sampled_cells.index(neighbor_loc))

            return neighbor_idx

        # find empty cells
        sampled_cells = [item['loc'] for item in self.local_maps]

        # build the graph dict
        graph_dict = {}
        for idx, loc in enumerate(sampled_cells):
            # create the dict for current vertex
            vertex_dict = {'loc': loc, 'edges': []}
            # find the indices of all neighbors
            neighbor_indices = find_neighbors(loc)
            # add all edges to the vertex
            for n_idx in neighbor_indices:
                vertex_dict['edges'].append((idx, n_idx))
            # save the vertex and its edges
            graph_dict[idx] = vertex_dict

        # build the graph from the dict
        graph = self.make_graph_from_dict(graph_dict)

        return graph, graph_dict, sampled_cells

    # function is used to plan the shortest path between two vertex using dijkstra's algorithm
    def dijkstra_path(self, s_node, e_node):
        # find the shortest path
        try:
            shortest_path = nx.dijkstra_path(self.global_map_graph, s_node, e_node)
        except nx.NetworkXNoPath:
            shortest_path = []

        return shortest_path

    def display_graph_on_map(self):
        # extract all nodes
        vertices = [self.global_map_dict[key]['loc'] for key in self.global_map_dict.keys()]
        edges = [self.global_map_dict[key]['edges'] for key in self.global_map_dict.keys()]

        # plot the vertices on the map with size 10  x 10 in pixels
        display_map = self.map_binary_arr.copy()

        # plot the edges on the map
        drawn_pair = []
        for idx, item in enumerate(edges):
            print(f"{idx}-{len(edges)}: {item}")
            for sub_item in item:
                if sub_item in drawn_pair:
                    continue
                else:
                    # start and goal locations
                    start_loc = self.sampled_locations[sub_item[0]]
                    end_loc = self.sampled_locations[sub_item[1]]

                    # align the coordinates
                    line_from = [start_loc[1], end_loc[1]]
                    line_to = [start_loc[0], end_loc[0]]
                    plt.plot(start_loc[1], start_loc[0], "ro")
                    plt.plot(end_loc[1], end_loc[0], 'ro')
                    plt.plot(line_from, line_to, linewidth=1, color='red')

                    # save the drawn pairs
                    drawn_pair.append(sub_item)
                    drawn_pair.append((sub_item[1], sub_item[0]))

        # plot the results
        plt.title(f"Local map size = {self.local_occ_size[0] * 2}")
        plt.imshow(1 - display_map, cmap="gray")
        plt.show()

    @staticmethod
    def make_graph_from_dict(g_dict):
        G = nx.Graph()
        for key, val in g_dict.items():
            # add the node
            G.add_node(key, loc=val['loc'])
            # add the edges
            for e in val['edges']:
                G.add_edge(*e, start=val['loc'], end=e)
        return G

    @staticmethod
    def compute_dist(s_loc, e_loc):
        s_loc = np.array(s_loc)
        e_loc = np.array(e_loc)
        dist = np.linalg.norm(s_loc - e_loc)
        return dist


def compute_similarity_iou(occ_map_1, occ_map_2):
    """ Use the Intersection over Union metric """
    # Round the prediction: 0 for empty space and 1 for occupied space
    occ_map_1 = occ_map_1.round()
    occ_map_2 = occ_map_2.round()

    # Compute the intersected cells number plus a small offset to deal with empty observation
    intersection = np.logical_and(occ_map_1, occ_map_2).sum() + 1e-8
    # Compute the union cells number plus a small offset to avoid divide by 0
    union = np.logical_or(occ_map_1, occ_map_2).sum() + 1e-8
    # Compute the IoU
    iou = intersection / union

    return iou


def compute_similarity_mse(occ_map_1, occ_map_2):
    """ Use the mean square error metric """
    # Round the prediction: 0 for empty space and 1 for occupied space
    return 1 - ((occ_map_1.round() - occ_map_2.round()) ** 2).mean()


def up_scale_grid(grid):
    if grid.shape[0] == 3 and grid.shape[1] == 3:
        """ Up scaling 3 x 3 grid to 128  x 128 grid"""
        r1_left_block = grid[0, 0] * np.ones((42, 42))
        r1_middle_block = grid[0, 1] * np.ones((42, 43))
        r1_right_block = grid[0, 2] * np.ones((42, 43))

        r2_left_block = grid[1, 0] * np.ones((43, 42))
        r2_middle_block = grid[1, 1] * np.ones((43, 43))
        r2_right_block = grid[1, 2] * np.ones((43, 43))

        r3_left_block = grid[2, 0] * np.ones((43, 42))
        r3_middle_block = grid[2, 1] * np.ones((43, 43))
        r3_right_block = grid[2, 2] * np.ones((43, 43))

        return np.block([
            [r1_left_block, r1_middle_block, r1_right_block],
            [r2_left_block, r2_middle_block, r2_right_block],
            [r3_left_block, r3_middle_block, r3_right_block]
        ])
    else:
        # Generic resize.
        return np.resize(grid, (128, 128))
