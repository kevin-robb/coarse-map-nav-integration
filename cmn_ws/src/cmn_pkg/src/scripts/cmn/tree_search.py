import numpy as np
# from habitat_sim.utils.common import quat_to_angle_axis, quat_to_coeffs, quat_from_coeffs
from typing import List

class TreeNode(object):
    def __init__(self, position, yaw:float, parent=None, act_name=None):
        """
        Init a new TreeNode object.
        @param position: array [x, y, z].
        @param yaw: orientation around world z axis.
        @param parent: Parent node, if applicable.
        @param act_name: Name of node (i.e., action name), if applicable.
        """
        # node parent
        self.parent = parent
        # node children
        self.child = []
        # node name
        self.name = act_name
        # node data
        self.position = position
        self.yaw = yaw

        # todo: for debug only
        self.dir = None


class BFTree(object):
    # Class variables:
    root_node:TreeNode = None # tree root
    tree_depth:int = None # tree depth
    branch_actions = ["move_forward", "turn_left", "turn_right"] # branching actions
    branch_leaf_nodes:List[TreeNode] = None # branching leaf nodes
    agent_forward_step:float = None # for blind model only

    def __init__(self, root_node:TreeNode, depth:int, agent_forward_step:float=0.25):
        self.root_node = root_node
        self.tree_depth = depth
        self.agent_forward_step = agent_forward_step

        # init the tree
        self.init_tree()

    # blind dynamics model assuming the local region is an empty space
    def get_leaf_node_blind(self, node:TreeNode, act:str):
        pos = node.position
        if node.dir is None:
            # Collapse yaw into a cardinal direction.
            agent_map_angle = node.yaw
            rot_control = int(np.round(agent_map_angle / (np.pi / 2)))
            if rot_control == 1:
                node.dir = "east"
            elif rot_control == -1:
                node.dir = "west"
            elif rot_control == 0:
                node.dir = "south"
            else:
                node.dir = "north"
        # create the leaf node
        leaf_node = TreeNode(None, 0, node, act)
        if act == "turn_left":
            if node.dir == "east":
                leaf_node.dir = "north"
            elif node.dir == "north":
                leaf_node.dir = "west"
            elif node.dir == "west":
                leaf_node.dir = "south"
            elif node.dir == "south":
                leaf_node.dir = "east"
            else:
                raise Exception("Invalid node direction")
            leaf_node.position = pos
        elif act == "turn_right":
            if node.dir == "east":
                leaf_node.dir = "south"
            elif node.dir == "north":
                leaf_node.dir = "east"
            elif node.dir == "west":
                leaf_node.dir = "north"
            elif node.dir == "south":
                leaf_node.dir = "west"
            else:
                raise Exception("Invalid node direction")
            leaf_node.position = pos
        elif act == "move_forward":
            x, y, z = pos[0], pos[1], pos[2]
            if node.dir == "east":
                leaf_node.dir = "east"
                x += self.agent_forward_step
            elif node.dir == "north":
                leaf_node.dir = "north"
                z -= self.agent_forward_step
            elif node.dir == "west":
                leaf_node.dir = "west"
                x -= self.agent_forward_step
            elif node.dir == "south":
                leaf_node.dir = "south"
                z += self.agent_forward_step
            else:
                raise Exception("Invalid node direction")
            leaf_node.position = [x, y, z]

        return leaf_node

    def init_tree(self):
        current_level_nodes = [self.root_node]
        next_level_nodes = []
        while self.tree_depth > 0:
            # print(f"Number of the tree nodes: {len(current_level_nodes)} in level {3 - self.tree_depth}")
            for idx, node in enumerate(current_level_nodes):
                # branch the current node by actions
                for act in self.branch_actions:
                    # compute the leaf node
                    leaf_node = self.get_leaf_node_blind(node, act)

                    # add the leaf node to next level
                    next_level_nodes.append(leaf_node)
            # decrease the depth
            self.tree_depth -= 1
            # update the level nodes
            current_level_nodes.clear()  # clear the node
            current_level_nodes = next_level_nodes.copy()  # copy to the current
            next_level_nodes.clear()

        # store the leaf nodes
        self.branch_leaf_nodes = current_level_nodes.copy()
        current_level_nodes.clear()

    @staticmethod
    def compute_the_heuristic_vec(loc_1, loc_2):
        arr_1 = np.array(loc_1)
        arr_2 = np.array(loc_2)
        heu_vec = arr_2 - arr_1
        # compute the heuristic vector
        heu_vec_norm = np.linalg.norm(heu_vec)

        # add 0 denominator check
        if heu_vec_norm == 0:
            return [0, 0]
        else:
            return heu_vec / np.linalg.norm(heu_vec)

    @staticmethod
    def cosine_similarity(vec_1, vec_2):
        vec_1 = np.array(vec_1)
        vec_2 = np.array(vec_2)
        vec_1_norm = np.linalg.norm(vec_1)
        vec_2_norm = np.linalg.norm(vec_2)

        # add 0 denominator check
        if vec_1_norm != 0 and vec_2_norm != 0:
            return np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
        else:
            return 0

    @staticmethod
    def l2_distance(loc_1, loc_2):
        loc_1 = np.array(loc_1)
        loc_2 = np.array(loc_2)
        return np.linalg.norm(loc_2 - loc_1)

    def find_the_best_child(self, vec):
        agent_loc = [self.root_node.position[0], self.root_node.position[2]]
        similarities = []
        future_dists = []
        future_vecs = []
        for node in self.branch_leaf_nodes:
            future_loc = [node.position[0], node.position[2]]
            future_vec = self.compute_the_heuristic_vec(agent_loc, future_loc)
            future_dist = self.l2_distance(agent_loc, future_loc)
            future_dists.append(future_dist)
            future_vecs.append(future_vec)
            similarity = self.cosine_similarity(vec, future_vec)

            similarities.append(similarity)
        scores = np.multiply(future_dists, similarities).tolist()
        idx = scores.index(np.max(scores))
        # print(self.branch_leaf_nodes[idx].position, future_vecs[idx])

        return self.branch_leaf_nodes[idx]


