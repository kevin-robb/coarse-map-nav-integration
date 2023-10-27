#!/usr/bin/env python3

import rospy
import numpy as np
from math import remainder, tau
from scripts.basic_types import PosePixels, yaw_to_cardinal_dir, cardinal_dir_to_yaw

class Astar:
    verbose = False
    include_diagonals = False
    map = None # 2D numpy array of the global map
    goal_cell:PosePixels = None # Place to save goal cell so we don't have to pass it in every time.
    # Neighbors for comparison and avoiding re-computations.
    nbrs = [(0, -1), (0, 1), (-1, 0), (1, 0)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if include_diagonals else [])

    # Last computed path. Saving it here makes it easier to get it for viz.
    last_path_px_reversed = None

    def run_astar(self, start_pose_px:PosePixels, goal_pose_px:PosePixels=None):
        """
        Use A* to generate a path from the current pose to the goal position. Must have set self.map already.
        @param start_pose_px, goal_pose_px PosePixels of start and end of path.
        @note goal_pose_px can be omitted to use self.goal_cell instead.
        @return List of PosePixels describing the path in reverse (i.e., from goal to start).
        """
        if self.map is None:
            rospy.logerr("A*: Cannot run_astar since self.map is None!")
            return
        # Define start and goal nodes as Cells.
        start_cell = Cell(start_pose_px)
        if goal_pose_px is None:
            goal_pose_px = self.goal_cell
        goal_cell = Cell(goal_pose_px)
        # make sure starting pose is on the map and not in collision.
        if start_cell.out_of_bounds(self.map):
            rospy.logerr("A*: Starting position not within map bounds. Exiting without computing a path.")
            return
        if start_cell.in_collision(self.map):
            rospy.logwarn("A*: Starting position is in collision. Computing a path, and encouraging motion to free space.")
        # make sure goal is on the map and not in collision.
        if goal_cell.out_of_bounds(self.map):
            rospy.logerr("A*: Goal position not within map bounds. Exiting without computing a path.")
            return
        if goal_cell.in_collision(self.map):
            rospy.logerr("A*: Goal position is in collision. Exiting without computing a path.")
            return

        # add starting node to open list.
        open_list = [start_cell]
        closed_list = []
        # iterate until reaching the goal or exhausting all cells.
        while len(open_list) > 0:
            if self.verbose:
                rospy.loginfo("A*: Iteration with len(open_list)={:}, len(closed_list)={:}".format(len(open_list), len(closed_list)))
            # move first element of open list to closed list.
            open_list.sort(key=lambda cell: cell.f)
            cur_cell = open_list.pop(0)
            # stop if we've found the goal.
            if cur_cell == goal_cell:
                # recurse up thru parents to get reverse of path from start to goal.
                path_to_start = []
                while cur_cell.parent is not None:
                    path_to_start.append(PosePixels(cur_cell.r, cur_cell.c))
                    cur_cell = cur_cell.parent
                return path_to_start
            # add this node to the closed list.
            closed_list.append(cur_cell)
            # add its unoccupied neighbors to the open list.
            for chg in self.nbrs:
                nbr = Cell(PosePixels(cur_cell.r+chg[0], cur_cell.c+chg[1]), parent=cur_cell)
                # skip if out of bounds.
                if nbr.r < 0 or nbr.c < 0 or nbr.r >= self.map.shape[0] or nbr.c >= self.map.shape[1]: continue
                # skip if occluded, unless parent is occluded.
                if nbr.in_collision(self.map) and not nbr.parent_in_collision(self.map): continue
                # skip if already in closed list.
                if any([nbr == c for c in closed_list]): continue
                # skip if already in open list, unless the cost is lower.
                seen = [nbr == open_cell for open_cell in open_list]
                try:
                    match_i = seen.index(True)
                    # this cell has been added to the open list already.
                    # check if the new or existing route here is better.
                    if nbr.g < open_list[match_i].g: 
                        # the cell has a shorter path this new way, so update its cost and parent.
                        open_list[match_i].set_cost(g=nbr.g)
                        open_list[match_i].parent = nbr.parent
                    continue
                except:
                    # there's no match, so proceed.
                    pass
                # compute heuristic "cost-to-go"
                if self.include_diagonals:
                    # chebyshev heuristic
                    nbr.set_cost(h=max(abs(goal_cell.r - nbr.r), abs(goal_cell.c - nbr.c)))
                else:
                    # euclidean heuristic. (keep squared to save unnecessary computation of square roots.)
                    nbr.set_cost(h=(goal_cell.r - nbr.r)**2 + (goal_cell.c - nbr.c)**2)
                # add cell to open list.
                open_list.append(nbr)


    def get_next_discrete_action(self, start_pose_px:PosePixels) -> str:
        """
        Use A* with the current self.map and self.goal_cell to plan a path, and determine the next action to take.
        @param start_pose_px Current estimate of the robot pose, in pixels on self.map.
        @return str - Next action to take, one of "move_forward", "turn_left", "turn_right".
        """
        if self.map is None or self.goal_cell is None:
            rospy.logerr("A*: Cannot get_next_discrete_action unless self.map and self.goal_cell have been set!")
            exit()
        # Force cardinal direction movement only.
        self.include_diagonals = False

        # Generate (reverse) path with A*.
        self.last_path_px_reversed = self.run_astar(start_pose_px)
        # Check if we were unable to plan a path.
        if self.last_path_px_reversed is None or len(self.last_path_px_reversed) < 1:
            rospy.logwarn("A*: Unable to plan a path, so commanding a random discrete action.")
            return np.random.choice(['move_forward', 'turn_left', 'turn_right'], 1)[0]

        # Check which direction from the current cell we should go to next.
        next_cell = self.last_path_px_reversed[-1]
        dir_to_next_cell = start_pose_px.direction_to_cell(next_cell)
        dir_current_yaw = start_pose_px.get_direction()
        if self.verbose:
            print("dir_to_next_cell is {:}, and dir_current_yaw is {:}".format(dir_to_next_cell, dir_current_yaw))
        if dir_to_next_cell == dir_current_yaw:
            return "move_forward"
        else:
            yaw_diff_rads = remainder(cardinal_dir_to_yaw[dir_to_next_cell] - cardinal_dir_to_yaw[dir_current_yaw], tau)
            return "turn_left" if yaw_diff_rads > 0 else "turn_right"


class Cell:
    """
    Simple representation of nodes to help with A*.
    """

    def __init__(self, pose_px:PosePixels, parent=None):
        if pose_px is None:
            rospy.logerr("A*: Illegal creation of Cell; pose_px cannot be None.")
            exit()
        self.r = int(pose_px.r)
        self.c = int(pose_px.c)
        self.parent = parent
        self.g = 0 if parent is None else parent.g + 1
        self.f = 0

    def out_of_bounds(self, map) -> bool:
        """
        Check if this cell is in bounds or not on the given map.
        @param map, a 2D numpy array of the occupancy grid map.
        @return true if it's out of bounds.
        """
        return self.r < 0 or self.c < 0 or self.r >= map.shape[0] or self.c >= map.shape[1]

    def in_collision(self, map) -> bool:
        """
        Check if this cell is occupied on the given map.
        @param map, a 2D numpy array of the occupancy grid map.
        @return true if it's occupied.
        """
        return map[self.r, self.c] == 0
    
    def parent_in_collision(self, map) -> bool:
        """
        If this cell has a parent, check if that parent is in collision.
        """
        if self.parent is None:
            return False
        return self.parent.in_collision(map)
    
    def set_cost(self, h=None, g=None, map=None):
        # set/update either g or h and recompute the cost, f.
        if h is not None:
            self.h = h
        if g is not None:
            self.g = g
        # update the cost.
        self.f = self.g + self.h
        # give huge penalty if in collision to encourage leaving occluded cells ASAP.
        if map is not None:
            if self.in_collision(map):
                self.f += 1000

    def __eq__(self, other):
        return self.r == other.r and self.c == other.c

    def __str__(self):
        return "Cell ("+str(self.r)+","+str(self.c)+") with costs "+str([self.g, self.f])
