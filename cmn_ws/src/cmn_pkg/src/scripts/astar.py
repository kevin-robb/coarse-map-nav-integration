#!/usr/bin/env python3

import rospy

from scripts.basic_types import PosePixels

class Astar:
    include_diagonals = False
    map = None # 2D numpy array of the global map
    # Neighbors for comparison and avoiding re-computations.
    nbrs = [(0, -1), (0, 1), (-1, 0), (1, 0)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if include_diagonals else [])

    def run_astar(self, start_pose_px:PosePixels, goal_pose_px:PosePixels):
        """
        Use A* to generate a path from the current pose to the goal position.
        @param start_pose_px, goal_pose_px PosePixels of start and end of path.
        @return List of PosePixels describing the path in reverse (i.e., from goal to start).
        """
        # Define start and goal nodes as Cells.
        start_cell = Cell(start_pose_px)
        goal_cell = Cell(goal_pose_px)
        # make sure starting pose is on the map and not in collision.
        if start_cell.r < 0 or start_cell.c < 0 or start_cell.r >= self.map.shape[0] or start_cell.c >= self.map.shape[1]:
            rospy.logerr("A*: Starting position not within map bounds. Exiting without computing a path.")
            return
        if start_cell.in_collision(self.map):
            rospy.logwarn("A*: Starting position is in collision. Computing a path, and encouraging motion to free space.")
        # make sure goal is on the map and not in collision.
        if goal_cell.r < 0 or goal_cell.c < 0 or goal_cell.r >= self.map.shape[0] or goal_cell.c >= self.map.shape[1]:
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
            # rospy.loginfo("A*: Iteration with len(open_list)={:}, len(closed_list)={:}".format(len(open_list), len(closed_list)))
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
                nbr = Cell([cur_cell.r+chg[0], cur_cell.c+chg[1]], parent=cur_cell)
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



class Cell:
    """
    Simple representation of nodes to help with A*.
    """

    def __init__(self, pose_px:PosePixels, parent=None):
        self.r = int(pose_px.r)
        self.c = int(pose_px.c)
        self.parent = parent
        self.g = 0 if parent is None else parent.g + 1
        self.f = 0

    def in_collision(self, map):
        """
        Check if this cell is occupied on the given map.
        @param map, a 2D numpy array of the occupancy grid map.
        @return true if it's occupied.
        """
        return map[self.r, self.c] == 0
    
    def parent_in_collision(self, map):
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
