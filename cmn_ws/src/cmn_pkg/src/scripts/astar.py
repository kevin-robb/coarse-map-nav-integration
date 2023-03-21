#!/usr/bin/env python3

"""
Set of static functions to perform A* path planning.
"""
from math import cos, sin

class Astar:
    include_diagonals = False
    # Always use the same map.
    occ_map = None
    # Neighbors for comparison and avoiding re-computations.
    nbrs = [(0, -1), (0, 1), (-1, 0), (1, 0)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if include_diagonals else [])

    @staticmethod
    def astar(start_row, start_col, goal_row, goal_col):
        """
        Use A* to generate a path from the current pose to the goal position.
        1 map grid cell = 0.1x0.1 units in ekf coords.
        map (0,0) = ekf (-10,10).
        """
        # Define start and goal nodes.
        start_cell = Cell((start_row, start_col))
        goal_cell = Cell((goal_row, goal_col))
        # make sure starting pose is on the map.
        if start_cell.r < 0 or start_cell.c < 0 or start_cell.r >= Astar.occ_map.shape[0] or start_cell.c >= Astar.occ_map.shape[1]:
            print("Starting position for A* not within map bounds.")
            return
        # check if starting node (veh pose) is in collision.
        start_cell.in_collision = Astar.occ_map[start_cell.r, start_cell.c] == 0
        # add starting node to open list.
        open_list = [start_cell]
        closed_list = []
        # iterate until reaching the goal or exhausting all cells.
        while len(open_list) > 0:
            # move first element of open list to closed list.
            open_list.sort(key=lambda cell: cell.f)
            cur_cell = open_list.pop(0)
            # stop if we've found the goal.
            if cur_cell == goal_cell:
                # recurse up thru parents to get reverse of path from start to goal.
                path_to_start = []
                while cur_cell.parent is not None:
                    path_to_start.append((cur_cell.r, cur_cell.c))
                    cur_cell = cur_cell.parent
                return path_to_start
            # add this node to the closed list.
            closed_list.append(cur_cell)
            # add its unoccupied neighbors to the open list.
            for chg in Astar.nbrs:
                nbr = Cell([cur_cell.r+chg[0], cur_cell.c+chg[1]], parent=cur_cell)
                # skip if out of bounds.
                if nbr.r < 0 or nbr.c < 0 or nbr.r >= Astar.occ_map.shape[0] or nbr.c >= Astar.occ_map.shape[1]: continue
                # skip if occluded, unless parent is occluded.
                nbr.in_collision = Astar.occ_map[nbr.r, nbr.c] == 0
                if nbr.in_collision and not nbr.parent.in_collision: continue
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
                if Astar.include_diagonals:
                    # chebyshev heuristic
                    nbr.set_cost(h=max(abs(goal_cell.r - nbr.r), abs(goal_cell.c - nbr.c)))
                else:
                    # euclidean heuristic. (keep squared to save unnecessary computation.)
                    nbr.set_cost(h=(goal_cell.r - nbr.r)**2 + (goal_cell.c - nbr.c)**2)
                # add cell to open list.
                open_list.append(nbr)



class Cell:
    def __init__(self, pos, parent=None):
        self.r = int(pos[0])
        self.c = int(pos[1])
        self.parent = parent
        self.g = 0 if parent is None else parent.g + 1
        self.f = 0
        self.in_collision = False
    
    def set_cost(self, h=None, g=None):
        # set/update either g or h and recompute the cost, f.
        if h is not None:
            self.h = h
        if g is not None:
            self.g = g
        # update the cost.
        self.f = self.g + self.h
        # give huge penalty if in collision to encourage leaving occluded cells ASAP.
        if self.in_collision: self.f += 1000

    def __eq__(self, other):
        return self.r == other.r and self.c == other.c

    def __str__(self):
        return "Cell ("+str(self.r)+","+str(self.c)+") with costs "+str([self.g, self.f])
