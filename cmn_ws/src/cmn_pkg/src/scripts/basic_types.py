 #!/usr/bin/env python3

"""
Basic datatypes that will be used throughout the project.
"""

import numpy as np
from math import remainder, pi, tau, atan2, asin
from typing import Tuple, List

class Pose:
    yaw = None # Orientation in radians. 0 = right/east. In range [-pi,pi]

    def get_direction(self) -> str:
        """
        Discretize the yaw into the nearest cardinal direction.
        @return string representation of the agent's direction, either 'east', 'north', 'west', or 'south'.
        """
        if self.yaw is None:
            return "none"
        dist_to_east = abs(remainder(self.yaw, tau))
        dist_to_north = abs(remainder(self.yaw - pi/2, tau))
        dist_to_west = abs(remainder(self.yaw - pi, tau))
        dist_to_south = abs(remainder(self.yaw + pi/2, tau))
        if dist_to_east < min([dist_to_north, dist_to_west, dist_to_south]):
            return "east"
        elif dist_to_north < min([dist_to_west, dist_to_south]):
            return "north"
        elif dist_to_west < dist_to_south:
            return "west"
        else:
            return "south"
        
        # Method from Chengguang's code:
        # rot_control = int(np.round(self.yaw / (np.pi / 2)))
        # if rot_control == 1:
        #     agent_dir = "east"
        # elif rot_control == -1:
        #     agent_dir = "west"
        # elif rot_control == 0:
        #     agent_dir = "south"
        # else:
        #     agent_dir = "north"
        # return agent_dir
        

class PoseMeters(Pose):
    """
    2D vehicle pose, represented in meters.
    """
    x, y = None, None # Position in meters. Origin is center of map.
    yaw = None # Heading in radians. 0 is to the right. In range [-pi, pi].

    def __init__(self, x:float=None, y:float=None, yaw:float=None):
        """
        Constructor from x, y, yaw.
        """
        self.x = x
        self.y = y
        if yaw is not None:
            self.yaw = remainder(yaw, tau)

    def init_from_se2(self, se2):
        """
        Set from 3x3 numpy array SE(2) representation.
        """
        self.x = se2[0, 2]
        self.y = se2[1, 2]
        self.yaw = np.arctan2(se2[1,0], se2[0,0])

    def as_np_array(self):
        """
        Return a 3x1 numpy array representation of this pose.
        """
        return np.array([self.x, self.y, self.yaw])
    
    def __str__(self):
        if self.yaw is None:
            return "({:.2f}, {:.2f})".format(self.x, self.y)
        else:
            return "({:.2f}, {:.2f}, {:.2f})".format(self.x, self.y, self.yaw)
        
    def as_se2(self):
        """
        Return the SE(2) representation (3x3 numpy array) of this pose.
        """
        return np.array([[np.cos(self.yaw), -np.sin(self.yaw), self.x],
                         [np.sin(self.yaw), np.cos(self.yaw), self.y],
                         [0, 0, 1]])


class PosePixels(Pose):
    """
    2D pose, represented in pixels on the map.
    """
    r, c = None, None # Position of cell in pixels on map. Origin is top-left of image.

    def __init__(self, r:int, c:int, yaw:float=None):
        self.r = r
        self.c = c
        self.yaw = yaw

    def __str__(self):
        if self.yaw is None:
            return "({:}, {:})".format(self.r, self.c)
        else:
            return "({:}, {:}, {:.2f})".format(self.r, self.c, self.yaw)
        
    def distance(self, p2) -> float:
        """
        Get the distance in pixels between this and another pose.
        @param p2 - a second PosePixels object.
        """
        if self.r is None or self.c is None or p2.r is None or p2.c is None:
            return None
        return np.sqrt((self.r - p2.r)**2 + (self.c - p2.c)**2)

    def as_tuple(self):
        """
        @return tuple (r,c) representation of this object.
        """
        return (self.r, self.c)
