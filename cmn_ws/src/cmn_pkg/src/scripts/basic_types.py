 #!/usr/bin/env python3

"""
Basic datatypes that will be used throughout the project.
"""

import numpy as np
from math import remainder, pi, tau

class Pose:
    yaw = None

    def get_direction(self) -> str:
        """
        Discretize the yaw into the nearest cardinal direction.
        @return string representation of the agent's direction, either 'east', 'north', 'west', or 'south'.
        """
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
        

class PoseMeters(Pose):
    """
    2D vehicle pose, represented in meters.
    """
    x = None
    y = None

    def __init__(self, x:float, y:float, yaw:float=None):
        self.x = x
        self.y = y
        self.yaw = yaw

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


class PosePixels(Pose):
    """
    2D pose, represented in pixels on the map.
    """
    r = None
    c = None

    def __init__(self, r:int, c:int, yaw:float=None):
        self.r = r
        self.c = c
        self.yaw = yaw

    def __str__(self):
        if self.yaw is None:
            return "({:}, {:})".format(self.r, self.c)
        else:
            return "({:}, {:}, {:.2f})".format(self.r, self.c, self.yaw)
