 #!/usr/bin/env python3

"""
Basic datatypes that will be used throughout the project.
"""

import numpy as np

class PoseMeters:
    """
    2D vehicle pose, represented in meters.
    """
    x = None
    y = None
    yaw = None

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
        return "({:.2f}, {:.2f}, {:.2f})".format(self.x, self.y, self.yaw)


class PosePixels:
    """
    2D pose, represented in pixels on the map.
    """
    r = None
    c = None
    yaw = None

    def __init__(self, r:int, c:int, yaw:float=None):
        self.r = r
        self.c = c
        self.yaw = yaw

    def __str__(self):
        return "({:}, {:}, {:.2f})".format(self.r, self.c, self.yaw)
