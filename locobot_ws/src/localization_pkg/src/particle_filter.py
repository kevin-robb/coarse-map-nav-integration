#!/usr/bin/env python3

"""
Particle Filter class implementation.
Can separately process its prediction and update steps at different, independent rates, and can be polled for the most likely particle estimate at any time.
"""

import numpy as np
from math import pi, sin, cos

class ParticleFilter:
    # GLOBAL VARIABLES
    num_particles = None
    state_size = None
    occ_map = None
    particle_set = None
    particle_weights = None
    scale = None
    # FILTER OUTPUT
    best_estimate = None


    def __init__(self):
        """
        Instantiate the particle filter object.
        """
        self.best_estimate = np.array([0.0, 0.0, 0.0])
    

    def set_params(self, config):
        """
        Set particle filter parameters from the yaml.
        @param config, a dictionary of the "particle filter" section from config.yaml.
        """
        self.num_particles = int(config["num_particles"])
        self.state_size = int(config["state_size"])
    

    def set_map(self, map):
        """
        Get the occupancy grid map, and save it to use each iteration.
        @param map, a 2D array containing the processed occupancy grid map.
        """
        self.occ_map = map


    def propagate_particles(self, dx, dy, dtheta):
        """
        Given a relative motion since last iteration, apply this to all particles.
        @param dx, commanded change in x.
        @param dy, commanded change in y.
        @param dtheta, commanded change in yaw.
        """
        # Compute coordinate-free representation of this motion.
        # TODO test on robot after finding exact topic. these might already be like  "x = forward motion from current position"
        pass


    def update_with_observation(self, observation):
        """
        Use an observation to evaluate the likelihood of all particles.
        @param observation
        @return 3x1 numpy vector of best particle estimate (x,y,yaw).
        """
        # TODO loop through all particles.
        # self.particle_weights[i] = self.compute_measurement_likelihood(i)

        # TODO update best_estimate based on the weights.
        return self.best_estimate


    def compute_measurement_likelihood(self, particle_index:int):
        """
        Determine the likelihood of a specific particle using the observation.
        """
        pass
        # NOTE the observation model gives an expected BEV of the environment. This is not limited to the robot's line-of-sight. As such, we will not use raycasting for particle evaluation, but rather a similarity check of the observation overlaid on the environment.


    def resample(self):
        """
        Use the weights vector to sample from the population and form the next generation.
        """
        # Save off the most likely particle as our filter output.
        i = 0 # TODO get index of highest weight.
        # self.best_estimate = self.particle_set[i,:]
        # TODO Sample from weights to form most of the population.
        # TODO Randomly generate small portion of population to prevent particle depletion.
