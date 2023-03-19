#!/usr/bin/env python3

"""
Particle Filter class implementation.
Can separately process its prediction and update steps at different, independent rates, and can be polled for the most likely particle estimate at any time.
"""

import rospkg, yaml
import numpy as np
from math import pi, sin, cos, remainder, tau

from scripts.cmn_utilities import clamp, ObservationGenerator

class ParticleFilter:
    # GLOBAL VARIABLES
    num_particles = None
    state_size = None
    particle_set = None
    particle_weights = None
    # NOTE Map scale and observation scale are assumed known for now. The former will eventually be estimated with another filter.
    obs_gen = None
    # FILTER OUTPUT
    best_weight = 0
    best_estimate = None


    def __init__(self):
        """
        Instantiate the particle filter object and set its params from the config yaml.
        """
        # Determine filepath.
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('cmn_pkg')
        # Open the yaml and get the relevant params.
        with open(pkg_path+'/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            # Particle filter params.
            self.num_particles = int(config["particle_filter"]["num_particles"])
            self.state_size = int(config["particle_filter"]["state_size"])

        # Init things with the correct dimensions.
        self.particle_set = np.zeros((self.num_particles, self.state_size))
        self.particle_weights = np.zeros(self.num_particles)
        self.best_estimate = np.zeros(self.state_size)

        # Init the utilities class for doing coord transforms and observation comparisons.
        self.obs_gen = ObservationGenerator()


    def set_map(self, map):
        """
        Get the occupancy grid map, and save it to use each iteration.
        @param map, a 2D array containing the processed occupancy grid map.
        """
        self.obs_gen.set_map(map)


    def propagate_particles(self, fwd, ang):
        """
        Given a relative motion since last iteration, apply this to all particles.
        @param fwd, commanded forward motion in meters.
        @param ang, commanded angular motion in radians (CCW).
        """
        for i in range(self.num_particles):
            self.particle_set[i,0] += fwd * cos(self.particle_set[i,2])
            self.particle_set[i,1] += fwd * sin(self.particle_set[i,2])
            # Keep yaw normalized to (-pi, pi).
            self.particle_set[i,2] = remainder(self.particle_set[i,2] + ang, tau)


    def update_with_observation(self, observation):
        """
        Use an observation to evaluate the likelihood of all particles, and update the filter estimate.
        @param observation, 2D numpy array of the observation for this iteration.
        @return 3x1 numpy vector of best particle estimate (x,y,yaw).
        """
        for i in range(self.num_particles):
            # For a certain particle, extract the region it would have given as an observation.
            obs_img_expected = self.obs_gen.extract_observation_region(self.particle_set[i,:])
            # Compare this to the actual observation to evaluate this particle's likelihood.
            self.particle_weights[i] = self.compute_measurement_likelihood(obs_img_expected, observation)

        # Find best particle this iteration.
        i_best = np.argmax(self.particle_weights)
        # Decay likelihood of current estimate.
        self.best_weight *= 0.99
        # Update our filter estimate.
        if self.particle_weights[i_best] > self.best_weight:
            self.best_weight = self.particle_weights[i_best]
            self.best_estimate = self.particle_set[i,:]
        return self.best_estimate


    def compute_measurement_likelihood(self, obs_expected, obs_actual) -> float:
        """
        Determine the likelihood of a specific particle.
        @param obs_expected, 2D numpy array of the observation we expect given a specific particle.
        @param obs_actual, 2D numpy array of the observation we actually got this iteration.
        @return float, likelihood of this particle given the expected vs actual observations.
        """
        # NOTE the observation model gives an expected BEV of the environment. This is not limited to the robot's line-of-sight. As such, we will not use raycasting for particle evaluation, but rather a similarity check of the observation overlaid on the environment.
        # TODO
        return 0.0


    def resample(self):
        """
        Use the weights vector to sample from the population and form the next generation.
        """
        # TODO Sample from weights to form most of the population.
        # TODO Randomly generate small portion of population to prevent particle depletion.
        pass
