#!/usr/bin/env python3

"""
Particle Filter class implementation.
Can separately process its prediction and update steps at different, independent rates, and can be polled for the most likely particle estimate at any time.
"""

import rospkg, yaml
import numpy as np
from math import sin, cos, remainder, tau
from random import choices

from scripts.cmn_utilities import ObservationGenerator

class ParticleFilter:
    # Config params.
    num_particles = None
    state_size = None
    num_to_resample_randomly = None
    # Utility class.
    obs_gen = None
    # Ongoing state.
    particle_set = None
    particle_weights = None
    # NOTE Map scale is assumed known for now, but it will eventually be estimated with another filter.
    # Filter output.
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
            self.all_indices = list(range(self.num_particles))
            self.state_size = int(config["particle_filter"]["state_size"])
            random_sampling_rate = config["particle_filter"]["random_sampling_rate"]
            self.num_to_resample_randomly = int(random_sampling_rate * self.num_particles)

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
        print("Propagating all particles by ({:}, {:})".format(fwd, ang))
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
            obs_img_expected, _ = self.obs_gen.extract_observation_region(self.particle_set[i,:])
            # Compare this to the actual observation to evaluate this particle's likelihood.
            self.particle_weights[i] = self.compute_measurement_likelihood(obs_img_expected, observation)
            # NOTE these likelihoods are intentionally NOT normalized.

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
        likelihood = 1.0
        for i in range(obs_expected.shape[0]):
            for j in range(obs_expected.shape[1]):
                diff = abs(obs_expected[i,j] - obs_actual[i,j])
                # want to encourage diff -> 0.
                likelihood *= (1.0 - diff)
        return likelihood


    def resample(self):
        """
        Use the weights vector to sample from the population and form the next generation.
        """
        new_particle_set = np.zeros((self.num_particles, self.state_size))

        # Sample from weights to form most of the population.
        selected_indices = choices(self.all_indices, list(self.particle_weights), k=self.num_particles - self.num_to_resample_randomly)
        for i_new, i_old in enumerate(selected_indices):
            # Get the particle whose index was chosen based on its weight.
            new_particle_set[i_new,:] = self.particle_set[i_old,:]
            # TODO Perturb it with some noise.

        # Randomly generate small portion of population to prevent particle depletion.
        for i in range(self.num_particles - self.num_to_resample_randomly, self.num_particles):
            # Do not attempt to use the utilities class until the map has been processed.
            if self.obs_gen.initialized:
                new_particle_set[i,:] = self.obs_gen.generate_random_valid_veh_pose()
            else:
                new_particle_set[i,:] = np.zeros(self.state_size)

        # Update the set of particles.
        self.particle_set = new_particle_set
            
