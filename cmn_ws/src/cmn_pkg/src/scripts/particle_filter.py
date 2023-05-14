#!/usr/bin/env python3

"""
Particle Filter class implementation.
Can separately process its prediction and update steps at different, independent rates, and can be polled for the most likely particle estimate at any time.
"""

import rospkg, yaml
import numpy as np
from math import sin, cos, remainder, tau
from random import choices

from scripts.map_handler import MapFrameManager
from scripts.basic_types import PoseMeters, PosePixels

class ParticleFilter:
    # Config params.
    num_particles = None
    state_size = None
    num_to_resample_randomly = None
    # Utility class.
    mfm = None
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
        # TODO init particles randomly rather than all at 0.
        self.particle_set = np.zeros((self.num_particles, self.state_size))
        self.particle_weights = np.zeros(self.num_particles)
        self.best_estimate = np.zeros(self.state_size)

    def set_map_frame_manager(self, mfm:MapFrameManager):
        """
        Set our reference to the map frame manager, which allows us to use the map and coordinate transform functions.
        @param mfg MapFrameManager instance that has already been initialized with a map.
        """
        self.mfm = mfm

    def propagate_particles(self, fwd:float, ang:float):
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
        # Propagate the overall filter estimate as well.
        if self.best_estimate is not None:
            self.best_estimate[0] += fwd * cos(self.best_estimate[2])
            self.best_estimate[1] += fwd * sin(self.best_estimate[2])
            # Keep yaw normalized to (-pi, pi).
            self.best_estimate[2] = remainder(self.best_estimate[2] + ang, tau)


    def update_with_observation(self, observation) -> PoseMeters:
        """
        Use an observation to evaluate the likelihood of all particles, and update the filter estimate.
        @param observation, 2D numpy array of the observation for this iteration.
        @return PoseMeters of best particle estimate (x,y,yaw).
        """
        # If there is no observation, take evasive maneuvers to prevent a runtime error. This should never happen though.
        if observation is not None:
            for i in range(self.num_particles):
                # For a certain particle, extract the region it would have given as an observation.
                obs_img_expected, _ = self.mfm.extract_observation_region(PoseMeters(self.particle_set[i,0], self.particle_set[i,1], self.particle_set[i,2]))
                # Compare this to the actual observation to evaluate this particle's likelihood.
                self.particle_weights[i] = self.compute_measurement_likelihood(obs_img_expected, observation)
                # NOTE these likelihoods are intentionally NOT normalized.

        # Find best particle this iteration.
        i_best = np.argmax(self.particle_weights)
        # Decay likelihood of current estimate.
        # self.best_weight *= 0.99
        # Update our filter estimate.
        if self.particle_weights[i_best] > self.best_weight:
            self.best_weight = self.particle_weights[i_best]
            self.best_estimate = self.particle_set[i_best,:]
        # Convert filter estimate to desired data type and return.
        return PoseMeters(self.best_estimate[0], self.best_estimate[1], self.best_estimate[2])


    def compute_measurement_likelihood(self, obs_expected, obs_actual) -> float:
        """
        Determine the likelihood of a specific particle.
        @param obs_expected - 2D numpy array of the observation we expect given a specific particle.
        @param obs_actual - 2D numpy array of the observation we actually got this iteration.
        @return float - likelihood of this particle given the expected vs actual observations.
        """
        # If the particle was too close to the edge of the map and failed to generate an observation image, kill it.
        if obs_expected is None:
            return 0.0
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

        # Ensure weights vector is not all zeros.
        if sum(self.particle_weights) == 0:
            self.particle_weights = [1 for _ in range(len(self.particle_weights))]

        # Sample from weights to form most of the population.
        selected_indices = choices(self.all_indices, list(self.particle_weights), k=self.num_particles - self.num_to_resample_randomly)
        for i_new, i_old in enumerate(selected_indices):
            # Get the particle whose index was chosen based on its weight.
            new_particle_set[i_new,:] = self.particle_set[i_old,:]
            # TODO Perturb it with some noise.

        # Randomly generate small portion of population to prevent particle depletion.
        for i in range(self.num_particles - self.num_to_resample_randomly, self.num_particles):
            # Do not attempt to use the utilities class until the map has been processed.
            if self.mfm.initialized:
                new_particle_set[i,:] = self.mfm.generate_random_valid_veh_pose().as_np_array()
            else:
                new_particle_set[i,:] = np.zeros(self.state_size)

        # Update the set of particles.
        self.particle_set = new_particle_set


    def get_particle_set_px(self):
        """
        Convert the particle set into a list of PosePixels to return. Used for visualization.
        """
        return [self.mfm.transform_pose_m_to_px(PoseMeters(self.particle_set[i,0], self.particle_set[i,1], self.particle_set[i,2])) for i in range(self.num_particles)]
            
