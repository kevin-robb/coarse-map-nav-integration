#!/usr/bin/env python3

"""
Particle Filter class implementation.
Can separately process its prediction and update steps at different, independent rates, and can be polled for the most likely particle estimate at any time.
"""

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
        pass
    

    def set_params(self):
        """
        Set particle filter parameters from the yaml.
        """
        pass
    

    def set_map(self, map):
        """
        Get the occupancy grid map, and save it to use each iteration.
        """
        self.occ_map = map


    def propagate_particles(self, motion):
        """
        Given a relative motion since last iteration, apply this to all particles.
        """
        pass


    def update_with_observation(self, observation):
        """
        Use an observation to evaluate the likelihood of all particles.
        """
        pass
        # TODO loop through all particles.
        # self.particle_weights[i] = self.compute_measurement_likelihood(i)


    def compute_measurement_likelihood(self, particle_index:int):
        """
        Determine the likelihood of a specific particle using the observation.
        """
        pass


    def resample(self):
        """
        Use the weights vector to sample from the population and form the next generation.
        """
        # Save off the most likely particle as our filter output.
        i = 0 # TODO get index of highest weight.
        self.best_estimate = self.particle_set[i,:]
        # TODO Sample from weights to form most of the population.
        # TODO Randomly generate small portion of population to prevent particle depletion.
