#include "localization_pkg/particle_filter.hpp"

ParticleFilter::ParticleFilter() {
    ///\todo: randomly distribute initial particle set uniformly over the free space on the map.
    particle_set.setZero(num_particles, state_size);
    // Set all weights to 0.
    particle_weights.setZero(num_particles);
    ///\todo: get the map somehow. maybe make separate function to set it that will be called from readParams.
}

void ParticleFilter::propagateParticles(const geometry_msgs::Vector3::ConstPtr& msg) {
    ///\todo: apply this motion to all particles.
}

void ParticleFilter::updateWithObservation(const sensor_msgs::Image::ConstPtr& msg) {
    ///\todo: save observation to latest_observation variable.
    // Compute weights of all particles using this new observation.
    for (uint i = 0; i < num_particles; ++i) {
        computeMeasurementLikelihood(i);
    }
    ///\todo: Grab the highest weight particle to return as the filter estimate.
    // Form next generation using the weights.
    resample();
    ///\todo: Make this function return a pose estimate.
}

void ParticleFilter::computeMeasurementLikelihood(const uint particle_index) {
    ///\todo: compute likelihood of a specific particle using the newest observation.
}

void ParticleFilter::resample() {
    ///\todo: form next generation of particles using the weights.
}

///\todo: Will need a function to transform coords on observation to coords on map given a particle pose.
