#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <ros/ros.h>
#include "geometry_msgs/Vector3.h"
#include "sensor_msgs/Image.h"
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>

#define pi 3.14159265358979323846

class ParticleFilter {
    public:
    // Config params.
    ///\todo: get from yaml.
    uint num_particles = 1000;
    uint state_size = 3;

    // Map to localize on.
    ///\todo: get path from yaml and read the image somehow.
    ///\todo: choose different datatype like cv::Mat or Eigen::MatrixXf.
    sensor_msgs::Image map;

    // State distribution: (x,y,yaw) for each particle.
    Eigen::MatrixXf particle_set;
    Eigen::VectorXf particle_weights;

    // Current estimate of the map scale in meters/cell.
    ///\todo: should make this scale of map relative to observation?
    float scale = 0.01;

    // Storage to prevent needing to pass as function param.
    ///\todo: maybe should just pass by reference after all.
    sensor_msgs::Image latest_observation;

    ParticleFilter();
    void propagateParticles(const geometry_msgs::Vector3::ConstPtr& msg);
    void updateWithObservation(const sensor_msgs::Image::ConstPtr& msg);
    void computeMeasurementLikelihood(const uint particle_index);
    void resample();
};

#endif // PARTICLE_FILTER_H