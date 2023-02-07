#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <ros/ros.h>
#include "geometry_msgs/Vector3.h"
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>

#define pi 3.14159265358979323846

class ParticleFilter {
    public:
    // current timestep.
    int timestep = 0;
    // state distribution.
    Eigen::MatrixXf particle_set;

    ParticleFilter();
    void propagateParticles(const geometry_msgs::Vector3::ConstPtr& msg);
};

#endif // PARTICLE_FILTER_H