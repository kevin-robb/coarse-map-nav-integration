#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <ros/ros.h>
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
};

#endif // PARTICLE_FILTER_H