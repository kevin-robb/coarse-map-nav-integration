// File that will handle the ROS interface of the filter, which is implemented in its own hpp/cpp file.

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "geometry_msgs/Vector3.h"
#include <iostream>
#include <string>
#include <fstream>
#include <ros/package.h>

#include "localization_pkg/particle_filter.hpp"

/////// GLOBAL VARIABLES ///////
ros::Publisher statePub;
ParticleFilter pf;
////////////////////////////////

float readParams() {
    ///\todo: read config parameters from yaml file.
    return 0.1;
}

void commandCallback(const geometry_msgs::Vector3::ConstPtr& msg) {
    ///\todo: rather than using commanded vel, actually figure out how much the robot moved between last observation and this new one. maybe use odom instead of commands.
    // Get a commanded motion, and use it to propagate all particles forward.
    pf.propagateParticles(msg);
}

void observationCallback(const sensor_msgs::Image::ConstPtr& msg) {
    // Get an "observation" from the perception node's processing of sensor data.
    ///\todo: Run the measurement likelihood for the particle filter.
    ///\todo: Perform an iteration of the filter for this timestep and publish the localization estimate.
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "localization_node");
    ros::NodeHandle node("~");

    // read config parameters.
    float DT = readParams();

    // subscribe to filter inputs.
    ros::Subscriber commandSub = node.subscribe("/command", 100, commandCallback);
    ros::Subscriber observationSub = node.subscribe("/observation", 1, observationCallback);
    // publish most likely particle as our localization estimate.
    statePub = node.advertise<geometry_msgs::Vector3>("state/particle_filter", 1);

    ros::spin();
    return 0;
}