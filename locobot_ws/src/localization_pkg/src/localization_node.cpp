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

void readParams() {
    ///\todo: read config parameters from yaml file.
    ///\todo: read in map.
}

void commandCallback(const geometry_msgs::Vector3::ConstPtr& msg) {
    ///\todo: rather than using commanded vel, actually figure out how much the robot moved between last observation and this new one. maybe use odom instead of commands.
    // Get the robot motion since last iteration, and run the predict step of the pf.
    pf.propagateParticles(msg);
}

void observationCallback(const sensor_msgs::Image::ConstPtr& msg) {
    // Get an "observation" from the perception node's processing of sensor data.
    // Run the particle filter's update set.
    ///\todo: change input type from msg pointer to an actual matrix type.
    pf.updateWithObservation(msg);
    ///\todo: Get the best particle estimate from the filter.
    geometry_msgs::Vector3 pose_estimate;
    statePub.publish(pose_estimate);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "localization_node");
    ros::NodeHandle node("~");

    // read config parameters.
    readParams();

    // subscribe to filter inputs.
    ros::Subscriber commandSub = node.subscribe("/command", 100, commandCallback);
    ros::Subscriber observationSub = node.subscribe("/observation", 1, observationCallback);
    // publish most likely particle as our localization estimate.
    statePub = node.advertise<geometry_msgs::Vector3>("state/particle_filter", 1);

    ros::spin();
    return 0;
}