// File that will handle the ROS interface of the filter, which is implemented in its own hpp/cpp file.

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Float32MultiArray.h"
#include <queue>
#include <iostream>
#include <string>
#include <fstream>
#include <ros/package.h>

#include "localization_pkg/particle_filter.hpp"

// define state estimate publisher.
ros::Publisher statePub;
// define particle filter instance.
ParticleFilter pf;

float readParams() {
    ///\todo: read config parameters from yaml file.
    return 0.1;
}

void ekfIterate(const ros::TimerEvent& event) {
    ///\todo: perform an iteration of the filter for this timestep and publish the localization estimate.
    geometry_msgs::Vector3 state_msg;
    state_msg.x = 1;
    state_msg.y = 1;
    state_msg.z = 1; // yaw in radians.
    statePub.publish(state_msg);
}

void commandCallback(const geometry_msgs::Vector3::ConstPtr& msg) {
    // Get a commanded motion, and use it to propagate all particles forward.
    ///\todo: Propagate all particles.
}

void observationCallback(const sensor_msgs::Image::ConstPtr& msg) {
    // Get an "observation" from the perception node's processing of sensor data.
    ///\todo: Run the measurement likelihood for the particle filter.
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "localization_node");
    ros::NodeHandle node("~");

    // read config parameters.
    float DT = readParams();
    // get the initial veh pose and init the ekf.
    // ros::Subscriber initSub = node.subscribe("/truth/init_veh_pose", 1, initCallback);

    // subscribe to filter inputs.
    ros::Subscriber commandSub = node.subscribe("/command", 100, commandCallback);
    ros::Subscriber observationSub = node.subscribe("/observation", 100, observationCallback);
    // publish most likely particle as our localization estimate.
    statePub = node.advertise<geometry_msgs::Vector3>("state/particle_filter", 1);

    // timer to update EKF at set frequency.
    ros::Timer ekfIterationTimer = node.createTimer(ros::Duration(DT), &ekfIterate, false);

    ros::spin();
    return 0;
}