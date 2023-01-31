// File that will handle the ROS interface of the filter, which is implemented in its own hpp/cpp file.

#include "ros/ros.h"
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Float32MultiArray.h"
#include <queue>
#include <iostream>
#include <string>
#include <fstream>
#include <ros/package.h>

#include "localization_pkg/ekf.hpp"

// define queues for messages.
// std::queue<base_pkg::Command::ConstPtr> cmdQueue;
std::queue<std_msgs::Float32MultiArray::ConstPtr> lmMeasQueue;
// define state estimate publisher.
ros::Publisher statePub;
// define EKF object from ekf.cpp class.
EKF ekf;

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

// void cmdCallback(const base_pkg::Command::ConstPtr& msg) {
//     // receive an odom command and add to the queue.
//     cmdQueue.push(msg);
// }

void lmMeasCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    // just for testing a subscriber, need to figure out msg type for actual inputs.
    // receive a landmark detection and add to the queue.
    lmMeasQueue.push(msg);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "localization_node");
    ros::NodeHandle node("~");

    // read config parameters.
    float DT = readParams();
    // get the initial veh pose and init the ekf.
    // ros::Subscriber initSub = node.subscribe("/truth/init_veh_pose", 1, initCallback);

    // subscribe to EKF inputs.
    // ros::Subscriber cmdSub = node.subscribe("/command", 100, cmdCallback);
    ros::Subscriber lmMeasSub = node.subscribe("/landmark", 100, lmMeasCallback);
    // publish EKF state.
    statePub = node.advertise<geometry_msgs::Vector3>("/state/ekf", 1);

    // timer to update EKF at set frequency.
    ros::Timer ekfIterationTimer = node.createTimer(ros::Duration(DT), &ekfIterate, false);

    ros::spin();
    return 0;
}