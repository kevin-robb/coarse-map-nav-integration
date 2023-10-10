/**
 * @file motion_cmd_node.cpp
 * @brief Modified code from https://wiki.ros.org/pr2_controllers/Tutorials/Using%20the%20base%20controller%20with%20odometry%20and%20transform%20information
 * to subscribe to discrete commands from CMN and convert them into robot motion.
 */

#include <iostream>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/Twist.h>
#include <tf/transform_listener.h>

// Define relevant tf frames.
const std::string base_footprint_frame = "locobot/base_footprint";
const std::string odom_frame = "locobot/odom";

class RobotDriver
{
private:
  //! The node handle we'll be using
  ros::NodeHandle nh_;
  //! We will be publishing to the "cmd_vel" topic to issue commands
  ros::Publisher cmd_vel_pub_;
  //! We will be listening to TF transforms as well
  tf::TransformListener listener_;

public:
  /**
   * @brief ROS node initialization.
   * @param nh NodeHandle.
   */
  RobotDriver(ros::NodeHandle &nh)
  {
    nh_ = nh;
    //set up the publisher for the cmd_vel topic
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/locobot/mobile_base/commands/velocity", 1);
  }

 /**
  * @brief Drive forward a specified distance based on odometry information.
  * @param distance Distance in meters to move forwards.
  * @return whether motion was successful.
  */
  bool driveForwardOdom(double distance)
  {
    //wait for the listener to get the first message
    listener_.waitForTransform(base_footprint_frame, odom_frame, 
                               ros::Time(0), ros::Duration(1.0));
    
    //we will record transforms here
    tf::StampedTransform start_transform;
    tf::StampedTransform current_transform;

    //record the starting transform from the odometry to the base frame
    listener_.lookupTransform(base_footprint_frame, odom_frame, 
                              ros::Time(0), start_transform);
    
    //we will be sending commands of type "twist"
    geometry_msgs::Twist base_cmd;
    //the command will be to go forward at 0.25 m/s
    base_cmd.linear.y = base_cmd.angular.z = 0;
    base_cmd.linear.x = 0.25;
    
    ros::Rate rate(10.0);
    bool done = false;
    while (!done && nh_.ok())
    {
      //send the drive command
      cmd_vel_pub_.publish(base_cmd);
      rate.sleep();
      //get the current transform
      try
      {
        listener_.lookupTransform(base_footprint_frame, odom_frame, 
                                  ros::Time(0), current_transform);
      }
      catch (tf::TransformException ex)
      {
        ROS_ERROR("%s",ex.what());
        break;
      }
      //see how far we've traveled
      tf::Transform relative_transform = 
        start_transform.inverse() * current_transform;
      double dist_moved = relative_transform.getOrigin().length();

      if(dist_moved > distance) done = true;
    }
    if (done) return true;
    return false;
  }

 /**
  * @brief Receive a direction and amount to turn, and command it to the robot.
  * @param clockwise True to turn clockwise, false for counterclockwise.
  * @param radians Amount to turn in radians.
  * @return whether motion was successful.
  */
  bool turnOdom(bool clockwise, double radians)
  {
    while(radians < 0) radians += 2*M_PI;
    while(radians > 2*M_PI) radians -= 2*M_PI;

    std::cout << "turnOdom called with angle " << radians << std::endl;

    //wait for the listener to get the first message
    listener_.waitForTransform(base_footprint_frame, odom_frame, 
                               ros::Time(0), ros::Duration(1.0));
    
    //we will record transforms here
    tf::StampedTransform start_transform;
    tf::StampedTransform current_transform;

    //record the starting transform from the odometry to the base frame
    listener_.lookupTransform(base_footprint_frame, odom_frame, 
                              ros::Time(0), start_transform);
    
    //we will be sending commands of type "twist"
    geometry_msgs::Twist base_cmd;
    //the command will be to turn at 0.75 rad/s
    base_cmd.linear.x = base_cmd.linear.y = 0.0;
    base_cmd.angular.z = 0.75;
    if (clockwise) base_cmd.angular.z = -base_cmd.angular.z;
    
    //the axis we want to be rotating by
    tf::Vector3 desired_turn_axis(0,0,1);
    if (!clockwise) desired_turn_axis = -desired_turn_axis;
    
    ros::Rate rate(10.0);
    bool done = false;
    while (!done && nh_.ok())
    {
      //send the drive command
      cmd_vel_pub_.publish(base_cmd);
      rate.sleep();
      //get the current transform
      try {
        listener_.lookupTransform(base_footprint_frame, odom_frame, ros::Time(0), current_transform);
      }
      catch (tf::TransformException ex) {
        ROS_ERROR("%s",ex.what());
        break;
      }
      tf::Transform relative_transform = start_transform.inverse() * current_transform;
      tf::Vector3 actual_turn_axis = relative_transform.getRotation().getAxis();
      double angle_turned = relative_transform.getRotation().getAngle();
      std::cout << "angle turned so far is " << angle_turned << std::endl;
      if ( fabs(angle_turned) < 1.0e-2) continue;

      if ( actual_turn_axis.dot( desired_turn_axis ) < 0 ) 
        angle_turned = 2 * M_PI - angle_turned;

      if (angle_turned > radians) done = true;
    }
    std::cout << "Exited the publishing loop" << std::endl;
    if (done) return true;
    return false;
  }
};

// Define RobotDriver instance as global so callbacks can access it easier.
std::unique_ptr<RobotDriver> driver;
//! We will publish a bool when a motion is finished.
ros::Publisher motion_finished_pub_;

/**
 * @brief Receive a discrete motion command, and execute it.
 * @param msg 
 */
void discreteCommandCallback(const geometry_msgs::Twist::ConstPtr& msg) {
    std::cout << "got discrete command ------------------" << std::endl;
    // Twist encodes the discrete motion to execute.
    bool forward_motion = msg->linear.x != 0.0;
    bool angular_motion = msg->angular.z != 0.0;
    if (forward_motion ^ angular_motion) {
        std_msgs::Bool success;
        // Command the proper motion.
        if (forward_motion) {
            success.data = driver->driveForwardOdom(msg->linear.x);
        } else { // angular_motion
            // Positive turn angle indicates CCW.
            bool turn_cw = msg->angular.z < 0;
            success.data = driver->turnOdom(turn_cw, abs(msg->angular.z));
        }
        // Publish success status.
        motion_finished_pub_.publish(success);
    } else {
        std::cout << "Must provide forward component XOR angular component. Invalid message received." << std::endl;
    }
}

int main(int argc, char** argv)
{
    //init the ROS node
    ros::init(argc, argv, "motion_cmd_node");
    ros::NodeHandle nh;

    // Init the driver/cmd publisher.
    driver = std::make_unique<RobotDriver>(nh);

    // set up the publisher for declaring a motion finished.
    motion_finished_pub_ = nh.advertise<std_msgs::Bool>("/cmn/motion_finished", 1);
    // set up the subscriber to discrete commands.
    ros::Subscriber discreteCommandSub = nh.subscribe("/cmn/motion_command", 100, discreteCommandCallback);

    ros::spin();
    return 0;
}