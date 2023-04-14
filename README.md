# Topological Localization on Course Maps
This work is my Thesis project @ Northeastern University for my M.S. in Robotics, advised by Lawson Wong.

## Overview
This project is an application of a research concept from Chengguang Xu, Lawson Wong, and Chris Amato. The idea is to take sensor data from a mobile robot, such as RGB-D images, and feed it through a trained network to yield a topological observation. This observation could be as simple a 3x3 binary image reflecting the basic large-scale environmental features around the robot. We then use some kind of Bayesian filter to match this to the rough ("coarse") map and attempt to localize.

The map we have will contain much less information than a highly curated, high-resolution map that is usually assumed for research-y localization & navigation tasks.
 1. Proportions will likely be close to correct but may be inaccurate.
 2. Some features in a hand-drawn map may differ from the map a robot would create with SLAM. i.e., furniture may appear in one but not the other.
 3. Map scale is unknown.

Due to point (2) in particular, it is possible for the vehicle to be "in collision" as far as the coarse map is concerned, as the robot may be under a table, or some environmental features may have moved since the map was created. We must be robust to this fact with our localization and path planning.

Some methods of generating said maps for this project are:
 - Creating a good map (using, say, Cartographer-ROS to run LiDAR-SLAM), and "coarse-ifying" it. i.e., downsampling it to a lower resolution, and perhaps distorting it. This will be the best-case-scenario assuming nothing in the environment has changed since the map was created, as it directly reflects what the robot can see and perceive as occluded.
 - Pulling building schematics for the environment in question. This will likely result in a sparser map than reality, as only permanent fixtures like walls and pillars will appear, and furniture will be absent.
 - Having a bored grad student draw a map in less than 2 minutes. This option will likely be more representative of the real space, but proportions cannot be guaranteed. Using this option also forces us to estimate the scale ourselves, since there is no assumed map <--> world transformation built in.

The basic architecture consists of three nodes:
 - Perception node: subscribes to sensor data, feeds it through the ML model, and publishes an observation.
 - Localization node: subscribes to observations and commands, runs a filter to localize, and publishes the robot pose estimate.
 - Motion planning node: subscribes to pose estimates (and the map), does path planning, navigation, and control, and publishes Twist commands to the robot.

My approach for the localization node is as follows:
 - We will use a particle filter for the main state estimator. A particle filter allows us to keep a continuous, non-Gaussian estimate of the robot pose, which is adaptable & forgiving to some inaccuracies in the map.
 - Rather than doing raycasting (the standard for particle filters), we will overlay the 2D observation on the map (relative to a certain particle) and compare to the intensities of the map in each overlapping cell. Similarity implies that the particle is likely, and that our current scale estimate of the map is likely good, too.
 - Commands sent to the robot will be used to propagate all particles during the prediction stage.
 - Alongside the particle filter, we'll run a simple Kalman filter to estimate the map scale, feeding in the highest likelihood determined in the PF, which we want to maximize.

## Simulation with Generated Data
For initial setup & testing, I've created an additional node which replaces the perception node. This simulation node keeps track of the "true" vehicle pose w.r.t. the provided map. Here we assume the coarse map is exactly equal to the true environment that's used to generate observations.
 1. It subscribes to velocity commands, and uses these to propagate forward the true vehicle pose (with some noise). 
 2. Then, using the vehicle pose and the map, we extract a region from the robot's perspective corresponding to the defined observation coordinates. It will potentially be scaled to a lower resolution with bilinear interpolation depending on specified parameters in `config.yaml`. 
 3. The sim node publishes this observation, which is then used to localize on the map. Since this is a best-case-scenario for the localization node, we can use it to tune and evaluate our localization method's feasibility. (For now we also assume the scale is known, but later we'll need to estimate this as well with its own filter.)
 4. The motion planning node receives the localization estimate, determines some Twist command for the robot, and publishes it. The sim node subscribes to this, and the cycle continues.

## Simulation with Real Data
To bring the project another step closer to working on a real robot, we can interface with a turtlebot. For this project, we're using an [interbotix LoCoBot](https://docs.trossenrobotics.com/interbotix_xslocobots_docs/getting_started/user_guide.html), which uses a Kobuki base and has a planar LiDAR, a D435 RealSense camera, and a wx200 5DOF manipulator arm. (For this project, we're not using the arm.) To use our simulator with real data, we will do the following:
 1. Run our existing code while the robot is online and connected via ROS, so all velocity commands we publish will actually move the robot.
 2. While this is happening, we collect a rosbag.
 3. Afterwards, we use Cartograhper-ROS to run LiDAR-SLAM with this bag to generate a good 2D occupancy map of the area.
 4. Then, we can use the generated map as a coarse map to run our simulation without the robot as before.

We can draw a coarse map of the same area used to generate this map, allowing us to test localizing on a different map than the one being used to generate observations. This allows us to confirm the robustness of our method to innacurate knowledge of the environment.

## Training the ML Model with Real Data
Ultimately, we need a way to get observations without relying on knowing the "true" map of the area. This is the purpose of Chengguang's ML model, though we must re-train it to our set of sensors and our robotic platform.

To gather data for re-training the model and for evaluating the overall effectiveness of the method, we can do the following:
 1. Run Cartographer-ROS to generate a high-resolution map of some environment of choice.
 2. Manually cleanup the map to ensure it is as high-quality and close to accurate as possible.
 3. Drive the robot around in the environment, saving all data from the RealSense sensor and all timestamped-commands sent to the robot. For each RS image, we will know the corresponding "ground-truth" robot pose on the map.
 4. For each measurement+pose pair, extract an ROI in front of the robot and process it to create the "ground-truth" observation corresponding to this place.
 5. We now have matching pairs of inputs and outputs for the model which we can use to train it.
 6. Repeat this several times to get as much data as we can, and to get many saved runs that we can play-back to test the model and the localization method. (This is essentially using real life as a simulator.)
 7. After training and validating, run the robot in the same environment again to validate real-world performance on new data.
