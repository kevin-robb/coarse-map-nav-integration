# Topological Localization on Course Maps

This project is an application of a research concept from Chengguang Xu, Lawson Wong, and Chris Amato. The idea is to take sensor data from a mobile robot, say 4 RGB-D images creating a panorama, and feed it through a trained network to yield a simple topological observation. This observation could be simply a 3x3 binary image reflecting the basic large-scale environmental features around the robot. We then use some kind of Bayesian filter to match this to the rough map and attempt to localize.

The map we have will contain much less information than a highly curated, high-resolution map that is usually assumed for research-y localization & navigation tasks.
 - Proportions will likely be close to correct but may be inaccurate.
 - Some features in a hand-drawn map may differ from the map a robot would create with SLAM.
 - Map scale is unknown.

Some methods of generating said maps are:
 - Creating a good map (using, say, Cartographer to run LiDAR-SLAM), and "coarse-ifying" it. i.e., downsampling it to a lower resolution, and perhaps distorting it.
 - Pulling building schematics for the environment in question. This will mean things like furniture that would appear on a robot-created map will not appear.
 - Having a bored grad student draw a map in less than 2 minutes.

The basic architecture consists of three nodes:
 - Perception node: subscribes to sensor data, feeds it through the model, and publishes an observation.
 - Localization node: subscribes to observations, runs a filter to localize, and publishes the robot pose estimate.
 - Motion planning node: subscribes to pose estimates and the map, does path planning, navigation, and control, and publishes Twist commands to the robot.

To gather data for re-training the model and for evaluating the overall effectiveness of the method, we can do the following:
 - Run LiDAR-SLAM to generate a high-resolution map of some environment of choice.
 - Manually cleanup the map to ensure it is as high-quality and close to accurate as possible.
 - Drive the robot around in the environment, saving all data from the RealSense sensor. For each RS image, we will know the corresponding "ground-truth" robot pose on the map.
 - For each measurement+pose pair, extract an ROI in front of the robot and process it to create the "ground-truth" observation corresponding to this place.
 - We now have matching pairs of inputs and outputs for the model which we can use to train it.
 - Repeat this several times to get as much data as we can, and to get many saved runs that we can play-back to test the model and the localization method. (This is essentially using real life as a simulator.)
 - After training and validating, run the robot in the same environment again to validate real-world performance on new data.

My approach for the localization node is as follows:
 - We will use a particle filter for the main state estimator. A particle filter allows us to keep a continuous, non-Gaussian estimate of the robot pose, which is adaptable & forgiving to some inaccuracies in the map.
 - Rather than doing raycasting (the standard for particle filters), we will overlay the 2D observation on the map (relative to a certain particle) and compare to the intensities of the map in each overlapping cell. Similarity implies that the particle is likely, and that our current scale estimate of the map is likely good, too.
 - Alongside the particle filter, we'll run a simple Kalman filter to estimate the map scale, feeding in the highest likelihood determed in the PF, which we want to maximize.

For now, we are using a turtlebot as our platform, but we hope this will generalize to a more complex mobile robot in the future.

This work is my MS Project/Thesis @ Northeastern University, advised by Lawson Wong.
