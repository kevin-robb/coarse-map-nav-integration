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


