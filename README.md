# Topological Localization on Coarse/Hand-Drawn Maps
This work is my Thesis project @ Northeastern University for my M.S. in Robotics, advised by Lawson Wong. My completed thesis report and slides from the defense can be found in the `docs/` folder. This folder also contains a full setup guide for the LoCoBot, the physical robot used for this project.

## Overview
This project is a implementation of Coarse Map Navigation (CMN) on a physical robot to demonstrate that the principle of this pipeline can work in the real world. We also extend CMN in several aspects, including expanded localization to estimate orientation, alternative perception methods, and a continuous state/action-space proof-of-concept.

The original CMN is a research concept from Chengguang Xu, Lawson Wong, and Chris Amato, and was demonstrated using the Habitat simulator in 2022-2023. The general architecture involves extracting local occupancy measurements from the environment, localizing position using a discrete Bayesian filter, and commanding motion via discrete actions. The local occupancy measurements were generated as predictions from a trained deep neural network that took an RGB panorama as input; we include this same network within our codebase and evaluate its effectiveness in the real world.

The coarse map will contain much less information than a highly curated, high-resolution map that is usually assumed for modern localization & navigation tasks.
 1. Proportions will likely be close to correct but may be inaccurate.
 2. Some features in a hand-drawn map may differ from the map a robot would create with SLAM. i.e., furniture may appear in one but not the other.
 3. Map scale is unknown.

However, the one guarantee with the coarse map (that is required for CMN to work) is local topological accuracy. e.g., following a particular wall leads to a corner in both the real world and in the map, a path exists in both between two points, etc.

Due to point (2) in particular, it is possible for the vehicle to be "in collision" as far as the coarse map is concerned, as the robot may be under a table, or some environmental features may have moved since the map was created. We must be robust to this fact with our localization and path planning.

Some methods of generating said maps for this project are:
 - Creating a good map (using, say, Cartographer-ROS to run LiDAR-SLAM), and "coarse-ifying" it. i.e., downsampling it to a lower resolution, and perhaps distorting it. This will be the best-case-scenario assuming nothing in the environment has changed since the map was created, as it directly reflects what the robot can see and perceive as occluded.
 - Pulling building schematics for the environment in question. This will likely result in a sparser map than reality, as only permanent fixtures like walls and pillars will appear, and furniture will be absent.
 - Having a bored grad student draw a map in less than 2 minutes. This option will likely be more representative of the real space, but proportions cannot be guaranteed. Using this option has no map <--> world transformation we could implicitly use, unlike the other cases.


## Running it
This project includes a simple simulator, which can track the robot pose on a given map and provide local occupancy measurements to the filter. This skips the first stage of the actual project, essentially allowing us to validate localization and path planning with "perfect" perception.

Several params can be altered in `cmn_ws/src/cmn_pkg/config/config.yaml`.

To run the project in the simulator without ROS, use the provided script:
```
python3 src/cmn_pkg/src/non_ros_runner.py -m <run_mode>
```
where `<run_mode>` can be one of `discrete`, `discrete_random`, or `continuous`.

To run the perception model (trained neural network) from the original CMN on the sets of RGB panoramas we've gathered, use the provided script:
```
python3 src/cmn_pkg/src/scripts/cmn/run_model_on_saved_pano.py -m ./cmn_ws/src/cmn_pkg/src/scripts/cmn/model/trained_local_occupancy_predictor_model.pt -d ./cmn_ws/src/cmn_pkg/data
```

For the remaining run methods, you must be using ROS Noetic, be connected to the LLPR network, and be able to ssh into the robot.
On the robot, start bringup so it will begin publishing sensor data and odometry, and subscribing to motor commands.
```
ssh locobot
start_all
```
Note that `start_all` is an ssh alias I created on the robot, which is equivalent to:
```
roslaunch interbotix_xslocobot_control xslocobot_control.launch robot_model:=locobot_wx200 use_base:=true use_camera:=true use_lidar:=true
```
More setup information on the locobot is available in this repository in `docs/locobot_setup.md`.

Now that the robot is running, you will start the remaining ROS nodes on your host PC. If using depth data is enabled in the config, you must start the depth processor nodelet using the launch file in this repo:
```
cd cmn_ws
catkin_make
source devel/setup.bash
roslaunch cmn_pkg start_depth_proc.launch
```

Optional: To view the local occupancy measurements being created live from LiDAR and depth data, run the standalone node in a new terminal:
```
cd cmn_ws
catkin_make
source devel/setup.bash
roslaunch cmn_pkg test_lidar.launch
```

You can now start the main CMN runner in a new terminal:
```
cd cmn_ws
catkin_make
source devel/setup.bash
roslaunch cmn_pkg discrete_with_viz.launch
```

And now the robot should be driving around, performing CMN, and showing its visualization on your host PC! Note this launch file has some params you can change on the command line when running it.


