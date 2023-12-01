# Robot Setup & Usage
## Locobot Overview
For our initial implementation and physical proof-of-concept, we're using an [interbotix LoCoBot](https://docs.trossenrobotics.com/interbotix_xslocobots_docs/getting_started/user_guide.html), which uses a Kobuki base and has a planar LiDAR, a D435 RealSense camera, and a wx200 5DOF manipulator arm. (For this project, we're not using the arm.)

<p align="center">
  <img src="images/wx200_locobot.png" alt="The locobot platform we are using"/>
  
  <i>The locobot platform we are using.</i>
</p>

We'll be using the LiDAR only for generating other SLAM maps for comparison, while the RealSense will be the primary sensor input used for our actual observation generation in this project.

The locobot is running Ubuntu 20.04 and ROS Noetic.

The [basic setup guide](https://docs.trossenrobotics.com/interbotix_xslocobots_docs/getting_started/user_guide.html) gives details about charging and basic setup for the robot. (Make sure to select the "Kobuki Version" tab.) Basic physical and software setup has already been completed for our robot, so I will not describe it in detail in this report.

Important to note: the locobot has two separate power supplies, so to use the robot you must turn on the Kobuki base, the external battery on the first tier, and then the computer on the second tier. 

## Connecting to the Robot
You can connect a mouse/keyboard and HDMI to the ports on the first tier to work on the robot directly. 
However, it has already been setup to connect to the LLPR WiFi network in the robotics lab, so you can use SSH to interface with the robot without needing extra peripherals. Connect your laptop to the LLPR network, 
<!-- LLPR password is llpr2016 iirc, probably don't want to publish it publicly tho -->
and determine the IP address of the locobot. You can then add a section to your `~/.ssh/config`:
```
Host locobot_llpr
  User locobot
  HostName 192.168.1.149
  StrictHostKeyChecking no
```
For the "Host" line, you choose an ssh alias that you'll be able to use on your machine to log into the robot from now on. I've chosen `locobot_llpr`. To complete setup of the ssh key, run:
```
ssh-copy-id locobot_llpr
```
and enter the password `locobot`.

## ROS Network Setup
We want to be able to run a single ROS instance with some nodes running on the locobot and some on our host PC. This can be accomplished by editing some environment variables used by ROS. Specifically, the following two variables:
- `ROS_MASTER_URI` - Controls what machine/IP will run roscore.
- `ROS_HOSTNAME` - Controls what this machine will use as host.

The default is for a machine to use its own IP for both of these. For the hostname, this is fine. For the master URI, we need to change it so that both machines use the same IP for master. 

Since we usually will run bringup on the robot first, which will start roscore, I have chosen to use it as master. So, in your terminal on the host PC, run the following. The IP address is that of the locobot on the LLPR network.
```
export ROS_MASTER_URI=http://192.168.1.149:11311
```

You will need to run this in each new terminal before starting any ROS processes. Or, you can add it to the `~/.bashrc` on your host PC so it will run automatically in every new terminal opened. 

Note that this will prevent ROS from being able to run on your machine alone, so you will need to comment it out and open a new terminal to do any ROS work without the locobot.

In `locobot_llpr:~/.bashrc`, verify/add the following lines:
```
source /opt/ros/noetic/setup.bash
source /home/locobot/realsense_ws/devel/setup.bash
source /home/locobot/apriltag_ws/devel_isolated/setup.bash
source /home/locobot/interbotix_ws/devel/setup.bash
export ROS_IP=$(echo `hostname -I | cut -d" " -f1`)
if [ -z "$ROS_IP" ]; then
        export ROS_IP=127.0.0.1
fi
export ROS_HOSTNAME=192.168.1.149
export ROS_MASTER_URI=http://192.168.1.149:11311
export ROS_IP=192.168.1.149
```

## Verifying ROS Connectivity
The [ROS 1 Quickstart Guide](https://docs.trossenrobotics.com/interbotix_xslocobots_docs/ros_interface/ros1/quickstart.html) has some information for checking that the environment is setup properly, but a simpler method is to view the sensor feeds on your host PC.
1. On the locobot, run bringup to get the sensors going.
```
roslaunch interbotix_xslocobot_control xslocobot_control.launch robot_model:=locobot_wx200 use_base:=true use_camera:=true use_lidar:=true
```
I have aliased this command to `start_all` on the locobot.
    
2. On the host PC, start the `rqt` utility by simply entering `rqt` in a terminal. Then select "Plugins" -> "Visualization" -> "Image View". Select image topics in the dropdown and you should see the RS camera's RGB and depth feeds. You may need to refresh if you launched `rqt` before the topics were being published.

<p align="center">
  <img src="images/RS_preview_kevin.png"/>
  
  <i>Testing the RealSense RGB image feed in rqt image_view.</i>
</p>

3.  On the host PC, start `rviz` by simply entering `rviz` in a terminal. It may complain that there is no `map` frame, so at the upper left, change the world frame to `locobot/base_link`. Then click "Add" in the lower left and select the LaserScan type. After adding it, ensure the topic is set to `/locobot/scan`, and you should see a live visualization of the current planar LiDAR pointcloud.

<p align="center">
  <img src="images/rviz_lidar_preview.png"/>
  
  <i>Testing the LiDAR data feed in rviz.</i>
</p>

## Running teleop with the robot
After starting roscore & bringup on the robot, run the following on the host PC:
```
rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/locobot/mobile_base/commands/velocity
```
You can now use the keyboard commands printed to the console to drive the locobot around.

## Getting a "Ground Truth" Map with Cartographer-ROS
### Setup
Install Cartographer on the locobot using [this guide](https://google-cartographer-ros.readthedocs.io/en/latest/compilation.html).
You will get an error during this process. The solution is to remove the line `<depend>libabsl-dev</depend>` from `src/cartographer/package.xml` and re-run the command that gave the error. This solution is described in [this github issue](https://github.com/cartographer-project/cartographer_ros/issues/1726).

Copy launch files and change specs for our specific case, as described [here](https://google-cartographer-ros.readthedocs.io/en/latest/your_bag.html).
The most crucial part of this is the rostopic remappings, as Cartographer expects messages on the `/scan` and `/imu` topics.

If you seem to be having topic issues, try running `rosnode info /cartographer_node` while it's running. This allows you to see, among other useful things, the exact topics it is subscribing to for scan, tf, and (if applicable) imu messages.

### Running it Live
1. Start bringup on the locobot to get all its sensors to begin publishing data. (This will also start roscore on the locobot.) Run the following, or use the alias `start_all`.
```
ssh locobot_llpr
roslaunch interbotix_xslocobot_control xslocobot_control.launch robot_model:=locobot_wx200 use_base:=true use_camera:=true use_lidar:=true
```

2. Now start Cartographer.
```
ssh locobot_llpr
cd ~/cartographer_ws
source install_isolated/setup.bash
roslaunch cartographer_ros my_robot.launch
```

Here there were complaints about various `tf` frames, so I ran these commands on my host PC (probably not all of which are strictly necessary):
```
rosrun tf static_transform_publisher 0 0 0 0 0 0 base_link camera_depth_frame 100
rosrun tf static_transform_publisher 0 0 0 0 0 0 base_link target_frame 100
rosrun tf static_transform_publisher 0 0 0 0 0 0 locobot/laser_frame_link laser 100
rosrun tf static_transform_publisher 0 0 0 0 0 0 locobot/laser_frame_link locobot/base_link 100
```
Note that you can analyze the tf tree by running `rosrun tf view_frames`. This will generate a file `frames.pdf` in the current working directory, which you can quickly view with `evince frames.pdf`.

3. Now we need to control the robot to drive around the space. We can do this using the built-in turtlebot keyboard control node, or start our own node to command random/test motions. These methods are described in prior sections of this document.

4. At any point while it's running, we can run the following command on the host PC to save the current map being built.
```
rosrun map_server map_saver -f mapname
```
This will save the map image as `mapname.pgm`, as well as some descriptive info to `mapname.yaml`. We can quickly view the map image with the command `feh mapname.pgm`.

<p align="center">
  <img src="images/carto_first_map.png"/>
  
  <i>The first map we've obtained from cartographer running on the locobot while it sits on the lab floor. (When we setup an environment to gather proper training data in the future, we will collect a much more meaningful map.)</i>
</p>

### Using Rosbags
At any point while the robot is running, we can collect a rosbag to use later. If you're getting a bag to use for offline mapping, you probably just need to start bringup and the node that controls the robot to move around. Then begin collecting a bag with
```
rosbag record -a --output-name=~/data/test_bag.bag
```
If you've run this on the robot, you can then copy it over to the host PC by running (on the host):
```
rsync -rvz locobot_llpr:~/data/test_bag.bag ~/data/
```

To replay this bag later, use `rosbag play ~/data/test_bag.bag`.

**However**, usually time synchronization is an issue, since the timestamps in the bag will be in the past. To avoid problems, we can globally set ROS to use the bag time instead of current time. 
To do this, it will be helpful to start `roscore` explicitly in its own terminal, then run the following in a new terminal before starting any other ROS processes.
```
rosparam set use_sim_time true
```
This will globally set ROS to use the sim time, so you only need to run it once; however, you **MUST** run this before starting anything else, such as static transform publishers. If you don't, you will get warnings such as "message removed becuase it is too old".

Then, when replaying the bag, append the `--clock` argument, i.e.,
```
rosbag play ~/data/test_bag.bag --clock
```

Cartographer has a utility to validate rosbags and tell you if it has any warnings or errors from the bag contents.
```
cd ~/cartographer_ws
source install_isolated/setup.bash
cartographer_rosbag_validate -bag_filename ~/data/test_bag.bag
```
If there are no errors, you can then run Cartographer from the bag:
```
roslaunch cartographer_ros my_robot.launch bag_filename:=~/data/test_bag.bag
```

When attempting to run cartographer from rosbags I gathered, it seems our IMU data is very messy; messages arrive out of order, and they seem to contain no acceleration information. Cartographer also does not like that both the `/locobot/mobile_base/sensors/imu_data` and `/locobot/mobile_base/sensors/imu_data_raw` claim the same tf frame. Due to these issues, I disabled the IMU being used in the configuration file `my_robot.lua`.

Note that you can analyze a rosbag in detail with the utility `rqt_bag`. Simply run `rqt_bag` in the terminal and select your bag from the "File" menu.


---

# Running this project
## Setup
Clone and setup this repository on your host PC.
```
cd ~/dev
git clone git@github.com:kevin-robb/coarse-map-nav-integration.git
cd coarse-map-integration
git submodule update --init
```
You should also setup the conda environment to install all needed python dependencies.
```
conda env create -f environment.yml
conda activate cmn_env
```

## Non-ROS runner
The only ROS dependency is the communication to the physical robot. So, I have provided a non-ROS runner for the project running in simulation.
```
cd ~/dev/coarse-map-integration/cmn_ws
python3 src/cmn_pkg/src/non_ros_runner.py -m <run_mode>
```
where `<run_mode>` is one of `discrete`, `discrete_random`, and `continuous`.

TODO sim image.

<!-- 
<p align="center">
  <img src="images/discrete-on-robot.png"/>
  
  <i>The CMN viz when running live on the physical robot.</i>
</p> -->

## Testing deep model predictions
There is also a non-ROS runner to test the model predictions on saved panoramic RGB images.
```
cd ~/dev/coarse-map-integration/cmn_ws
python3 src/cmn_pkg/src/scripts/cmn/run_model_on_saved_pano.py \
  -m src/cmn_pkg/src/scripts/cmn/model/trained_local_occupancy_predictor_model.pt \
  -d src/cmn_pkg/data
```

<p align="center">
  <img src="images/chair-hallway-prediction-1.png"/>
  
  <i>Four RGB images making up a panorama (front, right, back, left) are read from file and run through the saved model to produce the predicted local occupancy shown. In this case, the robot is between two rows of desks with chairs, which appear in the prediction.</i>
</p>

## ROS runner
Start bringup on the robot as described above, and set your host PC to use the locobot as ROS master. Then build and source our ROS workspace.
```
cd ~/dev/coarse-map-integration/cmn_ws
catkin_make
source devel/setup.bash
```

If using the depth pointcloud to generate local occupancy measurements, you must start the depth proc:
```
roslaunch cmn_pkg start_depth_proc.launch
```

To just evaluate local occupancy generation from LiDAR/depth data, use the locobot interface node:
```
roslaunch cmn_pkg test_lidar.launch
```

Now you can start CMN with one of the launch files in the `cmn_pkg`. These have parameters you can change as well. The primary one for running the project on the locobot is:
```
roslaunch cmn_pkg discrete_with_viz.launch
```

<p align="center">
  <img src="images/discrete-on-robot.png"/>
  
  <i>The CMN viz when running live on the physical robot.</i>
</p>
