### Aliases for the locobot
# Copy this file to locobot:
# rsync -rvz locobot_bash_aliases.sh locobot_llpr:~/.bash_aliases

# Start bringup/everything - starts sensors publishing and motors subscribing.
alias start_all="roslaunch interbotix_xslocobot_control xslocobot_control.launch robot_model:=locobot_wx200 use_base:=true use_camera:=true use_lidar:=true"
alias start_all_except_lidar="roslaunch interbotix_xslocobot_control xslocobot_control.launch robot_model:=locobot_wx200 use_base:=true use_camera:=true use_lidar:=false"

# Command a simple motion for test. Requires bringup is running already.
alias test_motion="rostopic pub --once /locobot/mobile_base/commands/velocity geometry_msgs/Twist '{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}'"

# Record a bag with the standard name.
alias record_bag="rosbag record --output-name=~/data/test_bag.bag -a"

# Save and view the TF tree.
view_tf_frames() {
    rosrun tf view_frames
    evince frames.pdf
}

# Setup static transforms for cartographer to work.
setup_static_tf() {
    rosrun tf static_transform_publisher 0 0 0 0 0 0 base_link camera_depth_frame 100 &
    rosrun tf static_transform_publisher 0 0 0 0 0 0 base_link target_frame 100 &
    rosrun tf static_transform_publisher 0 0 0 0 0 0 locobot/laser_frame_link laser 100 &
    rosrun tf static_transform_publisher 0 0 0 0 0 0 locobot/laser_frame_link locobot/base_link 100 &
}
# Kill the static transform publishers in the background.
alias kill_static_tf_pubs="pkill static_transform_publisher"
# Alternatively, search for them with 
# ps -eaf | grep static_transform_publisher
# or
# pgrep static_transform_publisher
# then for each process ID, run
# kill <process_id>

# Start cartographer.
start_cartographer() {
    cd ~/cartographer_ws
    source install_isolated/setup.bash
    roslaunch cartographer_ros my_robot.launch
}

# Save and view a map being created by cartographer.
save_map() {
    rosrun map_server map_saver -f mapname
    feh mapname.pgm
}
