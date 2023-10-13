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
