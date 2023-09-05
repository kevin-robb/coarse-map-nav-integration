#!/usr/bin/env python3

"""
Node to allow the user to manually send discrete commands to the robot.
Can be used to control its motion during creation of the 'ground truth' map.
"""

import rospy
from std_msgs.msg import String
import rospkg, yaml

#### GLOBAL VARIABLES ####
discr_cmd_pub = None
##########################


def read_params():
    """
    Read configuration params from the yaml.
    """
    # Determine filepath.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('cmn_pkg')
    # Open the yaml and get the relevant params.
    with open(pkg_path+'/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        global g_dt
        g_dt = config["dt"]


def run_loop(event):
    """
    Wait for a command line input from the user, then send it as a discrete motion command to the robot.
    """
    pass


def main():
    rospy.init_node('runner_node')

    read_params()

    # TODO get params from launch file args, same as runner node.

    # Publish discrete commands as strings that will be interpreted by the runner.
    global discr_cmd_pub
    discr_cmd_pub = rospy.Publisher("/cmn/commands/discrete", String, queue_size=10)

    rospy.Timer(rospy.Duration(g_dt), run_loop)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass