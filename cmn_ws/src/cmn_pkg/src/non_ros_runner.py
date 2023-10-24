#!/usr/bin/env python3

"""
Interface to run the non-ROS part of the project, since repeatedly starting and killing ros is slow.
"""

import runner_node
import argparse
from time import sleep


def main():
    parser = argparse.ArgumentParser(description="Run the CMN code without ROS.")
    parser.add_argument('-m', action="store", dest="run_mode", type=str, required=True, help="Run mode to use. Options: {:}".format(runner_node.g_run_modes))
    args = parser.parse_args()

    runner_node.read_params()
    
    # The sim & viz are always enabled when using this runner.
    runner_node.set_global_params(args.run_mode, True, True, None)

    # Set the dt to 0 so we can play the sim one frame at a time.
    runner_node.g_dt = 0.0

    # Call the main run loop at the same frequency it would be called through ROS.
    while True:
        runner_node.timer_update_loop()
        sleep(runner_node.g_dt)


if __name__ == "__main__":
    main()
