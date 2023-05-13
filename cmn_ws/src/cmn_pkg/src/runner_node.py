#!/usr/bin/env python3

"""
Node to mimic the discrete state space and action space used in the original Habitat simulation.
This node will handle all necessary steps that would otherwise be done by the rest of the nodes together.
"""

import rospy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
import rospkg, yaml
from cv_bridge import CvBridge

from scripts.cmn_utilities import ObservationGenerator, CoarseMapProcessor
from scripts.motion_planner import DiscreteMotionPlanner
from scripts.particle_filter import ParticleFilter

############ GLOBAL VARIABLES ###################
bridge = CvBridge()
# Instances of utility classes defined in src/scripts folder.
map_proc = CoarseMapProcessor()
obs_gen = ObservationGenerator()
dmp = DiscreteMotionPlanner()
pf = ParticleFilter()
# Flag to publish everything for the visualization.
g_pub_for_viz = False
# RealSense measurements buffer.
most_recent_RS_meas = None
#################################################

def run_loop_discrete(event=None):
    """
    Main run loop for discrete case.
    Commands motions on the robot to collect a panoramic measurement,
    uses the model to get an observation,
    localizes with a discrete bayesian filter,
    and commands discrete actions.
    """
    # Get a panoramic measurement.
    pano_meas = get_pano_meas()

    # Get an observation from this measurement.
    if g_use_ground_truth_map_to_generate_observations:
        # TODO sim
        observation = None
    else:
        # TODO Pass this panoramic measurement through the model to obtain an observation.
        observation = get_observation_from_pano_meas(pano_meas)

    # Convert to ROS image and publish it for viz.
    # observation_pub.publish(bridge.cv2_to_imgmsg(observation, encoding="passthrough"))

    # TODO Use discrete bayesian filter to localize the robot.
    robot_pose_estimate = None
    # TODO Determine path from estimated pose to the goal, using the coarse map, and determine a discrete action to command.
    # TODO for now just command random discrete action.
    dmp.cmd_random_discrete_action()
    # Proceed to the next iteration, where another measurement will be taken.

def run_loop_continuous(event=None):
    """
    Main run loop for continuous case.
    Uses only the current RS measurement,
    generates an observation,
    localizes with a particle filter,
    and commands continuous velocities.
    """
    # Get an image from the RealSense.
    meas = pop_from_RS_buffer()
    
    # Get an observation from this measurement.
    if g_use_ground_truth_map_to_generate_observations:
        # TODO sim
        observation = None
    else:
        # TODO Pass this measurement through the model to obtain an observation.
        observation = None # i.e., get_observation_from_meas(meas)

    # Use the particle filter to get a localization estimate from this observation.
    # Update the particle filter.
    pf_estimate = pf.update_with_observation(observation)

    if g_pub_for_viz:
        # Convert pf estimate into a message and publish it (for viz).
        loc_est = Vector3(pf_estimate[0], pf_estimate[1], pf_estimate[2])
        localization_pub.publish(loc_est)

        # Publish the full particle set (for viz).
        pf_set_msg = Float32MultiArray()
        pf_set_msg.data = list(pf.particle_set[:,0]) + list(pf.particle_set[:,1])
        particle_set_pub.publish(pf_set_msg)

    # Run the PF resampling step.
    pf.resample()

    # Choose velocity commands for the robot based on the pose estimate.
    fwd, ang = dmp.plan_path_to_goal(pf_estimate)
    dmp.pub_velocity_cmd(fwd, ang)

    # Propagate all particles by the commanded motion.
    pf.propagate_particles(fwd * g_dt, ang * g_dt)


# TODO make intermediary control_node that receives our commanded motion and either passes it through to the robot or uses sensors to perform reactive obstacle avoidance


def get_observation_from_pano_meas(pano_meas):
    """
    Pass the most recent measurement into the ML model, and get back a resulting observation.
    Publish the observation to be used for localization.
    @param pano_meas - dictionary with key:value pairs direction:image.
    """
    global most_recent_measurement
    if most_recent_measurement is None:
        return
    # Pull most recent measurement, and clear it so we'll be able to tell next iteration if there's nothing new.
    meas = most_recent_measurement
    most_recent_measurement = None

    # TODO make way to interface with the ML model, which will be in a separate file.
    observation = None # i.e., ml_model(pano_meas)

    return observation

##################### UTILITY FUNCTIONS #######################
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
        global g_debug_mode, g_do_path_planning
        g_debug_mode = config["test"]["run_debug_mode"]
        # In motion test mode, only this node will run, so it will handle the timer.
        global g_dt, g_run_mode, g_use_ground_truth_map_to_generate_observations
        g_dt = config["dt"]
        g_run_mode = config["run_mode"]
        g_use_ground_truth_map_to_generate_observations = config["use_ground_truth_map_to_generate_observations"]
        # Rostopics.
        global g_topic_measurements, g_topic_commands, g_topic_localization, g_topic_occ_map, g_topic_raw_map, g_topic_planned_path, g_topic_goal, g_topic_discrete_acions
        g_topic_measurements = config["topics"]["measurements"]
        g_topic_occ_map = config["topics"]["occ_map"]
        g_topic_raw_map = config["topics"]["raw_map"]
        g_topic_localization = config["topics"]["localization"]
        g_topic_goal = config["topics"]["goal"]
        g_topic_commands = config["topics"]["commands"]
        g_topic_discrete_acions = config["topics"]["discrete_actions"]
        g_topic_planned_path = config["topics"]["planned_path"]

######################## CALLBACKS ########################
def get_pano_meas():
    """
    Get a panoramic measurement.
    Since the robot has only a forward-facing camera, we must pivot in-place four times.
    @return dictionary with key:value pairs of direction:image.
    """
    pano_meas = {}
    pano_meas["front"] = pop_from_RS_buffer()
    dmp.cmd_discrete_action("90_LEFT") # pivot in-place 90 deg CCW, and then stop.
    pano_meas["left"] = pop_from_RS_buffer()
    dmp.cmd_discrete_action("90_LEFT")
    pano_meas["back"] = pop_from_RS_buffer()
    dmp.cmd_discrete_action("90_LEFT")
    pano_meas["right"] = pop_from_RS_buffer()
    dmp.cmd_discrete_action("90_LEFT")
    # Vehicle should now be facing forwards again (its original direction).
    return pano_meas

def pop_from_RS_buffer():
    """
    Wait for a new RealSense measurement to be available, and return it.
    """
    global most_recent_RS_meas
    while most_recent_RS_meas is None:
        rospy.sleep(0.01)
    # Convert from ROS Image message to an OpenCV image.
    cv_img_meas = bridge.imgmsg_to_cv2(most_recent_RS_meas, desired_encoding='passthrough')
    # Ensure this same measurement will not be used again.
    most_recent_RS_meas = None
    return cv_img_meas

def get_RS_image(msg):
    """
    Get a measurement Image from the RealSense camera.
    Could be changed multiple times before we need a measurement, so this allows skipping measurements to prefer recency.
    """
    global most_recent_RS_meas
    most_recent_RS_meas = msg

def get_goal_pos(msg:Vector3):
    """
    Get goal position in pixels.
    For now, this is obtained from the user clicking on the map in the sim viz.
    """
    dmp.set_goal_point((int(msg.x), int(msg.y)))

def main():
    rospy.init_node('runner_node')

    read_params()

    # Subscribe to sensor images from RealSense.
    # TODO may want to check /locobot/camera/color/camera_info
    rospy.Subscriber(g_topic_measurements, Image, get_RS_image, queue_size=1)

    # Publish control commands (velocities in m/s and rad/s).
    cmd_vel_pub = rospy.Publisher(g_topic_commands, Twist, queue_size=1)
    dmp.set_vel_pub(cmd_vel_pub)
    discrete_action_pub = rospy.Publisher(g_topic_discrete_acions, String, queue_size=1)
    dmp.set_discrete_action_pub(discrete_action_pub)

    # Subscribe to goal position in pixels on the map. This is obtained by the user clicking on the viz
    rospy.Subscriber(g_topic_goal, Vector3, get_goal_pos, queue_size=1)

    # Publishers for viz.
    if g_pub_for_viz:
        global path_pub, localization_pub, particle_set_pub
        # Publish planned path to the goal.
        path_pub = rospy.Publisher(g_topic_planned_path, Float32MultiArray, queue_size=1)
        dmp.set_path_pub(path_pub)
        # Publish localization estimate.
        localization_pub = rospy.Publisher(g_topic_localization, Vector3, queue_size=1)
        # Publish the full particle set (for viz only).
        particle_set_pub = rospy.Publisher(g_topic_localization + "/set", Float32MultiArray, queue_size=1)

    # Publish the coarse map for other nodes to use.
    occ_map_pub = rospy.Publisher(g_topic_occ_map, Image, queue_size=1)
    raw_map_pub = rospy.Publisher(g_topic_raw_map, Image, queue_size=1)
    # Wait to make sure the other nodes are ready to receive the map.
    rospy.sleep(1)
    occ_map_pub.publish(map_proc.get_occ_map_msg())
    raw_map_pub.publish(map_proc.get_raw_map_msg())
    # Set the map for utility classes to use.
    obs_gen.set_map(map_proc.occ_map)
    dmp.set_map(map_proc.occ_map)
    pf.set_map(map_proc.occ_map)

    if g_run_mode == "discrete":
        rospy.Timer(rospy.Duration(g_dt), run_loop_discrete)
    elif g_run_mode == "continuous":
        rospy.Timer(rospy.Duration(g_dt), run_loop_continuous)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass