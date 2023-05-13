#!/usr/bin/env python3

"""
Node to mimic the discrete state space and action space used in the original Habitat simulation.
This node will handle all necessary steps that would otherwise be done by the rest of the nodes together.
"""

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import rospkg, yaml
from cv_bridge import CvBridge

from cmn_pkg.src.scripts.map_handler import CoarseMapProcessor, Simulator
from scripts.motion_planner import DiscreteMotionPlanner
from scripts.particle_filter import ParticleFilter
from scripts.visualizer import Visualizer

############ GLOBAL VARIABLES ###################
bridge = CvBridge()
# Instances of utility classes defined in src/scripts folder.
map_proc = CoarseMapProcessor()
sim = Simulator()
dmp = DiscreteMotionPlanner()
pf = ParticleFilter()
viz = Visualizer()
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
    fwd, ang = dmp.cmd_random_discrete_action()

    if g_use_ground_truth_map_to_generate_observations:
        # Propagate the true vehicle pose by this discrete action.
        sim.propagate_with_dist(fwd, ang)


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
    observation, rect = None, None
    if g_use_ground_truth_map_to_generate_observations:
        # Do not attempt to use the utilities class until the map has been processed.
        while not sim.initialized:
            rospy.sleep(0.1)
        observation, rect = sim.get_true_observation()
    else:
        # TODO Pass this measurement through the ML model to obtain an observation.
        observation = None # i.e., get_observation_from_meas(meas)

    # Use the particle filter to get a localization estimate from this observation.
    pf_estimate = pf.update_with_observation(observation)

    if viz.enabled:
        # Update data for the viz.
        viz.set_observation(observation, rect)
        # Convert meters to pixels using our map transform class.
        row, col = sim.transform_map_m_to_px(pf_estimate[0], pf_estimate[1])
        viz.set_estimated_veh_pose_px(row, col, pf_estimate[2])

        # Update ground-truth data if we're running the sim.
        if g_use_ground_truth_map_to_generate_observations:
            # Convert meters to pixels using our map transform class.
            row, col = sim.transform_map_m_to_px(sim.veh_pose_true[0], sim.veh_pose_true[1])
            viz.set_true_veh_pose_px(row, col, sim.veh_pose_true[2])

        # Convert particle set to pixels as well.
        particle_set_px = pf.particle_set
        for i in range(pf.num_particles):
            particle_set_px[i,0], particle_set_px[i,0] = sim.transform_map_m_to_px(pf.particle_set[i,0], pf.particle_set[i,0])
        viz.set_particle_set(particle_set_px)

    # Run the PF resampling step.
    pf.resample()

    # Check if a new goal cell has been set.
    if viz.goal_cell is not None:
        dmp.set_goal_point(viz.goal_cell)

    # Choose velocity commands for the robot based on the pose estimate.
    fwd, ang = dmp.plan_path_to_goal(pf_estimate)
    dmp.pub_velocity_cmd(fwd, ang)
    sim.propagate_with_vel(fwd, ang) # Apply to the ground truth vehicle pose.

    if viz.enabled:
        # Update data for the viz.
        viz.set_planned_path(dmp.path_px_reversed)

    # Propagate all particles by the commanded motion.
    pf.propagate_particles(fwd * g_dt, ang * g_dt)

    if viz.enabled:
        # Update the viz.
        viz.update()


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
        global g_dt, g_run_mode, g_use_ground_truth_map_to_generate_observations
        g_dt = config["dt"]
        g_run_mode = config["run_mode"]
        g_use_ground_truth_map_to_generate_observations = config["use_ground_truth_map_to_generate_observations"]
        # Rostopics.
        global g_topic_measurements, g_topic_commands
        g_topic_measurements = config["topics"]["measurements"]
        g_topic_commands = config["topics"]["commands"]

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

def main():
    rospy.init_node('runner_node')

    read_params()

    # Subscribe to sensor images from RealSense.
    # TODO may want to check /locobot/camera/color/camera_info
    rospy.Subscriber(g_topic_measurements, Image, get_RS_image, queue_size=1)

    # Publish control commands (velocities in m/s and rad/s).
    cmd_vel_pub = rospy.Publisher(g_topic_commands, Twist, queue_size=1)
    dmp.set_vel_pub(cmd_vel_pub)

    # Set the map for utility classes to use.
    sim.set_map(map_proc.occ_map)
    dmp.set_map(map_proc.occ_map)
    pf.set_map(map_proc.occ_map)
    # Set things in viz that depend on coord systems.
    viz.set_map(sim.map) # Use the map after sim's pre-processing.
    viz.set_veh_pose_in_obs_region(sim.veh_px_vert_from_bottom_on_obs, sim.veh_px_horz_from_center_on_obs, sim.obs_width_px, sim.obs_resolution, sim.map_downscale_ratio)

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