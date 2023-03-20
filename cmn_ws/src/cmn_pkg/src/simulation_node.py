#!/usr/bin/env python3

"""
Node for extremely basic testing of localization node in best-case-scenario:
 - "coarse map" used by pf is exactly the same as the ground truth map.
 - observations will be ground truth.
 - motion commands will be followed exactly, with no noise.
"""

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray
import rospkg, yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backend_bases import MouseButton
import cv2
from cv_bridge import CvBridge
from math import remainder, tau, sin, cos, pi, ceil

from scripts.cmn_utilities import clamp, ObservationGenerator

############ GLOBAL VARIABLES ###################
# ROS stuff.
bridge = CvBridge()
# Instantiate class for utility functions.
obs_gen = ObservationGenerator()
# Ground-truth vehicle pose (x,y,yaw) in meters and radians, in the global map frame (origin at center).
veh_pose_true = np.array([0.0, 0.0, 0.0])
veh_pose_est = np.array([0.0, 0.0, 0.0]) # PF estimate.
particles_x, particles_y = None, None # Full set of particles in PF.
# Place to store plots so we can remove/update them some time later.
plots = {}
#################################################

########################### UTILITY/HELPER FUNCTIONS ###############################
def remove_plot(name):
    """
    Remove the plot if it already exists.
    """
    if name in plots.keys():
        try:
            # this works for stuff like scatter(), arrow()
            plots[name].remove()
        except:
            # the first way doesn't work for plt.plot()
            line = plots[name].pop(0)
            line.remove()
        # remove the key.
        del plots[name]

def on_click(event):
    # global clicked_points
    if event.button is MouseButton.LEFT:
        # kill the node.
        rospy.loginfo("Killing simulation_node because you clicked on the plot.")
        exit()
    elif event.button is MouseButton.RIGHT:
        # may want to use this for something eventually
        pass

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
        global g_debug_mode, g_dt
        g_debug_mode = config["test"]["run_debug_mode"]
        g_dt = config["perception_node_dt"]
        # Rostopics.
        global g_topic_commands, g_topic_localization, g_topic_observations, g_topic_occ_map
        g_topic_observations = config["topics"]["observations"]
        g_topic_occ_map = config["topics"]["occ_map"]
        g_topic_localization = config["topics"]["localization"]
        g_topic_commands = config["topics"]["commands"]
        # Particle filter params.
        global g_pf_num_particles
        g_pf_num_particles = config["particle_filter"]["num_particles"]
        # Constraints.
        global g_max_fwd_cmd, g_max_ang_cmd
        g_max_fwd_cmd = config["constraints"]["fwd"]
        g_max_ang_cmd = config["constraints"]["ang"]

################################ CALLBACKS #########################################
def get_command(msg:Vector3):
    """
    Receive a commanded motion, which will move the robot accordingly.
    """
    global veh_pose_true
    # TODO Perturb with some noise.
    # Clamp the commands at the allowed motion in a single timestep.
    fwd_dist = clamp(msg.x, 0, g_max_fwd_cmd) # meters forward
    dtheta = clamp(msg.z, -g_max_ang_cmd, g_max_ang_cmd) # radians CCW
    veh_pose_true[0] += fwd_dist * cos(veh_pose_true[2])
    veh_pose_true[1] += fwd_dist * sin(veh_pose_true[2])
    # Clamp the vehicle pose to remain inside the map bounds.
    veh_pose_true[0] = clamp(veh_pose_true[0], obs_gen.map_x_min_meters, obs_gen.map_x_max_meters)
    veh_pose_true[1] = clamp(veh_pose_true[1], obs_gen.map_y_min_meters, obs_gen.map_y_max_meters)
    # Keep yaw normalized to (-pi, pi).
    veh_pose_true[2] = remainder(veh_pose_true[2] + dtheta, tau)

    print("Veh pose is now " + str(veh_pose_true))

    # Generate an observation for the new vehicle pose.
    generate_observation()


def get_occ_map(msg):
    """
    Get the processed occupancy grid map to use as the "ground truth" map.
    """
    global occ_map_true
    # Convert from ROS Image message to an OpenCV image.
    occ_map_true = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    # Set map in our utilities class.
    obs_gen.set_map(occ_map_true)

    # Add the full map to our visualization (after it's processed).
    ax0.imshow(obs_gen.map, cmap="gray", vmin=0, vmax=1)

    """
    NOTE this architecture forms a cycle, observation -> localization -> command, so to complete it we will generate a new observation upon receiving a command.
    To kick-start this cycle, we will wait until the map has been processed, and then assume we've just received a zero command.
    """
    generate_observation()


def get_localization_est(msg):
    """
    Get localization estimate from the particle filter, and save it to be displayed on the viz.
    """
    global veh_pose_est
    # Yaw is always estimated globally in radians.
    veh_pose_est[0] = msg.x
    veh_pose_est[1] = msg.y
    veh_pose_est[2] = msg.z

def get_particle_set(msg):
    """
    Get the full set of particle positions from the current iteration.
    @param msg, Float32MultiArray message whose data is an array containing [x1,..,xn,y1,..,yn].
    """
    global particles_x, particles_y
    particles_x = msg.data[:g_pf_num_particles]
    particles_y = msg.data[g_pf_num_particles:]


############################ SIMULATOR FUNCTIONS ####################################
def generate_observation():
    """
    Use the map and known ground-truth robot pose to generate the best possible observation.
    """
    # Do not attempt to use the utilities class until the map has been processed.
    while not obs_gen.initialized:
        rospy.sleep(0.1)

    # Add the new (ground truth) vehicle pose to the viz.
    veh_row, veh_col = obs_gen.transform_map_m_to_px(veh_pose_true[0], veh_pose_true[1])
    remove_plot("veh_pose_true")
    plots["veh_pose_true"] = ax0.arrow(veh_col, veh_row, 0.5*cos(veh_pose_true[2]), -0.5*sin(veh_pose_true[2]), color="blue", width=1.0, label="True Vehicle Pose")

    # Add the most recent localization estimate to the viz.
    veh_row_est, veh_col_est = obs_gen.transform_map_m_to_px(veh_pose_est[0], veh_pose_est[1])
    remove_plot("veh_pose_est")
    plots["veh_pose_est"] = ax0.arrow(veh_col_est, veh_row_est, 0.5*cos(veh_pose_est[2]), -0.5*sin(veh_pose_est[2]), color="green", width=1.0, zorder = 3, label="PF Estimate")

    # Plot the set of particles in the PF.
    if particles_x is not None and particles_y is not None:
        remove_plot("particle_set")
        # Convert all particles from meters to pixels.
        particles_r, particles_c = [], []
        for i in range(g_pf_num_particles):
            r, c = obs_gen.transform_map_m_to_px(particles_x[i], particles_y[i])
            particles_r.append(r)
            particles_c.append(c)
        plots["particle_set"] = ax0.scatter(particles_c, particles_r, s=10, color="red", zorder=1, label="All Particles")

    # Use utilities class to generate the observation.
    obs_img, rect = obs_gen.extract_observation_region(veh_pose_true)

    # Plot the bounding box on the base map.
    box = cv2.boxPoints(rect)
    box_x_coords = [box[i,0] for i in range(box.shape[0])] + [box[0,0]]
    box_y_coords = [box[i,1] for i in range(box.shape[0])] + [box[0,1]]
    remove_plot("obs_bounding_box")
    plots["obs_bounding_box"] = ax0.plot(box_x_coords, box_y_coords, "r-", zorder=2, label="Observed Area")

    # Plot the new observation.
    remove_plot("obs_img")
    plots["obs_img"] = ax1.imshow(obs_img, cmap="gray", vmin=0, vmax=1)
    # Add vehicle pose relative to observation region for clarity.
    # NOTE since it's plotted sideways, robot pose is on the left side.
    if "veh_pose_obs" not in plots.keys():
        plots["veh_pose_obs"] = ax1.arrow(obs_gen.veh_px_vert_from_bottom_on_obs-0.5, obs_gen.obs_width_px // 2 + obs_gen.veh_px_horz_from_center_on_obs, 0.5*obs_gen.obs_resolution, 0.0, color="blue", width=0.01/obs_gen.obs_resolution, zorder = 2)

    ax0.legend(loc="upper left")
    plt.draw()
    plt.pause(g_dt)

    # Publish this observation for the localization node to use.
    observation_pub.publish(bridge.cv2_to_imgmsg(obs_img, encoding="passthrough"))


def main():
    global observation_pub
    rospy.init_node('simulation_node')

    read_params()

    # make live plot bigger.
    plt.rcParams["figure.figsize"] = (9,9)

    # Subscribe to occupancy grid map to use as ground-truth.
    rospy.Subscriber(g_topic_occ_map, Image, get_occ_map, queue_size=1)
    # Subscribe to commanded motion.
    rospy.Subscriber(g_topic_commands, Vector3, get_command, queue_size=1)

    # Subscribe to localization est (for viz only).
    rospy.Subscriber(g_topic_localization, Vector3, get_localization_est, queue_size=1)
    rospy.Subscriber(g_topic_localization + "/set", Float32MultiArray, get_particle_set, queue_size=1)

    # Publish ground-truth observation
    observation_pub = rospy.Publisher(g_topic_observations, Image, queue_size=1)

    # startup the plot.
    global fig, ax0, ax1
    fig = plt.figure(figsize=(8, 6)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    plt.title("Ground Truth Map & Vehicle Pose")
    plt.axis("off")
    ax1 = plt.subplot(gs[1])
    plt.title("Observation")
    # set constant plot params.
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    # allow clicking on the plot to do things (like kill the node).
    plt.connect('button_press_event', on_click)
    plt.show()

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass