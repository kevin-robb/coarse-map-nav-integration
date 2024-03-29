#!/usr/bin/env python3

"""
Set of static functions to perform pure pursuit navigation.
"""

import rospy
from math import remainder, tau, pi, atan2, sqrt
from time import time

from scripts.basic_types import PoseMeters

class PurePursuit:
    verbose = False
    # Pure pursuit params.
    use_finite_lookahead_dist = True # if false, rather than computing a lookahead point, just use the goal point.
    lookahead_dist_init = 0.2 # meters.
    lookahead_dist_max = 2 # meters.
    k_p = 1.0 # proportional gain.
    k_i = 0.0 # integral gain.
    k_d = 0.0 # derivative gain.
    k_fwd_lin = 1 # mult. gain on fwd vel.
    k_fwd_power = 5 # exponential term in fwd vel calculation.
    k_fwd_add = 0.0 # additive term in fwd vel calculation.
    # Path to follow.
    path_meters = []
    # PID vars.
    integ = 0 # accumulating integral term.
    err_prev = 0.0 # error from last iteration used for derivative term.
    last_time = 0.0 # time from last iteration used for computing dt.


    def compute_command(self, cur_pose_m:PoseMeters, path):
        """
        Determine odom command to stay on the path.
        @param cur_pose_m, PoseMeters of current vehicle pose in meters (x,y,yaw).
        @param path, List of PoseMeters making up the path to follow.
        @return tuple of fwd, ang velocities to command, in m/s, rad/s.
        """
        self.path_meters = path
        # pare the path up to current veh pos.
        self.pare_path(cur_pose_m)

        if len(self.path_meters) < 1: 
            # if there's no path yet, just wait. (send 0 cmd)
            rospy.logwarn("PP: Pure pursuit called with no path. Commanding zeros.")
            return 0.0, 0.0

        if self.use_finite_lookahead_dist:
            # Define lookahead point.
            lookahead_pt = None
            lookahead_dist = self.lookahead_dist_init # starting search radius.
            # Look until we find the path, or give up at the maximum dist.
            while lookahead_pt is None and lookahead_dist <= self.lookahead_dist_max: 
                lookahead_pt = self.choose_lookahead_pt(cur_pose_m, lookahead_dist)
                lookahead_dist *= 1.25
            # Make sure we actually found the path.
            if lookahead_pt is None:
                # We can't see the path, so just try to go to the first pt.
                lookahead_pt = self.path_meters[0]
        else:
            # Just use goal as lookahead point.
            lookahead_pt = self.path_meters[-1]
        
        if self.verbose:
            rospy.loginfo("PP: Choosing lookahead point ({:.2f}, {:.2f}).".format(lookahead_pt.x, lookahead_pt.y))
        # Compute global heading to lookahead_pt
        gb = atan2(lookahead_pt.y - cur_pose_m.y, lookahead_pt.x - cur_pose_m.x)
        # Compute hdg relative to veh pose.
        beta = remainder(gb - cur_pose_m.yaw, tau)
        if self.verbose:
            rospy.loginfo("PP: Angle difference is {:.2f}, or {:.2f} relative to current vehicle pose.".format(gb, beta))

        # Compute time since last iteration.
        dt = 0
        if self.last_time != 0:
            dt = time() - self.last_time
        self.last_time = time()
            
        # Update PID terms.
        P = self.k_p * beta # proportional to hdg error.
        # Update global integral term.
        self.integ += beta * dt
        I = self.k_i * self.integ # integral to correct systematic error.
        D = 0.0 # slope to reduce oscillation.
        if dt != 0:
            D = self.k_d * (beta - self.err_prev) / dt
        # Save err for next iteration.
        self.err_prev = beta
        ang = P + I + D
        # Compute forward velocity control command using hdg error beta.
        fwd = self.k_fwd_lin * (1 - abs(beta / pi))**self.k_fwd_power + self.k_fwd_add
        
        return fwd, ang


    def pare_path(self, cur_pose_m:PoseMeters):
        """
        If the vehicle is near a path pt, cut the path off up to this pt.
        @param cur_pose_m - PoseMeters of vehicle pose in meters (x,y,yaw).
        """
        for i in range(len(self.path_meters)):
            dist = ((cur_pose_m.x-self.path_meters[i].x)**2 + (cur_pose_m.y-self.path_meters[i].y)**2)**(1/2)
            if dist < 0.15:
                # Remove whole path up to this pt.
                del self.path_meters[0:i+1]
                return


    def choose_lookahead_pt(self, cur_pose_m:PoseMeters, lookahead_dist:float) -> PoseMeters:
        """
        Find the point on the path at the specified radius from the current veh pos.
        @param cur_pose_m - PoseMeters of vehicle pose in meters (x,y,yaw).
        @param lookahead_dist - search radius to use to find a goal point to aim for when navigating.
        @return The chosen lookahead point as a PoseMeters object.
        """
        # If there's only one path point, go straight to it.
        if len(self.path_meters) == 1:
            return self.path_meters[0]
        lookahead_pt = None
        # Check the line segments between each pair of path points.
        for i in range(1, len(self.path_meters)):
            # Get vector between path pts.
            diff = [self.path_meters[i].x-self.path_meters[i-1].x, self.path_meters[i].y-self.path_meters[i-1].y]
            # Get vector from veh to first path pt.
            v1 = [self.path_meters[i-1].x-cur_pose_m.x, self.path_meters[i-1].y-cur_pose_m.y]
            # Compute coefficients for quadratic eqn to solve.
            a = diff[0]**2 + diff[1]**2
            b = 2*(v1[0]*diff[0] + v1[1]*diff[1])
            c = v1[0]**2 + v1[1]**2 - lookahead_dist**2
            try:
                discr = sqrt(b**2 - 4*a*c)
            except:
                # Discriminant is negative, so there are no real roots (i.e., line segment is too far away).
                continue
            # Compute solutions to the quadratic. These will tell us what point along the 'diff' line segment is a solution.
            q = [(-b-discr)/(2*a), (-b+discr)/(2*a)]
            # Check validity of solutions.
            valid = [q[i] >= 0 and q[i] <= 1 for i in range(2)]
            # Compute the intersection pt. it's the first seg pt + q percent along diff vector.
            if valid[0]:
                lookahead_pt = [self.path_meters[i-1].x+q[0]*diff[0], self.path_meters[i-1].y+q[0]*diff[1]]
            elif valid[1]:
                lookahead_pt = [self.path_meters[i-1].x+q[1]*diff[0], self.path_meters[i-1].y+q[1]*diff[1]]
            else:
                # No intersection pt in the allowable range.
                continue
        return lookahead_pt

