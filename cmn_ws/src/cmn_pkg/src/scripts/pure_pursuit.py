#!/usr/bin/env python3

"""
Set of static functions to perform pure pursuit navigation.
"""

import rospy
from math import remainder, tau, pi, atan2, sqrt
from time import time

class PurePursuit:
    # Pure pursuit params.
    lookahead_dist_init = 0.2 # meters.
    lookahead_dist_max = 2 # meters.
    k_p = 0.5 # proportional gain.
    k_i = 0.0 # integral gain.
    k_d = 0.0 # derivative gain.
    k_fwd_lin = 0.02 # mult. gain on fwd vel.
    k_fwd_power = 12 # exponential term in fwd vel calculation.
    k_fwd_add = 0.01 # additive term in fwd vel calculation.
    # Path to follow.
    path_meters = []
    # PID vars.
    integ = 0 # accumulating integral term.
    err_prev = 0.0 # error from last iteration used for derivative term.
    last_time = 0.0 # time from last iteration used for computing dt.

    @staticmethod
    def compute_command(cur):
        """
        Determine odom command to stay on the path.
        @param cur, 3x1 numpy array of vehicle pose in meters (x,y,yaw).
        """
        # pare the path up to current veh pos.
        PurePursuit.pare_path(cur)

        if len(PurePursuit.path_meters) < 1: 
            # if there's no path yet, just wait. (send 0 cmd)
            rospy.logwarn("PP: Pure pursuit called with no path. Commanding zeros.")
            return 0.0, 0.0

        # define lookahead point.
        lookahead_pt = None
        lookahead_dist = PurePursuit.lookahead_dist_init # starting search radius.
        # look until we find the path, or give up at the maximum dist.
        while lookahead_pt is None and lookahead_dist <= PurePursuit.lookahead_dist_max: 
            lookahead_pt = PurePursuit.choose_lookahead_pt(cur, lookahead_dist)
            lookahead_dist *= 1.25
        # make sure we actually found the path.
        if lookahead_pt is None:
            # we can't see the path, so just try to go to the first pt.
            lookahead_pt = PurePursuit.path_meters[0]
        
        # TODO get this working right. for now, always use goal as lookahead point.
        lookahead_pt = PurePursuit.path_meters[-1]
        
        rospy.logwarn("PP: Choosing lookahead point ({:}, {:}).".format(lookahead_pt[0], lookahead_pt[1]))
        # compute global heading to lookahead_pt
        gb = atan2(lookahead_pt[1] - cur[1], lookahead_pt[0] - cur[0])
        # compute hdg relative to veh pose.
        beta = remainder(gb - cur[2], tau)
        rospy.logwarn("PP: Angle difference is {:.2f}, or {:.2f} relative to current vehicle pose.".format(gb, beta))

        # compute time since last iteration.
        dt = 0
        if PurePursuit.last_time != 0:
            dt = time() - PurePursuit.last_time
        PurePursuit.last_time = time()
            
        # Update PID terms.
        P = PurePursuit.k_p * beta # proportional to hdg error.
        # Update global integral term.
        PurePursuit.integ += beta * dt
        I = PurePursuit.k_i * PurePursuit.integ # integral to correct systematic error.
        D = 0.0 # slope to reduce oscillation.
        if dt != 0:
            D = PurePursuit.k_d * (beta - PurePursuit.err_prev) / dt
        # Save err for next iteration.
        PurePursuit.err_prev = beta
        ang = P + I + D
        # Compute forward velocity control command using hdg error beta.
        fwd = PurePursuit.k_fwd_lin * (1 - abs(beta / pi))**PurePursuit.k_fwd_power + PurePursuit.k_fwd_add
        
        return fwd, ang


    @staticmethod
    def pare_path(cur):
        """
        If the vehicle is near a path pt, cut the path off up to this pt.
        @param cur, 3x1 numpy array of vehicle pose in meters (x,y,yaw).
        """
        for i in range(len(PurePursuit.path_meters)):
            r = ((cur[0]-PurePursuit.path_meters[i][0])**2 + (cur[1]-PurePursuit.path_meters[i][1])**2)**(1/2)
            if r < 0.15:
                # remove whole path up to this pt.
                del PurePursuit.path_meters[0:i+1]
                return


    @staticmethod
    def choose_lookahead_pt(cur, lookahead_dist):
        """
        Find the point on the path at the specified radius from the current veh pos.
        @param cur, 3x1 numpy array of vehicle pose in meters (x,y,yaw).
        @param lookahead_dist, float, search radius to use to find a goal point to aim for when navigating.
        """
        # if there's only one path point, go straight to it.
        if len(PurePursuit.path_meters) == 1:
            return PurePursuit.path_meters[0]
        lookahead_pt = None
        # check the line segments between each pair of path points.
        for i in range(1, len(PurePursuit.path_meters)):
            # get vector between path pts.
            diff = [PurePursuit.path_meters[i][0]-PurePursuit.path_meters[i-1][0], PurePursuit.path_meters[i][1]-PurePursuit.path_meters[i-1][1]]
            # get vector from veh to first path pt.
            v1 = [PurePursuit.path_meters[i-1][0]-cur[0], PurePursuit.path_meters[i-1][1]-cur[1]]
            # compute coefficients for quadratic eqn to solve.
            a = diff[0]**2 + diff[1]**2
            b = 2*(v1[0]*diff[0] + v1[1]*diff[1])
            c = v1[0]**2 + v1[1]**2 - lookahead_dist**2
            try:
                discr = sqrt(b**2 - 4*a*c)
            except:
                # discriminant is negative, so there are no real roots.
                # (line segment is too far away)
                continue
            # compute solutions to the quadratic.
            # these will tell us what point along the 'diff' line segment is a solution.
            q = [(-b-discr)/(2*a), (-b+discr)/(2*a)]
            # check validity of solutions.
            valid = [q[i] >= 0 and q[i] <= 1 for i in range(2)]
            # compute the intersection pt. it's the first seg pt + q percent along diff vector.
            if valid[0]: lookahead_pt = [PurePursuit.path_meters[i-1][0]+q[0]*diff[0], PurePursuit.path_meters[i-1][1]+q[0]*diff[1]]
            elif valid[1]: lookahead_pt = [PurePursuit.path_meters[i-1][0]+q[1]*diff[0], PurePursuit.path_meters[i-1][1]+q[1]*diff[1]]
            else: continue # no intersection pt in the allowable range.
        return lookahead_pt

