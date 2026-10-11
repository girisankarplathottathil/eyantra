#!/usr/bin/env python3


"""
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1C of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
"""

# Team ID:          eYRC#4101
# Author List:		Joyal George,Adithyan A,Anandhu Saji,P H Girisankar
# Filename:		    path_follower.py
# Functions:
# 			        wrap, yaw_from_quat, clamp, choose_heading, main
# Nodes:
# 			        Publishing Topics  - [ /cmd_vel ]
#                   Subscribing Topics - [ /ebot_path, /odom, /scan, /tf ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
import numpy as np
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

##################### TASK CONSTANTS #######################

# The route is published here ONCE, before your node starts, and held for late
# subscribers. An ordinary subscription waits forever for a message already sent.
path_topic = "/ebot_path"

odom_topic = "/odom"
scan_topic = "/scan"
cmd_topic = "/cmd_vel"

# What /cmd_vel accepts. The base clips anything beyond these, and clipping one of the
# two changes their ratio - which is the arc the base actually drives.
max_linear_mps = 0.5
max_angular_rps = 1.0

# ---- Tuning constants (change these while testing) ----

WP_TOL = 0.12  # m, waypoint counts as reached (the 5-mark bonus needs EVERY waypoint within 0.3 m)
WP_PASS_TOL = 0.22  # m, fallback: if we got this close and are now moving away, the waypoint is done
V_CRUISE = 0.35  # m/s, normal forward speed (limit is 0.5)
KP_ANG = 1.5  # P gain on heading error
TURN_GATE = 0.7  # rad, above this heading error the base turns on the spot

# Robot footprint used for obstacle checks. CHECK THESE AGAINST THE eBot URDF: the widest
# part of the robot is not its chassis (wheels / mounted parts may stick out further).
ROBOT_HALF_WIDTH = 0.30  # m, half of the widest part of the robot
SAFETY_MARGIN = 0.08  # m, extra clearance on each side
BODY_FRONT = 0.30  # m, distance from base centre to the front of the robot

LOOKAHEAD = 1.6  # m, how far ahead the corridor check looks
CAND_LIMIT_DEG = 70  # candidate headings searched within +/- this angle
CAND_STEP_DEG = 5  # step between candidate headings
HYSTERESIS = 0.3  # penalty on changing the chosen heading (stops dithering)
MAX_SCAN_RANGE = 4.0  # m, lidar points further than this are ignored


##################### HELPER FUNCTIONS #######################


def wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(q):
    """Yaw (rotation about z) of a geometry_msgs Quaternion."""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def choose_heading(pts, bearing, reach, prev):
    """
    Description:    Pick the direction to drive in. Candidate headings (relative to the base,
                    0 = straight ahead) are each tested with a corridor as wide as the robot.
                    Among the candidates whose corridor is clear for 'reach' metres, the one
                    closest to the waypoint bearing wins.

    Args:
        pts (ndarray Nx2):  lidar points in the base frame (x forward, y left)
        bearing (float):    direction of the waypoint relative to the base (rad)
        reach (float):      how far the corridor must be clear (m)
        prev (float):       heading chosen on the previous cycle (rad)

    Returns:
        (theta, free, clear):  chosen heading (rad), clear distance along it (m),
                               and whether it was fully clear
    """
    lim = math.radians(CAND_LIMIT_DEG)
    cands = np.radians(
        np.arange(-CAND_LIMIT_DEG, CAND_LIMIT_DEG + 1, CAND_STEP_DEG, dtype=float)
    )
    cands = np.unique(np.append(cands, clamp(bearing, -lim, lim)))

    half_w = ROBOT_HALF_WIDTH + SAFETY_MARGIN

    if pts.shape[0] == 0:
        free = np.full(cands.shape, reach)
    else:
        c = np.cos(cands)[:, None]
        s = np.sin(cands)[:, None]
        along = (
            pts[None, :, 0] * c + pts[None, :, 1] * s
        )  # distance along the candidate
        across = -pts[None, :, 0] * s + pts[None, :, 1] * c  # sideways offset from it
        blocked = (along > 0.0) & (along < reach) & (np.abs(across) < half_w)
        free = np.where(blocked, along, reach).min(axis=1)

    d_goal = cands - bearing
    cost = np.abs(np.arctan2(np.sin(d_goal), np.cos(d_goal))) + HYSTERESIS * np.abs(
        cands - prev
    )

    ok = free >= reach - 1e-6
    if ok.any():
        idx = int(np.argmin(np.where(ok, cost, np.inf)))
        return float(cands[idx]), float(free[idx]), True

    # Nothing is fully clear: head for the most open direction (ties go to the goal side).
    idx = int(np.argmax(free - 0.05 * cost))
    return float(cands[idx]), float(free[idx]), False


##################### CLASS DEFINITION #######################


class ebot_nav(Node):
    """
    ___CLASS___

    Description:    Class which serves the purpose to drive the eBot along the route
                    published on /ebot_path, through its waypoints, in order.
    """

    def __init__(self):
        """
        Description:    Initialization of class ebot_nav
        """

        # use_sim_time is set here, not on the command line, so this node runs on the
        # simulation clock however it is started.
        super().__init__(  # registering node
            "ebot_nav_node",
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    "use_sim_time", rclpy.Parameter.Type.BOOL, True
                )
            ],
        )

        ############ Topic SUBSCRIPTIONS ############

        # The route is published once and latched, so the subscription must ask for the
        # same durability (TRANSIENT_LOCAL) or it never receives it.
        path_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.path_sub = self.create_subscription(
            Path, path_topic, self.pathcb, path_qos
        )
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scancb, qos_profile_sensor_data
        )

        ############ Topic PUBLISHERS ############

        self.cmd_pub = self.create_publisher(
            Twist, cmd_topic, 10
        )  # the eBot drives on what you publish here

        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05  # rate of time to run one control cycle (seconds)
        self.timer = self.create_timer(
            control_rate, self.process_navigation
        )  # creating a timer based function which gets called on every 0.05 seconds (as defined by 'control_rate' variable)

        self.route = None  # the route to drive (from pathcb())
        self.odom = None  # where the base is (from odomcb())
        self.scan = None  # what the lidar sees (from scancb())

        ############ ADD YOUR CODE HERE ############

        self.wp_index = 0  # index of the waypoint being driven to
        self.done = False  # True once the last waypoint is reached
        self.prev_theta = 0.0  # heading chosen last cycle (for hysteresis)
        self.wp_min = float("inf")  # closest distance reached to the current waypoint
        self.worst_pass = 0.0  # worst closest-approach over all waypoints so far

        # Version banner: if you do not see this line when the node starts, ros2 run is
        # still running an OLD build. Rebuild with colcon and source the workspace again.
        self.get_logger().info(
            f"path_follower v3 started (WP_TOL={WP_TOL}, WP_PASS_TOL={WP_PASS_TOL})"
        )

        self.base_frame = "base_link"  # replaced by odom's child_frame_id once known
        self.lidar_tf = None  # (x, y, yaw) of the lidar in the base frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        ############################################

    def pathcb(self, data):
        """
        Description:    Callback function for the route topic.
                        Use this function to receive the waypoints the eBot has to drive.

        Args:
            data (Path):    The route, as a sequence of poses

        Returns:
        """

        ############ ADD YOUR CODE HERE ############

        # Take the route once. Replacing it mid-run would reset the progress.
        if self.route is not None:
            return

        self.route = [(p.pose.position.x, p.pose.position.y) for p in data.poses]
        self.get_logger().info(
            f'Route received: {len(self.route)} waypoints in frame "{data.header.frame_id}"'
        )

        ############################################

    def odomcb(self, data):
        """
        Description:    Callback function for the odometry topic.
                        Use this function to receive where the base currently is.

        Args:
            data (Odometry):    Pose and velocity of the base

        Returns:
        """

        ############ ADD YOUR CODE HERE ############

        p = data.pose.pose.position
        self.odom = (p.x, p.y, yaw_from_quat(data.pose.pose.orientation))

        if data.child_frame_id:
            self.base_frame = data.child_frame_id

        ############################################

    def scancb(self, data):
        """
        Description:    Callback function for the lidar topic.
                        Use this function to receive what the lidar currently sees.

        Args:
            data (LaserScan):    One lidar sweep

        Returns:
        """

        ############ ADD YOUR CODE HERE ############

        # Where the lidar is mounted, read from /tf (static, so looked up until it succeeds once).
        if self.lidar_tf is None:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.base_frame, data.header.frame_id, Time()
                )
                self.lidar_tf = (
                    t.transform.translation.x,
                    t.transform.translation.y,
                    yaw_from_quat(t.transform.rotation),
                )
                self.get_logger().info(
                    f'Lidar frame "{data.header.frame_id}" in "{self.base_frame}": '
                    f"x={self.lidar_tf[0]:.3f} y={self.lidar_tf[1]:.3f} yaw={self.lidar_tf[2]:.3f}"
                )
            except Exception as e:
                self.get_logger().warn(
                    f"Lidar transform not available yet, using no offset: {e}",
                    throttle_duration_sec=2.0,
                )

        tx, ty, psi = self.lidar_tf if self.lidar_tf is not None else (0.0, 0.0, 0.0)

        ranges = np.asarray(data.ranges, dtype=float)
        angles = data.angle_min + np.arange(ranges.size) * data.angle_increment

        # Keep only real measurements (inf / nan / out of range are not obstacles).
        valid = (
            np.isfinite(ranges)
            & (ranges >= data.range_min)
            & (ranges <= data.range_max)
            & (ranges <= MAX_SCAN_RANGE)
        )
        r = ranges[valid]
        a = angles[valid] + psi

        # Points in the base frame (x forward, y left), measured from the base centre.
        px = tx + r * np.cos(a)
        py = ty + r * np.sin(a)
        self.scan = np.column_stack((px, py)) if r.size else np.empty((0, 2))

        ############################################

    def stop(self):
        """Publish a zero velocity command."""
        self.cmd_pub.publish(Twist())

    def process_navigation(self):
        """
        Description:    Timer function used to drive the eBot along the route.

        Args:
        Returns:
        """

        ############ ADD YOUR CODE HERE ############

        try:
            self._navigate()
        except Exception as e:  # never let the node die with the base coasting
            self.get_logger().error(f"Navigation error: {e}", throttle_duration_sec=1.0)
            self.stop()

        ############################################

    def _navigate(self):
        """One control cycle."""

        if self.done:
            self.stop()
            return

        # Wait until the route, odometry and scan have all arrived.
        if self.route is None or self.odom is None or self.scan is None:
            return
        if len(self.route) == 0:
            self.stop()
            return

        x, y, yaw = self.odom
        last = len(self.route) - 1

        # Advance through the waypoints strictly in order.
        while True:
            tx, ty = self.route[self.wp_index]
            dist = math.hypot(tx - x, ty - y)
            self.wp_min = min(self.wp_min, dist)

            # Reached when truly close, or when we came within WP_PASS_TOL and are now
            # moving away (closest approach already happened, so it still counts).
            reached = dist < WP_TOL or (
                self.wp_min < WP_PASS_TOL and dist > self.wp_min + 0.05
            )
            if not reached:
                break
            self.worst_pass = max(self.worst_pass, self.wp_min)
            self.get_logger().info(
                f"Waypoint {self.wp_index + 1}/{last + 1} reached (closest {self.wp_min:.3f} m)"
            )
            if self.wp_index == last:
                self.done = True
                self.stop()
                self.get_logger().info(
                    f"Route complete, stopping. Worst closest-approach: {self.worst_pass:.3f} m "
                    f"(bonus needs <= 0.3 m)"
                )
                return
            self.wp_index += 1
            self.prev_theta = 0.0
            self.wp_min = float("inf")

        # Direction of the waypoint relative to the base.
        bearing = wrap(math.atan2(ty - y, tx - x) - yaw)

        # Pick a direction that is clear of rocks and closest to the waypoint.
        reach = max(min(LOOKAHEAD, dist), BODY_FRONT + 0.2)
        theta, free, clear = choose_heading(self.scan, bearing, reach, self.prev_theta)
        self.prev_theta = theta

        # Steer towards the chosen direction; slow down for large errors and close rocks.
        err = theta
        v = V_CRUISE * max(0.0, 1.0 - abs(err) / TURN_GATE)
        if not clear:
            v *= clamp((free - BODY_FRONT - 0.1) / 0.6, 0.0, 1.0)
        if self.wp_index == last:
            v = min(v, max(0.12, 0.8 * dist))  # ease onto the final waypoint
        w = KP_ANG * err

        # Keep inside the limits without changing the v/w ratio.
        s = max(1.0, abs(v) / max_linear_mps, abs(w) / max_angular_rps)
        v /= s
        w /= s

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

        self.get_logger().debug(
            f"wp={self.wp_index + 1} d={dist:.2f} bearing={bearing:.2f} theta={theta:.2f} "
            f"free={free:.2f} clear={clear} v={v:.2f} w={w:.2f}"
        )


##################### FUNCTION DEFINITION #######################


def main():
    """
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_nav class to perform its task
    """

    rclpy.init(args=sys.argv)  # initialisation

    node = rclpy.create_node("ebot_nav_process")  # creating ROS node

    node.get_logger().info(
        "Node created: eBot navigation process"
    )  # logging information

    ebot_nav_class = ebot_nav()  # creating a new object for class 'ebot_nav'

    try:
        rclpy.spin(
            ebot_nav_class
        )  # spining on the object to make it alive in ROS 2 DDS
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ebot_nav_class.stop()  # leave the base stationary
        except Exception:
            pass
        ebot_nav_class.destroy_node()  # destroy node after spin ends
        if rclpy.ok():
            rclpy.shutdown()  # shutdown process


if __name__ == "__main__":

    main()
