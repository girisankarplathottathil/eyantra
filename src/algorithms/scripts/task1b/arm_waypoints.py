#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		       StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1B of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:		   task1b_boilerplate.py
# Functions:
#			       [ Comma separated list of functions in this file ]
# Nodes:		   Add your publishing and subscribing node
#                   Example:
#			       Publishing Topics  - [ /delta_twist_cmds, /delta_joint_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw, /joint_states, /etc... ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
from rclpy.node import Node
from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
from std_srvs.srv import Trigger                                # ADDED: unlock_protective_stop


##################### TASK CONSTANTS #######################

# Tool positions in base_link, in metres, in the order they must be reached. The tool
# stops at each one and holds it for at least two seconds. Copy the signs as they are:
# base_link is the UR7e's own frame, not the Gazebo world's.
waypoints = [
    (-0.4085, -0.5379, 0.1967),   # 1
    (-0.8000, -0.0005, 0.3967),   # 2
    (-0.7430,  0.5280, 0.1967),   # 3
    (-0.4097,  0.5280, 0.1967),   # 4
    (-0.0763,  0.5280, 0.1967),   # 5
]

# The two command interfaces. Only ONE is active at a time; messages to the other are
# accepted and ignored.
servo_ns = '/ur_arm_controller'
twist_controller = 'delta_twist_controller'
joint_controller = 'delta_joint_controller'

# The only frame a twist may be stamped with; any other is refused, not converted. Note
# 'base' is base_link turned through 180 degrees, not another name for it.
base_frame = 'base_link'

# JointJog velocities are matched to these names, in this order.
joint_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# What the servo accepts. A command above one of these is dropped WHOLE, not clamped.
cap_linear_mps = 0.15     # magnitude of a twist's linear part
cap_angular_rps = 0.35    # magnitude of its angular part
cap_joint_rps = 0.35      # per joint

# Dead-man switch: the arm stops this long after the last message it received.
command_timeout_s = 0.15


##################### ADDED CONSTANTS #######################

# Joint-space pose used to unfold at the start and to retreat to after a latch.
# Elbow and wrist_2 sit mid-range here, far from their singular angles.
OPEN_POSE = [0.0, -1.20, -1.60, -1.90, 1.57, 1.57]
UNFOLD_TOL = 0.02          # rad

V_MAX = 0.10               # m/s, safely under cap_linear_mps
V_MIN = 0.008              # m/s, keeps creeping so the tool always arrives
KP = 1.2                   # speed = KP * distance, so the tool brakes smoothly
W_MAX = 0.25               # rad/s, orientation correction (under cap_angular_rps)
J_MAX = 0.30               # rad/s per joint (under cap_joint_rps)
KP_JOINT = 0.8

STOP_TOL = 0.004           # m, arrival radius
DRIFT_TOL = 0.02           # m, re-approach if the hold drifts further than this
HOLD_S = 2.1               # a little over 2 s so a hold is never short

REACH_SOFT, REACH_HARD = 0.68, 0.85       # slow down near full extension
WARN_SCALE = 0.4                          # speed factor while a warning code shows
SING_SOFT, SING_HARD = 0.50, 0.15         # sin() of elbow / wrist_2 thresholds
SING_MIN_SCALE = 0.25

MAX_RECOVERIES = 10                       # safety cap on unlocks over the whole run
RETREAT_TIMEOUT_S = 6.0                   # then unlock again

# Latch handling per waypoint: 1st latch -> recover, retry via a detour;
# 2nd latch on the same waypoint -> recover, skip it and go to the next.
VIA_INWARD = 0.12          # m, pull the detour point toward the base axis (away from full reach)
VIA_LIFT = 0.05            # m, and raise it a little
VIA_Z_MAX = 0.45           # m
VIA_TOL = 0.02             # m, detour point counts as reached inside this


##################### HELPER FUNCTIONS (ADDED) #####################

def clamp_norm(v, maximum):
    '''Scale a vector so its magnitude does not exceed `maximum` (direction kept).'''
    n = math.sqrt(sum(c * c for c in v))
    if n <= maximum or n < 1e-12:
        return tuple(v)
    return tuple(c * maximum / n for c in v)


def quat_mult(a, b):
    '''Hamilton product of two quaternions given as (w, x, y, z).'''
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)


def quat_to_rotvec(q):
    '''Axis * angle of a unit quaternion (w, x, y, z), shortest way round.'''
    w, x, y, z = q
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    n = math.sqrt(x*x + y*y + z*z)
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(n, w)
    return (angle * x / n, angle * y / n, angle * z / n)


##################### CLASS DEFINITION #######################

class arm_waypoints(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the UR7e's tool through the
                    given waypoints using the arm's velocity command interfaces.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class arm_waypoints
        '''

        # use_sim_time is set here, not on the command line, so this node runs on the
        # simulation clock however it is started.
        super().__init__(                                                               # registering node
            'arm_waypoints_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ############ Topic PUBLISHERS ############

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)    # end-effector velocity, in base_link
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)        # per-joint velocity

        ############ Topic SUBSCRIPTIONS ############

        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.jointstatecb, 50)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05                                                             # rate of time to run one control cycle (seconds)
        self.switch_cli = self.create_client(                                           # client used to pick which command topic is live
            SwitchController, f'{servo_ns}/switch_controller')
        self.timer = self.create_timer(control_rate, self.process_waypoints)            # creating a timer based function which gets called on every 0.05 seconds (as defined by 'control_rate' variable)

        self.tcp_pose = None                                                            # tool pose variable (from tcpposecb())
        self.joint_angles = None                                                        # joint feedback variable (from jointstatecb())
        self.arm_status = None                                                          # arm state code variable (from armstatuscb())

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Add any variable your motion needs to keep between cycles.
        #       ->  HINT: Which waypoint you are on, and what the arm is doing about it.

        self.unlock_cli = self.create_client(                                           # clears a protective stop
            Trigger, f'{servo_ns}/unlock_protective_stop')

        # Which command interface is live (the servo starts on end-effector servoing)
        self.mode = 'twist_controller'
        self.switch_fut = None
        self.switch_target = None

        # What the arm is doing: UNFOLD -> MOVE <-> HOLD -> DONE, plus RECOVER / ABORT
        self.state = 'UNFOLD'
        self.wp_index = 0                                                               # which waypoint we are on
        self.hold_t0 = None
        self.target_orientation = None                                                  # (w, x, y, z), held for the whole run

        # Recovery from a latch
        self.recoveries = 0
        self.caution = 1.0                                                              # shrinks after each recovery
        self.resume_state = 'UNFOLD'
        self.rec_step = 'unlock'
        self.unlock_fut = None
        self.rec_t0 = None
        self.wp_attempt = 0                                                             # latches seen on the current waypoint
        self.via = None                                                                 # detour point for the retry, if any

        self.last_status_logged = None

        ############################################


    def tcpposecb(self, data):
        '''
        Description:    Callback function for the tool pose topic.
                        Use this function to receive where the tool currently is.

        Args:
            data (PoseStamped):    Pose of the tool, reported in base_link

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Store the tool's position and orientation.
        #       ->  HINT: data.pose.position, data.pose.orientation

        self.tcp_pose = data.pose

        ############################################


    def jointstatecb(self, data):
        '''
        Description:    Callback function for the joint states topic.
                        Use this function to receive the current angle of each joint.

        Args:
            data (JointState):    Joint feedback published by the arm

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Store the joint angles, matched BY NAME - the order is not promised.
        #       ->  HINT: angles = [data.position[data.name.index(j)] for j in joint_names]
        #       ->  NOTE: A joint assumed to be at zero when it is not reads as a large
        #                 error, and a controller acting on it drives hard towards it.

        try:
            self.joint_angles = [data.position[data.name.index(j)] for j in joint_names]
        except ValueError:
            pass                                                                        # keep the last good angles

        ############################################


    def armstatuscb(self, data):
        '''
        Description:    Callback function for the arm status topic.
                        Use this function to receive the arm's current state code.

        Args:
            data (Int32):    One state code describing what the arm is doing

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Store the code, and log it while you are developing. Zero is healthy; anything
        #       else is the arm telling you about the last command or its own state.
        #       ->  HINT: /arm_status_detail says the same thing in words.
        #       ->  NOTE: A protective stop LATCHES and cannot be cleared - the run is over.

        self.arm_status = data.data
        if data.data != self.last_status_logged:                                        # log changes only
            self.get_logger().info(f'arm_status: {data.data}')
            self.last_status_logged = data.data

        ############################################


    def switch_controller(self, controller):
        '''
        Description:    Function to make one of the arm's two command interfaces the active
                        one, so that commands published to it are acted on.

        Args:
            controller  (str):      Name of the controller to activate, either
                                    'twist_controller' or 'joint_controller'

        Returns:
            success     (bool):     Whether the controller was activated
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Call the switch_controller service on self.switch_cli-
        #           req = SwitchController.Request()
        #           req.activate_controllers = [ the one you want ]
        #           req.deactivate_controllers = [ the other one ]
        #           req.strictness = SwitchController.Request.STRICT

        #   ->  Wait for it first, and expect the first attempt to fail-
        #           self.switch_cli.wait_for_service(timeout_sec=20.0)
        #           future = self.switch_cli.call_async(req)
        #           rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        #       ->  NOTE: That last call HANGS if made from a callback of a node that is
        #                 already spinning. __init__ is safe; from the timer, poll
        #                 'future.done()' instead.

        #   ->  Do not switch more often than you need to.

        # Non-blocking: called every tick from the timer until it returns True.
        if controller not in ('twist_controller', 'joint_controller'):
            self.get_logger().error(f'Unknown controller: {controller}')
            return False

        if self.mode == controller:
            return True                                                                 # already live, nothing to do

        if self.switch_fut is None:                                                     # no request in flight: send one
            if not self.switch_cli.service_is_ready():
                return False                                                            # first attempts are expected to fail
            req = SwitchController.Request()
            if controller == 'twist_controller':
                req.activate_controllers = [twist_controller]
                req.deactivate_controllers = [joint_controller]
            else:
                req.activate_controllers = [joint_controller]
                req.deactivate_controllers = [twist_controller]
            req.strictness = SwitchController.Request.STRICT
            self.switch_fut = self.switch_cli.call_async(req)
            self.switch_target = controller
            return False

        if self.switch_fut.done():                                                      # request finished: poll, never spin
            res = self.switch_fut.result()
            if res is not None and res.ok:
                self.mode = self.switch_target
                self.get_logger().info(f'controller -> {self.switch_target}')
            self.switch_fut = None
            return self.mode == controller

        return False

        ############################################


    def process_waypoints(self):
        '''
        Description:    Timer function used to drive the tool through the waypoints.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Return early until pose, joints and status have all arrived.

        #   ->  Both topics carry VELOCITIES, never positions.

        #   ->  Build the message for whichever interface you made active:
        #           TwistStamped on self.twist_pub  - linear and angular velocity, in 'base_frame'
        #           JointJog on self.joint_pub      - 'joint_names' and a velocity for each
        #       ->  HINT: msg.header.stamp = self.get_clock().now().to_msg()      (both)
        #                 msg.header.frame_id = base_frame                        (twist)
        #                 msg.twist.linear.x/.y/.z, msg.twist.angular.x/.y/.z     (twist)
        #                 msg.joint_names = joint_names, msg.velocities = [...]   (jog)

        #   ->  Publish on EVERY tick, zero included - see 'command_timeout_s'. Never sleep
        #       inside this function.

        #   ->  Drive the tool at a velocity proportional to the error-
        #           v = Kp * (target - current)
        #       ->  HINT: Cap it by scaling the WHOLE vector: v = v * (cap / |v|)
        #       ->  NOTE: The servo ramps down rather than stopping dead, so the arm settles
        #                 PAST the pose that satisfied you.

        #   ->  Command the tool's ORIENTATION too, or the servo decides the wrist for you.
        #       ->  HINT: The turn from unit vector 'a' onto unit vector 'b', as an axis
        #                 times an angle-
        #                     c     = cross(a, b)
        #                     e     = c / |c| * atan2(|c|, dot(a, b))
        #                     omega = Kp * e
        #                 'a' is the tool's own axis - a column of
        #                 Rotation.from_quat([x, y, z, w]).as_matrix().

        #   ->  Think about the PATH. The arm has joint limits and configurations it cannot
        #       pass through.

        #   ->  Track which waypoint you are on, and log the distance to it while developing.

        if self.tcp_pose is None or self.joint_angles is None or self.arm_status is None:
            return                                                                      # wait for pose, joints and status

        status = self.arm_status

        if self.state in ('DONE', 'ABORT'):
            self._publish_zero()
            return

        if status == 92:                                                                # description modified: never clears
            self.get_logger().error('arm description modified (92): cannot recover')
            self.state = 'ABORT'
            self._publish_zero()
            return
        if status >= 90:                                                                # not ready / stale joints: wait it out
            self._publish_zero()
            return

        # A critical code latches the arm (21 = singularity). Recover instead of giving up.
        if status >= 20 and self.state != 'RECOVER':
            self._start_recovery()
            if self.state == 'ABORT':
                self._publish_zero()
                return
        if self.state == 'RECOVER':
            self._recover()
            return

        # Make sure the right interface is live before commanding it
        want = 'joint_controller' if self.state == 'UNFOLD' else 'twist_controller'
        if not self.switch_controller(want):
            self._publish_zero()
            return

        # The start pose is 20 deg from the elbow limit: unfold in joint space first
        if self.state == 'UNFOLD':
            errors = self._open_pose_error()
            if max(abs(e) for e in errors) < UNFOLD_TOL:
                self._publish_zero()
                o = self.tcp_pose.orientation
                self.target_orientation = (o.w, o.x, o.y, o.z)                          # hold this orientation all run
                self.state = 'MOVE'
                self.get_logger().info('unfolded, starting waypoints')
            else:
                self._publish_joint([KP_JOINT * e for e in errors])
            return

        err, dist = self._error_to(waypoints[self.wp_index])
        self.get_logger().info(
            f'WP{self.wp_index + 1} dist={dist * 1000:.1f}mm state={self.state}',
            throttle_duration_sec=1.0)

        if self.state == 'MOVE':
            if self.via is not None:                                                    # retry path: go via the detour point first
                v_err, v_dist = self._error_to(self.via)
                if v_dist <= VIA_TOL:
                    self.get_logger().info('detour point reached, heading to waypoint')
                    self.via = None
                else:
                    speed = max(V_MIN, min(V_MAX * self._speed_scale(), KP * v_dist))
                    lin = tuple(c / v_dist * speed for c in v_err)
                    self._publish_twist(lin, self._hold_orientation())
                    return
            if dist <= STOP_TOL:
                self.state = 'HOLD'
                self.hold_t0 = self.get_clock().now()
                self._publish_zero()
                return
            # speed ~ distance (brakes before the target), capped and de-rated near singularities
            speed = max(V_MIN, min(V_MAX * self._speed_scale(), KP * dist))
            lin = tuple(c / dist * speed for c in err)
            self._publish_twist(lin, self._hold_orientation())

        elif self.state == 'HOLD':
            self._publish_twist((0.0, 0.0, 0.0), self._hold_orientation())              # zero position velocity, keep attitude
            if dist > DRIFT_TOL:
                self.state = 'MOVE'
                return
            held = (self.get_clock().now() - self.hold_t0).nanoseconds / 1e9
            if held >= HOLD_S:
                self.get_logger().info(f'WP{self.wp_index + 1} done ({dist * 1000:.1f} mm)')
                self.wp_index += 1
                self.wp_attempt = 0
                self.via = None
                if self.wp_index >= len(waypoints):
                    self.state = 'DONE'
                    self.get_logger().info('all waypoints complete')
                else:
                    self.state = 'MOVE'

        ############################################


    ############ ADDED HELPER METHODS ############

    def _publish_twist(self, lin, ang):
        '''Publish one TwistStamped in base_link.'''
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = lin
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = ang
        self.twist_pub.publish(msg)

    def _publish_joint(self, velocities):
        '''Publish one JointJog, each joint clamped to J_MAX.'''
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        msg.velocities = [max(-J_MAX, min(J_MAX, v)) for v in velocities]
        self.joint_pub.publish(msg)

    def _publish_zero(self):
        '''Zero velocity on whichever interface is live (publish every tick).'''
        if self.mode == 'joint_controller':
            self._publish_joint([0.0] * 6)
        else:
            self._publish_twist((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def _error_to(self, target):
        '''Error vector and distance from the tool to a target.'''
        p = self.tcp_pose.position
        e = (target[0] - p.x, target[1] - p.y, target[2] - p.z)
        return e, math.sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2])

    def _open_pose_error(self):
        return [t - c for t, c in zip(OPEN_POSE, self.joint_angles)]

    def _hold_orientation(self):
        '''Angular velocity (base_link) that returns the tool to the captured orientation.'''
        o = self.tcp_pose.orientation
        q_inv = (o.w, -o.x, -o.y, -o.z)
        rv = quat_to_rotvec(quat_mult(self.target_orientation, q_inv))
        return clamp_norm(rv, W_MAX)

    def _speed_scale(self):
        '''
        Slow-down factor in (0, 1]; the smallest cause wins.
          reach   - tool horizontal radius near full extension (elbow singularity)
          sing    - elbow or wrist_2 close to a singular angle (sin -> 0)
          warn    - any warning code (10-19, incl. 11 WARNING_SINGULARITY) showing
          caution - shrinks after every recovery
        '''
        p = self.tcp_pose.position
        r = math.hypot(p.x, p.y)
        reach = 1.0 - 0.7 * min(1.0, max(0.0, (r - REACH_SOFT) / (REACH_HARD - REACH_SOFT)))

        s = min(abs(math.sin(self.joint_angles[2])), abs(math.sin(self.joint_angles[4])))
        f = min(1.0, max(0.0, (s - SING_HARD) / (SING_SOFT - SING_HARD)))
        sing = SING_MIN_SCALE + (1.0 - SING_MIN_SCALE) * f

        warn = WARN_SCALE if 10 <= self.arm_status < 20 else 1.0
        return min(reach, sing, warn) * self.caution

    def _start_recovery(self):
        '''Begin (or restart) recovery from a latched critical status.'''
        self.recoveries += 1
        if self.recoveries > MAX_RECOVERIES:
            self.get_logger().error('too many latches, giving up')
            self.state = 'ABORT'
            return
        if self.state != 'RECOVER':                                                     # a fresh latch (not an unlock retry)
            if self.target_orientation is None:
                self.resume_state = 'UNFOLD'
            else:
                self.resume_state = 'MOVE'
                self.wp_attempt += 1                                                    # this waypoint's path just failed
        self.caution = max(0.4, self.caution * 0.7)                                     # approach more gently next time
        self.get_logger().warn(
            f'latched (status {self.arm_status}); recovery {self.recoveries}/{MAX_RECOVERIES}')
        self.state = 'RECOVER'
        self.rec_step = 'unlock'
        self.unlock_fut = None
        self.rec_t0 = None

    def _plan_detour(self):
        '''
        A different route to the current waypoint: via a point halfway there, pulled toward
        the base axis (away from full extension) and lifted slightly. The tool then arrives
        from inside instead of repeating the path that latched.
        '''
        p = self.tcp_pose.position
        t = waypoints[self.wp_index]
        mx, my, mz = (p.x + t[0]) / 2.0, (p.y + t[1]) / 2.0, (p.z + t[2]) / 2.0
        r = math.hypot(mx, my)
        if r > 0.2:
            k = max(0.0, r - VIA_INWARD) / r
            mx, my = mx * k, my * k
        return (mx, my, min(mz + VIA_LIFT, VIA_Z_MAX))

    def _resume_after_recovery(self):
        '''
        Recovered from the latch. Continue from where the arm now is:
          UNFOLD       -> carry on unfolding
          1st latch    -> retry the same waypoint along a detour
          2nd latch    -> give up on this waypoint, go to the next one
        '''
        self._publish_zero()
        self.get_logger().info('recovered, resuming')
        self.state = self.resume_state
        if self.resume_state != 'MOVE':
            return
        self.via = None
        if self.wp_attempt == 1:
            self.via = self._plan_detour()
            self.get_logger().warn(
                f'WP{self.wp_index + 1}: retrying via detour '
                f'({self.via[0]:.3f}, {self.via[1]:.3f}, {self.via[2]:.3f})')
        else:
            self.get_logger().warn(f'WP{self.wp_index + 1}: latched twice, skipping it')
            self.wp_index += 1
            self.wp_attempt = 0
            if self.wp_index >= len(waypoints):
                self.state = 'DONE'
                self.get_logger().info('no waypoints left')

    def _recover(self):
        '''
        unlock_protective_stop, then retreat in joint space to OPEN_POSE (elbow and wrist_2
        mid-range) until the status clears. The 5 s recovery window counts commanding time,
        so the controller switch in between does not use it up.
        '''
        if self.rec_step == 'unlock':
            self._publish_zero()
            if self.unlock_fut is None:
                if self.unlock_cli.service_is_ready():
                    self.unlock_fut = self.unlock_cli.call_async(Trigger.Request())
            elif self.unlock_fut.done():
                res = self.unlock_fut.result()
                self.unlock_fut = None
                if res is not None and res.success:
                    self.rec_step = 'retreat'
                    self.rec_t0 = None
            return

        if not self.switch_controller('joint_controller'):
            self._publish_zero()
            return
        if self.rec_t0 is None:
            self.rec_t0 = self.get_clock().now()                                        # the window starts here
        t = (self.get_clock().now() - self.rec_t0).nanoseconds / 1e9
        errors = self._open_pose_error()
        max_err = max(abs(e) for e in errors)

        if self.arm_status < 20 and t >= 1.0 and (self.arm_status == 0 or max_err < 0.1):
            self._resume_after_recovery()
            return
        if t > RETREAT_TIMEOUT_S and self.arm_status >= 20:                             # window expired: unlock again
            self._start_recovery()
            return
        self._publish_joint([KP_JOINT * e for e in errors])

    ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    arm_waypoints class to perform its task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('arm_waypoints_process')               # creating ROS node

    node.get_logger().info('Node created: Arm waypoints process')   # logging information

    arm_waypoints_class = arm_waypoints()                           # creating a new object for class 'arm_waypoints'

    rclpy.spin(arm_waypoints_class)                                 # spining on the object to make it alive in ROS 2 DDS

    arm_waypoints_class.destroy_node()                              # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':

    main()
