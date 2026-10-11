#!/usr/bin/env python3


"""
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1A of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
"""

# Team ID:           eYRC#4101
# Author List:		Joyal George,Adithyan A,Anandhu Saji,P H Girisankar
# Filename:		    ore_detector.py
# Functions:
# 			        detect_ores, main
# Nodes:
# 			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/camera/color/image_raw,
#                                          /camera/camera/aligned_depth_to_color/image_raw,
#                                          /camera/camera/color/camera_info, /tf ]


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import tf2_ros
import numpy as np
import math
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, PointStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point

##################### TASK CONSTANTS #######################

# Two ores of each type are spawned - six in all - told apart by an id of 1 or 2.
ore_types = ["azurite_ore", "malachite_ore", "vanadinite_ore"]

# The RealSense topics. The depth image is ALIGNED to the colour image.
color_topic = "/camera/camera/color/image_raw"
depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"
camera_info_topic = "/camera/camera/color/camera_info"

# The parent frame every ore transform is published against.
base_frame = "base_link"

# ---- Tuning constants ----

# HSV bounds per ore type (H is 0-179, S and V are 0-255, the frame is BGR).
HSV_RANGES = {
    "azurite_ore": ((100, 150, 100), (110, 255, 255)),
    "malachite_ore": ((70, 150, 100), (80, 255, 255)),
    "vanadinite_ore": ((5, 220, 150), (15, 255, 255)),
}

MIN_AREA = 150                  # px^2, smaller regions are not ores
MATCH_DIST_PX = 100.0           # px, max movement for a detection to keep its id
DEPTH_WINDOWS = (4, 8, 16, 32)  # half-sizes (px) tried in turn until a valid depth is found
TOP_TO_MIDDLE_OFFSET = 0.0      # m, camera sees the TOP face; set to half the ore height once measured

DEBUG_WINDOW = True             # show the annotated frame (click on it to print HSV values)
DEBUG_SAVE_PATH = None          # e.g. "/tmp/ores.png" to save the annotated frame each cycle

# Filled by detect_ores: areas of regions that were found but rejected as too small.
DETECT_DEBUG = {}


##################### FUNCTION DEFINITIONS #######################


def detect_ores(image):
    """
    Description:    Function to detect the ores present in a colour image frame and
                    return the pixel location and the type of each one found.

    Args:
        image                   (Image):    Input colour image frame received from the camera topic

    Returns:
        center_ore_list         (list):     Center pixel (cX, cY) of every ore detected in the frame
        ore_type_list           (list):     Type of each ore detected, taken from 'ore_types'
    """

    center_ore_list = []
    ore_type_list = []

    ############ ADD YOUR CODE HERE ############

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)

    for ore_type in ore_types:
        lower, upper = HSV_RANGES[ore_type]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Clean the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        DETECT_DEBUG[ore_type] = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < MIN_AREA:
                DETECT_DEBUG[ore_type].append(int(area))    # remember what was rejected
                continue

            M = cv2.moments(contour)

            if M["m00"] == 0:
                continue

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            center_ore_list.append((cX, cY))
            ore_type_list.append(ore_type)

            # Draw detection for debugging
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

    ############################################

    return center_ore_list, ore_type_list


##################### CLASS DEFINITION #######################


class ore_tf(Node):
    """
    ___CLASS___

    Description:    Class which serves the purpose to detect the ores in the cell and
                    broadcast a transform for each one.
    """

    def __init__(self):
        """
        Description:    Initialization of class ore_tf
        """

        super().__init__("ore_tf_publisher")  # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(
            Image, color_topic, self.colorimagecb, 10
        )
        self.depth_cam_sub = self.create_subscription(
            Image, depth_topic, self.depthimagecb, 10
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.caminfocb, 10
        )

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2  # rate of time to process image (seconds)
        self.bridge = CvBridge()  # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.Buffer()

        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(
            self
        )  # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(
            image_processing_rate, self.process_image
        )  # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)

        self.cv_image = None  # colour raw image variable (from colorimagecb())
        self.depth_image = None  # depth image variable (from depthimagecb())
        self.cam_info = None  # camera intrinsics variable (from caminfocb())

        ############ ADD YOUR CODE HERE ############

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_frame = None

        # One track per ore: {"id": 1 or 2, "pixel": (u, v), "pos": (x, y, z) in base_link or None}.
        # Tracks live for the whole run so an id never changes, and every track with a
        # known position is broadcast on every cycle, even if the ore is missed in one frame.
        self.ore_tracks = {ore_type: [] for ore_type in ore_types}

        self.debug_window = DEBUG_WINDOW
        if self.debug_window:
            try:
                cv2.namedWindow("Camera Image")
                cv2.setMouseCallback("Camera Image", self.on_click)
            except Exception as e:
                self.get_logger().warn(f"No display available, debug window off: {e}")
                self.debug_window = False

        ############################################

    def on_click(self, event, x, y, flags, param):
        """Print the HSV value under the mouse click (used to tune HSV_RANGES)."""
        if event == cv2.EVENT_LBUTTONDOWN and self.cv_image is not None:
            h, s, v = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)[y, x]
            self.get_logger().info(f"HSV at ({x}, {y}) = ({h}, {s}, {v})")

    def depthimagecb(self, data):
        """
        Description:    Callback function for the aligned depth camera topic.
                        Use this function to receive the depth image and convert it to a CV2 image.

        Args:
            data (Image):    Input depth image frame received from the aligned depth camera topic

        Returns:
        """

        ############ ADD YOUR CODE HERE ############

        depth = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

        # 16UC1 is MILLIMETRES, 32FC1 is already METRES.
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0

        self.depth_image = depth

        ############################################

    def colorimagecb(self, data):
        """
        Description:    Callback function for the colour camera raw topic.
                        Use this function to receive the raw image and convert it to a CV2 image.

        Args:
            data (Image):    Input coloured raw image frame received from the image_raw camera topic

        Returns:
        """

        ############ ADD YOUR CODE HERE ############

        self.cv_image = self.bridge.imgmsg_to_cv2(
            data, desired_encoding="bgr8"
        )  # IMAGE to CV numpy array

        ############################################

    def caminfocb(self, data):
        """
        Description:    Callback function for the camera info topic.
                        Use this function to receive the camera's intrinsic parameters.

        Args:
            data (CameraInfo):    Camera calibration published by the camera

        Returns:
        """

        ############ ADD YOUR CODE HERE ############

        self.fx = float(data.k[0])
        self.fy = float(data.k[4])
        self.cx = float(data.k[2])
        self.cy = float(data.k[5])

        self.camera_frame = data.header.frame_id

        ############################################

    def get_depth(self, u, v):
        """
        Median of the valid depth readings around pixel (u, v). If the small window holds
        no valid reading (a hole in the depth image), the window is grown until one does.
        """
        h, w = self.depth_image.shape[:2]

        for half in DEPTH_WINDOWS:
            x1, x2 = max(0, u - half), min(w, u + half + 1)
            y1, y2 = max(0, v - half), min(h, v + half + 1)

            patch = self.depth_image[y1:y2, x1:x2]
            valid = patch[np.isfinite(patch) & (patch > 0.0)]

            if valid.size > 0:
                return float(np.median(valid))

        return None

    def update_tracks(self, ore_type, detections):
        """
        Give each detection of one ore type an id (1 or 2).

        Detections are matched ONE-TO-ONE to the tracks already known (closest pixel first),
        so two ores of the same type can never share an id, even when they are close
        together. A detection with no track close enough starts a new track, using whichever
        of the ids 1 / 2 is still free.
        """
        tracks = self.ore_tracks[ore_type]

        pairs = sorted(
            (math.hypot(d["pixel"][0] - t["pixel"][0], d["pixel"][1] - t["pixel"][1]), di, ti)
            for di, d in enumerate(detections)
            for ti, t in enumerate(tracks)
        )

        used_det, used_track = set(), set()

        for dist, di, ti in pairs:
            if dist > MATCH_DIST_PX:
                break
            if di in used_det or ti in used_track:
                continue
            used_det.add(di)
            used_track.add(ti)
            tracks[ti]["pixel"] = detections[di]["pixel"]
            if detections[di]["pos"] is not None:
                tracks[ti]["pos"] = detections[di]["pos"]

        for di, d in enumerate(detections):
            if di in used_det:
                continue

            taken = {t["id"] for t in tracks}
            free_ids = [i for i in (1, 2) if i not in taken]

            if not free_ids:
                self.get_logger().warn(
                    f"More than two {ore_type} regions seen - ignoring the extra one",
                    throttle_duration_sec=2.0,
                )
                continue

            tracks.append({"id": free_ids[0], "pixel": d["pixel"], "pos": d["pos"]})
            self.get_logger().info(f"New track: {ore_type}_{free_ids[0]} at pixel {d['pixel']}")

    def process_image(self):
        """
        Description:    Timer function used to detect the ores and publish a transform for
                        each one on its estimated position.

        Args:
        Returns:
        """

        ############ ADD YOUR CODE HERE ############

        # Wait until all required data has arrived
        if (self.cv_image is None or self.depth_image is None
                or self.fx is None or self.camera_frame is None):
            return

        frame = self.cv_image.copy()
        center_ore_list, ore_type_list = detect_ores(frame)

        # Diagnostics: tell which type is short and what was thrown away.
        for ore_type in ore_types:
            n = ore_type_list.count(ore_type)
            if n != 2:
                self.get_logger().warn(
                    f"{ore_type}: found {n}/2 (rejected small regions, px^2: "
                    f"{DETECT_DEBUG.get(ore_type, [])})",
                    throttle_duration_sec=2.0,
                )

        # Camera -> base_link transform, looked up once per cycle.
        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame, self.camera_frame, rclpy.time.Time()
            )
        except tf2_ros.TransformException as ex:
            transform = None
            self.get_logger().warn(f"TF lookup failed: {ex}", throttle_duration_sec=2.0)

        # Measure every detection.
        detections = {ore_type: [] for ore_type in ore_types}

        for (u, v), ore_type in zip(center_ore_list, ore_type_list):

            pos = None
            z = self.get_depth(u, v)

            if z is None:
                self.get_logger().warn(
                    f"No valid depth for {ore_type} at ({u}, {v})", throttle_duration_sec=2.0
                )
            elif transform is not None:
                # Deproject the pixel into the camera optical frame
                point_camera = PointStamped()
                point_camera.header.frame_id = self.camera_frame
                point_camera.header.stamp = self.get_clock().now().to_msg()
                point_camera.point.x = float((u - self.cx) * z / self.fx)
                point_camera.point.y = float((v - self.cy) * z / self.fy)
                point_camera.point.z = float(z)

                point_base = do_transform_point(point_camera, transform)

                # The camera sees the TOP face; the ore is named by its middle.
                pos = (
                    float(point_base.point.x),
                    float(point_base.point.y),
                    float(point_base.point.z) - TOP_TO_MIDDLE_OFFSET,
                )

            detections[ore_type].append({"pixel": (u, v), "pos": pos})

        # Ids: one-to-one matching per type, kept for the whole run.
        for ore_type in ore_types:
            self.update_tracks(ore_type, detections[ore_type])

        # Broadcast EVERY known ore on EVERY cycle (even if missed in this frame).
        for ore_type in ore_types:
            for track in self.ore_tracks[ore_type]:

                u, v = track["pixel"]
                name = f"{ore_type}_{track['id']}"
                cv2.putText(frame, name, (u + 10, v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if track["pos"] is None:
                    continue

                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = base_frame
                t.child_frame_id = name

                t.transform.translation.x = float(track["pos"][0])
                t.transform.translation.y = float(track["pos"][1])
                t.transform.translation.z = float(track["pos"][2])

                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0

                self.br.sendTransform(t)

        n_tracks = sum(len(v) for v in self.ore_tracks.values())
        self.get_logger().info(
            f"detected {len(center_ore_list)} regions, tracking {n_tracks}/6 ores",
            throttle_duration_sec=2.0,
        )

        # Visualization (never allowed to crash the node, e.g. on a machine with no display)
        if DEBUG_SAVE_PATH:
            cv2.imwrite(DEBUG_SAVE_PATH, frame)

        if self.debug_window:
            try:
                cv2.imshow("Camera Image", frame)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warn(f"Debug window disabled: {e}")
                self.debug_window = False

        ############################################


##################### FUNCTION DEFINITION #######################


def main():
    """
    Description:    Main function which creates a ROS node and spins around for the ore_tf
                    class to perform its task
    """

    rclpy.init(args=sys.argv)  # initialisation

    node = rclpy.create_node("ore_tf_process")  # creating ROS node

    node.get_logger().info("Node created: Ore tf process")  # logging information

    ore_tf_class = ore_tf()  # creating a new object for class 'ore_tf'

    rclpy.spin(ore_tf_class)  # spining on the object to make it alive in ROS 2 DDS

    ore_tf_class.destroy_node()  # destroy node after spin ends

    rclpy.shutdown()  # shutdown process


if __name__ == "__main__":

    main()