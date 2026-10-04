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
# 			        [ Comma separated list of functions in this file ]
# Nodes:		    Add your publishing and subscribing node
#                   Example:
# 			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/camera/color/image_raw, /etc... ]


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

    ############ Function VARIABLES ############

    # ->  You can remove these variables if needed. These are just for suggestions to let you get started

    center_ore_list = []
    ore_type_list = []

    ############ ADD YOUR CODE HERE ############
    # Creating the color masks
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # BLUE / AZURITE
    lower_blue = np.array([100, 150, 100])
    upper_blue = np.array([110, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # GREEN / MALACHITE
    lower_green = np.array([70, 150, 100])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # ORANGE / VANADINITE
    lower_orange = np.array([5, 220, 150])
    upper_orange = np.array([15, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    # ============================================================
    # 2. CLEAN THE MASKS
    # ============================================================

    kernel = np.ones((5, 5), np.uint8)

    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

    # ============================================================

    # 3. FIND ORE CONTOURS
    # ============================================================

    masks = [
        (blue_mask, "azurite_ore"),
        (green_mask, "malachite_ore"),
        (orange_mask, "vanadinite_ore"),
    ]
    for mask, ore_type in masks:

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            # Ignore very small regions
            area = cv2.contourArea(contour)

            if area < 150:
                continue

            # Calculate contour center
            M = cv2.moments(contour)

            if M["m00"] == 0:
                continue

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Store detection
            center_ore_list.append((cX, cY))
            ore_type_list.append(ore_type)

            # Draw detection for debugging
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

            cv2.putText(
                image,
                f"{ore_type} ({cX},{cY})",
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
            )

    # INSTRUCTIONS & HELP :

    # 	->  Detect the ores by COLOUR, and return a center pixel and a type for each.
    #       ->  HINT: hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #                 mask = cv2.inRange(hsv, lower, upper)       # one pair per type
    #                 H is 0-179, S and V are 0-255. The frame is BGR, not RGB.

    #   ->  Read the bounds off a saved frame rather than copying them from a tutorial.

    #   ->  Clean the mask, and drop anything too small to be an ore.
    #       ->  HINT: cv2.morphologyEx (MORPH_OPEN, MORPH_CLOSE), cv2.contourArea

    #   ->  Reduce each region to one center pixel.
    #       ->  HINT: M  = cv2.moments(contour)                   # guard against m00 == 0
    #                 cX = int(M['m10'] / M['m00'])
    #                 cY = int(M['m01'] / M['m00'])

    #   ->  Draw your detections on the frame while you are developing.
    #       ->  HINT: cv2.circle, cv2.putText

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
        # Stores the pixel position and ID of each ore across frames
        self.ore_tracks = {"azurite_ore": [], "malachite_ore": [], "vanadinite_ore": []}

        # INSTRUCTIONS & HELP :

        # 	->  Add any variable your detection needs to keep between frames.
        #       ->  HINT: The two ores of a type must keep their ids for the whole run, and
        #                 'detect_ores' returns them unordered.

        ############################################

    def depthimagecb(self, data):
        """
        Callback function for depth image topic.
        """
        self.depth_image = self.bridge.imgmsg_to_cv2(
            data, desired_encoding="passthrough"
        )
        """
        Description:    Callback function for the aligned depth camera topic.
                        Use this function to receive the depth image and convert it to a CV2 image.

        Args:
            data (Image):    Input depth image frame received from the aligned depth camera topic

        Returns:
        """

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        # 	->  Convert the ROS Image message to a CV2 image and store it.
        #       ->  HINT: self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

        #   ->  Get the units right. Print 'depth.dtype'.
        #       ->  HINT: 32FC1 is METRES; 16UC1 is MILLIMETRES, so divide by 1000.

        #   ->  Drop the pixels that carry no reading.
        #       ->  HINT: depth[np.isfinite(depth) & (depth > 0.0)]

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

        # INSTRUCTIONS & HELP :

        # 	->  Convert the ROS Image message to a CV2 image and store it.
        #       ->  HINT: self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

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
        """
        Callback function for the camera info topic.
        """

        self.fx = float(data.k[0])
        self.fy = float(data.k[4])
        self.cx = float(data.k[2])
        self.cy = float(data.k[5])

        self.camera_frame = data.header.frame_id

        # INSTRUCTIONS & HELP :

        # 	->  Store the focal lengths and the principal point. Read them from the topic;
        #       never hard-code them.
        #       ->  HINT: 'k' is the pinhole matrix flattened row by row-
        #                     k = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

    ############################################
    def get_median_depth(self, u, v):
        half = 4

        x1 = max(0, u - half)
        x2 = min(self.depth_image.shape[1], u + half + 1)

        y1 = max(0, v - half)
        y2 = min(self.depth_image.shape[0], v + half + 1)

        depth_patch = self.depth_image[y1:y2, x1:x2]

        valid_depth = depth_patch[np.isfinite(depth_patch) & (depth_patch > 0)]

        if len(valid_depth) == 0:
            return None

        return float(np.median(valid_depth))
        ############################################

    def assign_ore_id(self, ore_type, pixel):

        u, v = pixel

        tracks = self.ore_tracks[ore_type]

        # Find the closest previously known ore of this type
        best_track = None
        best_distance = float("inf")

        for track in tracks:

            old_u, old_v = track["pixel"]

            distance = math.sqrt((u - old_u) ** 2 + (v - old_v) ** 2)

            if distance < best_distance:
                best_distance = distance
                best_track = track

        # If a previous ore is close enough, keep its ID
        if best_track is not None and best_distance < 100.0:

            best_track["pixel"] = (u, v)

            return best_track["id"]

        # Otherwise assign a new ID
        used_ids = {track["id"] for track in tracks}

        for new_id in [1, 2]:

            if new_id not in used_ids:

                tracks.append({"id": new_id, "pixel": (u, v)})

                return new_id

        return None
        ############################################

    def process_image(self):
        """
        Description:    Timer function used to detect the ores and publish a transform for
                        each one on its estimated position.

        Args:
        Returns:
        """

        # --------------------------

        ############ ADD YOUR CODE HERE ############

        # Wait until all required data has arrived
        if self.cv_image is None:
            return

        if self.depth_image is None:
            return

        if self.fx is None:
            return

        if self.camera_frame is None:
            return

        # Detect ores
        center_ore_list, ore_type_list = detect_ores(self.cv_image)
        annotated_frame = self.cv_image.copy()

        for center, ore_type in zip(center_ore_list, ore_type_list):

            u, v = center

            # --------------------------------------------------
            # STEP 1: Get median depth
            # --------------------------------------------------

            depth = self.get_median_depth(u, v)

            if depth is None:
                continue

            z = float(depth)

            # --------------------------------------------------
            # STEP 2: Deproject pixel -> camera 3D point
            # --------------------------------------------------

            x = float((u - self.cx) * z / self.fx)
            y = float((v - self.cy) * z / self.fy)

            # --------------------------------------------------
            # STEP 3: Create PointStamped in camera frame
            # --------------------------------------------------

            point_camera = PointStamped()

            point_camera.header.frame_id = self.camera_frame
            point_camera.header.stamp = self.get_clock().now().to_msg()

            point_camera.point.x = x
            point_camera.point.y = y
            point_camera.point.z = z

            # --------------------------------------------------
            # STEP 4: Camera frame -> base_link
            # --------------------------------------------------

            try:

                transform = self.tf_buffer.lookup_transform(
                    base_frame, self.camera_frame, rclpy.time.Time()
                )

                point_base = do_transform_point(point_camera, transform)

            except tf2_ros.TransformException as ex:

                self.get_logger().warn(f"TF lookup failed: {ex}")

                continue

            # --------------------------------------------------
            # STEP 5: Correct top-face measurement
            # --------------------------------------------------

            # IMPORTANT:
            # The depth camera sees the TOP of the ore.
            # The evaluator wants the ore's MIDDLE.
            #
            # We will put the verified correction value here
            # once the ore model height is confirmed.

            corrected_x = float(point_base.point.x)
            corrected_y = float(point_base.point.y)
            corrected_z = float(point_base.point.z)

            # --------------------------------------------------
            # STEP 6: Assign stable ID
            # --------------------------------------------------

            ore_id = self.assign_ore_id(ore_type, (u, v))
            cv2.circle(annotated_frame, (u, v), 5, (0, 255, 0), -1)

            cv2.putText(
                annotated_frame,
                f"{ore_type}_{ore_id}",
                (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            if ore_id is None:
                continue

            # --------------------------------------------------
            # STEP 7: Create TF
            # --------------------------------------------------

            t = TransformStamped()

            t.header.stamp = self.get_clock().now().to_msg()

            t.header.frame_id = base_frame

            t.child_frame_id = f"{ore_type}_{ore_id}"

            t.transform.translation.x = float(corrected_x)
            t.transform.translation.y = float(corrected_y)
            t.transform.translation.z = float(corrected_z)

            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            # --------------------------------------------------
            # STEP 8: Broadcast TF
            # --------------------------------------------------

            self.br.sendTransform(t)

            self.get_logger().info(
                f"{t.child_frame_id}: "
                f"base_link = "
                f"({corrected_x:.3f}, "
                f"{corrected_y:.3f}, "
                f"{corrected_z:.3f})"
            )

        # ------------------------------------------------------
        # OpenCV visualization
        # ------------------------------------------------------
        cv2.imwrite("/home/giri/ros2_ws/src/SC#4101_task1A_detection.png", annotated_frame)

        cv2.imshow("Camera Image", annotated_frame)

        cv2.waitKey(1)

        # self.get_logger().info("process_image is running")
        # if self.cv_image is None:
        #     self.get_logger().info("Waiting for color image")
        #     return

        # if self.depth_image is None:
        #         self.get_logger().info("Waiting for depth image")
        #         return

        # center_ore_list, ore_type_list = detect_ores(self.cv_image)
        # self.get_logger().info(f"Detected {len(center_ore_list)} ores")
        # for center, ore_type in zip(center_ore_list, ore_type_list):

        #         u, v = center

        #         depth = self.get_median_depth(u, v)

        #         if depth is None:
        #             self.get_logger().warn(
        #                 f"Invalid depth for {ore_type} at ({u}, {v})"
        #             )
        #             continue

        #         self.get_logger().info(
        #             f"{ore_type}: pixel=({u}, {v}), depth={depth:.3f} m"
        # )

        # cv2.imshow("Camera Image", self.cv_image)
        # cv2.waitKey(1)
        # INSTRUCTIONS & HELP :

        # 	->  Return early until both images and the camera info have arrived.

        #   ->  Get the ore centers and their types from 'detect_ores' defined above

        #   ->  Read the depth at each center pixel. Depth is ALIGNED to colour.
        #       ->  HINT: Take the MEDIAN of a small window, not the one pixel-
        #                     patch = self.depth_image[cY-4:cY+5, cX-4:cX+5]
        #                     z     = np.median(patch[np.isfinite(patch) & (patch > 0.0)])

        #   ->  Deproject the center pixel (u, v) and its depth z into a 3D point-
        #           x = (u - cx) * z / fx
        #           y = (v - cy) * z / fy
        #           z = z
        #       ->  HINT: That point is in the camera's OPTICAL frame - 'data.header.frame_id'.

        #   ->  Transform it into 'base_frame'-
        #           tf = self.tf_buffer.lookup_transform(
        #                    base_frame, <optical frame>, rclpy.time.Time())
        #           p  = do_transform_point(point_in_camera, tf)
        #       ->  HINT: PointStamped and tf2_geometry_msgs are not imported above, and the
        #                 lookup raises until the tree has filled in.

        #   ->  Correct for what you measured: the camera sees the ore's top face, and the
        #       ore is named by its middle.

        #   ->  Give each ore an id and keep it for the whole run.
        #       ->  HINT: Match a detection to the nearest ore of that type already named,
        #                 on the pixel rather than the depth.

        #   ->  Publish one transform per ore, using Geometry Message - TransformStamped
        #       Use the following frame_id-
        #           frame_id = 'base_link'
        #           child_frame_id = '<ore_type>_<id>'      Ex: azurite_ore_1, where azurite_ore
        #                                                   is one of 'ore_types' and the id is 1 or 2
        #       ->  HINT: t.header.stamp, t.header.frame_id, t.child_frame_id,
        #                 t.transform.translation.x/.y/.z, t.transform.rotation.w = 1.0,
        #                 then self.br.sendTransform(t). Every number must be a float.
        #       ->  NOTE: Only the translation is read; the names are matched exactly.

        #   ->  Broadcast every ore on every cycle, not once.

        #   ->  Show the frame with your detections drawn on it using 'cv2.imshow'.

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
