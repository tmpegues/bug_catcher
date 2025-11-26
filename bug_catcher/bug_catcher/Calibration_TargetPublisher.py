"""
This node locates Aruco AR markers in images and publishes their ids and poses.

The node also publishes a target color based on interpolated visual identification at the switch
position, as well as calibration for determining robot base to world enclosure.

This node has been modified from:
https://github.com/JMU-ROBOTICS-VIVA/ros2_aruco/tree/main/ros2_aruco.

Subscriptions:
   /camera/image_raw (sensor_msgs.msg.Image)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)

Published Topics:
    /aruco_markers (bug_catcher_interfaces.msg.ArucoMarkers)
       Provides an array of all poses along with the corresponding
       marker ids.

Parameters:
    marker_size - size of the markers in meters (default .0625)
    aruco_dictionary_id - dictionary that was used to generate markers
                          (default DICT_5X5_250)
    image_topic - image topic to subscribe to (default /camera/image_raw)
    camera_info_topic - camera info topic to subscribe to
                         (default /camera/camera_info)
"""

import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from bug_catcher_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R
from tf_transformations import quaternion_from_matrix


class CalibrationNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("calibration_node")
        # ################################### Begin_Citation[NK1] ##############################
        # Declare and read camera parameters:
        self.declare_parameter(
            name="marker_size",
            value=0.0625,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Size of the markers in meters.",
            ),
        )
        self.declare_parameter(
            name="aruco_dictionary_id",
            value="DICT_5X5_250",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Dictionary that was used to generate markers.",
            ),
        )
        self.declare_parameter(
            name="image_topic",
            value="/camera/camera/color/image_raw",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Image topic to subscribe to.",
            ),
        )
        self.declare_parameter(
            name="camera_info_topic",
            value="/camera/camera/color/camera_info",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera info topic to subscribe to.",
            ),
        )
        self.declare_parameter(
            name="camera_frame",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera optical frame to use.",
            ),
        )
        # Declare tag calibration parameters:
        self.declare_parameter("calibration.tags.tag_1.x", -0.1143)
        self.declare_parameter("calibration.tags.tag_1.y", -0.4572)
        self.declare_parameter("calibration.tags.tag_2.x", -0.1143)
        self.declare_parameter("calibration.tags.tag_2.y", 0.4064)
        self.declare_parameter("calibration.tags.tag_3.x", 0.6858)
        self.declare_parameter("calibration.tags.tag_3.y", 0.4064)
        self.declare_parameter("calibration.tags.tag_4.x", 0.6858)
        self.declare_parameter("calibration.tags.tag_4.y", -0.4572)

        # Read and set the values of each parameter:
        self.marker_size = (self.get_parameter("marker_size").get_parameter_value().double_value)
        dictionary_id_name = (
            self.get_parameter("aruco_dictionary_id").get_parameter_value().string_value
        )
        image_topic = (self.get_parameter("image_topic").get_parameter_value().string_value)
        info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.camera_frame = (self.get_parameter("camera_frame").get_parameter_value().string_value)
        # Set the tag parameter values:
        self.tag_params = {
            1: (self.get_parameter("calibration.tags.tag_2.x").value,
                self.get_parameter("calibration.tags.tag_2.y").value),
            2: (self.get_parameter("calibration.tags.tag_3.x").value,
                self.get_parameter("calibration.tags.tag_3.y").value),
            3: (self.get_parameter("calibration.tags.tag_4.x").value,
                self.get_parameter("calibration.tags.tag_4.y").value),
            4: (self.get_parameter("calibration.tags.tag_1.x").value,
                self.get_parameter("calibration.tags.tag_1.y").value),
        }

        # Make sure we have a valid dictionary id for Aruco markers:
        try:
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            if type(dictionary_id) != type(cv2.aruco.DICT_5X5_100):
                raise AttributeError
        except AttributeError:
            self.get_logger().error(
                "bad aruco_dictionary_id: {}".format(dictionary_id_name)
            )
            options = "\n".join([s for s in dir(cv2.aruco) if s.startswith("DICT")])
            self.get_logger().error("valid options: {}".format(options))

        # SUBSCRIPTIONS:
        # Camera_info subscription
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self.camera_info_callback, qos_profile_sensor_data
        )
        # Camera_image subscription
        self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data
        )

        # PUBLISHERS:
        # Aruco marker publisher: (ID,Pose)
        self.markers_pub = self.create_publisher(ArucoMarkers, "aruco_markers", 10)
        self.poses_pub = self.create_publisher(PoseArray, "aruco_poses", 10)

        # Listener:
        # The buffer stores received tf frames
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        # Broadcaster:
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.dynamic_broadcaster = TransformBroadcaster(self)

        # Establish the transforms of Robot to Tags that we already know.
        #   These are established from environment set up (Designed to be constant/known)
        for marker_id, (x, y) in self.tag_params.items():
            static_tf = TransformStamped()
            static_tf.header.stamp = self.get_clock().now().to_msg()
            static_tf.header.frame_id = 'base'
            static_tf.child_frame_id = f"aruco_{marker_id}"
            static_tf.transform.translation.x = x
            static_tf.transform.translation.y = y
            static_tf.transform.translation.z = 0.025    # Z offset of base on table top.
            static_tf.transform.rotation.x = 0.0
            static_tf.transform.rotation.y = 0.0
            static_tf.transform.rotation.z = 0.0
            static_tf.transform.rotation.w = 1.0
            self.static_broadcaster.sendTransform(static_tf)

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.aruco_dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self.bridge = CvBridge()
    # ################################### End_Citation[NK1] ##############################

    def camera_info_callback(self, info_msg):
        """
        Retrieve the camera Intrinsics.

        Params:
        ------
        info_msg - Camera intrinsic values to be parsed and set for aruco node detection.

        """
        self.get_logger().info('Getting Camera Intrinsics...')
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)
        # Assume that camera parameters will remain the same during simulation and delete:
        self.destroy_subscription(self.info_sub)

    def calibrateCamera_Aruco(self, markers, num_markers):
        """
        Update the marker frames and publishes them.

        Args_
            location: The location of the marker.
            orientation: The orientation of the marker.

        """
        # Get the position of the camera to marker, then invert:
        marker_camera_tf = {}  # Initialize a dictionary to hold transforms from camera.
        for i in range(num_markers):
            # Convert the camera transform to a 4x4 transformation matrix:
            cam_rot = R.from_quat([
                markers.poses[i].orientation.x,
                markers.poses[i].orientation.y,
                markers.poses[i].orientation.z,
                markers.poses[i].orientation.w,
            ]).as_matrix()
            cam_trans = np.array([
                markers.poses[i].position.x,
                markers.poses[i].position.y,
                markers.poses[i].position.z,
            ]).reshape(3, 1)
            cam_matrix = np.eye(4)
            cam_matrix[:3, :3] = cam_rot
            cam_matrix[:3, 3] = cam_trans.flatten()

            # Invert the matrix:
            cam_matrix = np.linalg.inv(cam_matrix)
            marker_camera_tf[markers.marker_ids[i]] = cam_matrix

        # Average all base to camera transforms:
        translations = np.array([m[:3, 3] for m in marker_camera_tf.values()])
        avg_translation = translations.mean(axis=0)

        quats = np.array([R.from_matrix(m[:3, :3]).as_quat() for m in marker_camera_tf.values()])
        avg_quat = R.from_quat(quats).mean().as_quat()
        avg_quat /= np.linalg.norm(avg_quat)

        # Create final 4x4 matrix of averaged location data:
        marker_camera_avg = np.eye(4)
        marker_camera_avg[:3, :3] = R.from_quat(avg_quat).as_matrix()
        marker_camera_avg[:3, 3] = avg_translation

        # Connect the Robot using a dynamic broadcaster from world to brick
        marker_transform = TransformStamped()
        marker_transform.header.stamp = self.get_clock().now().to_msg()
        marker_transform.header.frame_id = f'aruco_{markers.marker_ids[i]}'
        marker_transform.child_frame_id = 'camera_link'
        # Set the location of the marker frame:
        marker_transform.transform.translation.x = marker_camera_avg[0, 3]
        marker_transform.transform.translation.y = marker_camera_avg[1, 3]
        marker_transform.transform.translation.z = marker_camera_avg[2, 3]
        # Compute the Quaternion:
        q = quaternion_from_matrix(marker_camera_avg)
        marker_transform.transform.rotation.x = q[0]
        marker_transform.transform.rotation.y = q[1]
        marker_transform.transform.rotation.z = q[2]
        marker_transform.transform.rotation.w = q[3]
        self.dynamic_broadcaster.sendTransform(marker_transform)

    def image_callback(self, img_msg):
        """Update each frame by publishing the position of the markers."""
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return
        # Convert the image message to cv2:
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")

        # Establish the Marker and position array:
        markers = ArucoMarkers()
        pose_array = PoseArray()

        # Set the camera frame:
        if self.camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame

        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        # Get the corners and marker IDs of each marker in the frame:
        corners, marker_ids, rejected = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dictionary, parameters=self.aruco_parameters
        )

        # If marker IDs is not None, then get the position of the markers w.r.t the camera:
        # Also, calibrate the robot base to the camera if atleast 2 markers have been identified.
        if marker_ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.intrinsic_mat, self.distortion)
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(cv_image, corners, marker_ids)
            for i in range(len(marker_ids)):
                cv2.drawFrameAxes(cv_image, self.intrinsic_mat, self.distortion, rvecs[i], tvecs[i], length=0.1)

            for i, marker_id in enumerate(marker_ids):
                pose = Pose()
                pose.position.x = float(tvecs[i][0][0])
                pose.position.y = float(tvecs[i][0][1])
                pose.position.z = float(tvecs[i][0][2])

                rot_matrix = np.eye(4)
                rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                quat = quaternion_from_matrix(rot_matrix)

                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]

                pose_array.poses.append(pose)
                markers.poses.append(pose)
                markers.marker_ids.append(marker_id[0])

            # Publish the markers to the topic:
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

            # Publish the camera relative to the markers to the tf:
            self.calibrateCamera_Aruco(markers, len(pose_array.poses))


def main():
    rclpy.init()
    node = CalibrationNode()
    rclpy.spin(node)
    # Destroy all nodes and windows:
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
