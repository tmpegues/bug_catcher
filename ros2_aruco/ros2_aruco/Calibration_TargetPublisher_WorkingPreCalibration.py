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
    /aruco_markers (ros2_aruco_interfaces.msg.ArucoMarkers)
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
from rclpy.time import Duration
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from geometry_msgs.msg import TransformStamped
import tf2_ros
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R
from tf_transformations import quaternion_from_matrix


class CalibrationNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("calibration_node")
        self.get_logger().info("CalibrationNode STARTING __init__")

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
            "1": (self.get_parameter("calibration.tags.tag_1.x").value,
                  self.get_parameter("calibration.tags.tag_1.y").value),
            "2": (self.get_parameter("calibration.tags.tag_2.x").value,
                  self.get_parameter("calibration.tags.tag_2.y").value),
            "3": (self.get_parameter("calibration.tags.tag_3.x").value,
                  self.get_parameter("calibration.tags.tag_3.y").value),
            "4": (self.get_parameter("calibration.tags.tag_4.x").value,
                  self.get_parameter("calibration.tags.tag_4.y").value),
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

        # Set an initial pre-calibrated location between the base and the camera:
        base_camera_tf = TransformStamped()
        base_camera_tf.header.stamp = self.get_clock().now().to_msg()
        base_camera_tf.header.frame_id = 'base'          # Will relate it to base of the robot
        # This will be the location of the camera.
        base_camera_tf.child_frame_id = 'camera_link'
        base_camera_tf.transform.translation.x = 0.05     # Set camera location
        base_camera_tf.transform.translation.y = 0.0
        base_camera_tf.transform.translation.z = 2.0
        base_camera_tf.transform.rotation.x = 0.0
        base_camera_tf.transform.rotation.y = 0.0
        base_camera_tf.transform.rotation.z = 0.0
        base_camera_tf.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(base_camera_tf)

        # Establish the transforms of Robot to Tags that we already know.
        #   These are established from environment set up (Designed to be constant/known)
        for marker_id, (x, y) in self.tag_params.items():
            static_tf = TransformStamped()
            static_tf.header.stamp = self.get_clock().now().to_msg()
            static_tf.header.frame_id = 'base'
            static_tf.child_frame_id = f"aruco_{marker_id}"
            static_tf.transform.translation.x = x
            static_tf.transform.translation.y = y
            static_tf.transform.translation.z = 0.0
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
        self.get_logger().info("CalibrationNode FINISHED __init__")
    # ################################### End_Citation[NK1] ##############################

    def camera_info_callback(self, info_msg):
        self.get_logger().info('Getting Camera Intrinsics...')
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)
        # Assume that camera parameters will remain the same during simulation and delete:
        self.destroy_subscription(self.info_sub)

    def update_markerframes(self, markers, num_markers):
        """
        Update the marker frames and publishes them.

        Args_
            location: The location of the marker.
            orientation: The orientation of the marker.

        """
        self.get_logger().info('Publishing marker frames...')
        for i in range(num_markers):
            # Connect the Robot using a dynamic broadcaster from world to brick
            marker_transform = TransformStamped()
            marker_transform.header.stamp = self.get_clock().now().to_msg()
            marker_transform.header.frame_id = 'camera_color_optical_frame'
            marker_transform.child_frame_id = f'aruco_{markers.marker_ids[i]}'

            # Set the location of the marker frame:
            marker_transform.transform.translation.x = markers.poses[i].position.x
            marker_transform.transform.translation.y = markers.poses[i].position.y
            marker_transform.transform.translation.z = markers.poses[i].position.z
            marker_transform.transform.rotation.x = markers.poses[i].orientation.x
            marker_transform.transform.rotation.y = markers.poses[i].orientation.y
            marker_transform.transform.rotation.z = markers.poses[i].orientation.z
            marker_transform.transform.rotation.w = markers.poses[i].orientation.w
            try:
                self.dynamic_broadcaster.sendTransform(marker_transform)
            except Exception as e:
                self.get_logger().error(f"Failed to send dynamic transform for {marker_transform.child_frame_id}: {e}")

    def calibrate(self, markers, num_markers):
        """
        Calibrate the camera to the base of the robot if atleast 2 markers are identified.

        Args_
            Markers: The markers being used for calibration  (ID, Pose)
            num_markers: The number of markers being used

        """
        # Given the markers identified, calculate and average the transform from camera to base.
        self.get_logger().info('Calibrating Camera Position to Robot base...')
        camera_marker_tf = {}  # Initialize a dictionary to hold transforms from camera.
        robot_marker_tf = {}  # Initialize a dictionary to hold known transforms
        for i in range(num_markers):
            # Get the latest transform between the marker and camera published:
            try:
                trans = self.buffer.lookup_transform(
                    'camera_color_optical_frame',
                    f'aruco_{markers.marker_ids[i]}',
                    rclpy.time.Time(),
                    Duration(seconds=1.0)
                )
                camera_marker_tf[markers.marker_ids[i]] = trans
            except (tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException) as e:
                self.get_logger().warning(f"Transform for marker {markers.marker_ids[i]} not available: {e}")

            # We should now have a dict of the transforms that exist from camera to markers:
            try:
                trans = self.buffer.lookup_transform(
                    'base',
                    f'aruco_{markers.marker_ids[i]}',
                    rclpy.time.Time(),
                    Duration(seconds=1.0)
                )
                robot_marker_tf[markers.marker_ids[i]] = trans
            except (tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException) as e:
                self.get_logger().warning(f"Transform for marker {markers.marker_ids[i]} not available: {e}")

        # We now have the camera to marker tranfroms and the base to marker transforms.
        #   It is now possible to calibrate the position of the base relative to the camera.
        base_camera_matrices = []   # Store all matrices from the tracked markers.

        for marker_id in camera_marker_tf.keys():
            cam_tf = camera_marker_tf[marker_id]
            base_tf = robot_marker_tf[marker_id]

            # Convert the camera transform to a 4x4 transformation matrix:
            cam_rot = R.from_quat([
                cam_tf.transform.rotation.x,
                cam_tf.transform.rotation.y,
                cam_tf.transform.rotation.z,
                cam_tf.transform.rotation.w
            ]).as_matrix()
            cam_trans = np.array([
                cam_tf.transform.translation.x,
                cam_tf.transform.translation.y,
                cam_tf.transform.translation.z
            ]).reshape(3, 1)
            cam_matrix = np.eye(4)
            cam_matrix[:3, :3] = cam_rot
            cam_matrix[:3, 3] = cam_trans.flatten()

            # Convert the base transform to a 4x4 transformation matrix:
            base_rot = R.from_quat([
                base_tf.transform.rotation.x,
                base_tf.transform.rotation.y,
                base_tf.transform.rotation.z,
                base_tf.transform.rotation.w
            ]).as_matrix()
            base_trans = np.array([
                base_tf.transform.translation.x,
                base_tf.transform.translation.y,
                base_tf.transform.translation.z
            ]).reshape(3, 1)
            base_matrix = np.eye(4)
            base_matrix[:3, :3] = base_rot
            base_matrix[:3, 3] = base_trans.flatten()

            # Compute base to camera transformation:
            base_camera_matrix = base_matrix @ np.linalg.inv(cam_matrix)
            base_camera_matrices.append(base_camera_matrix)

        # Average all base to camera transforms:
        translations = np.array([m[:3, 3] for m in base_camera_matrices])
        avg_translation = translations.mean(axis=0)

        quats = np.array([R.from_matrix(m[:3, :3]).as_quat() for m in base_camera_matrices])
        avg_quat = quats.mean(axis=0)
        avg_quat /= np.linalg.norm(avg_quat)

        # Create final 4x4 matrix of averaged location data:
        base_camera_avg = np.eye(4)
        base_camera_avg[:3, :3] = R.from_quat(avg_quat).as_matrix()
        base_camera_avg[:3, 3] = avg_translation

        # Update the static transform of the base to camera transformation to the tf:
        base_camera_tf = TransformStamped()
        base_camera_tf.header.stamp = self.get_clock().now().to_msg()
        base_camera_tf.header.frame_id = 'base'          # Will relate it to base of the robot
        # This will be the location of the camera.
        base_camera_tf.child_frame_id = 'camera_color_optical_frame'
        base_camera_tf.transform.translation.x = base_camera_avg[0, 3]
        base_camera_tf.transform.translation.y = base_camera_avg[1, 3]
        base_camera_tf.transform.translation.z = base_camera_avg[2, 3]
        # Compute the Quaternion:
        q = quaternion_from_matrix(base_camera_avg[:3, :3])
        base_camera_tf.transform.rotation.x = q[0]
        base_camera_tf.transform.rotation.y = q[1]
        base_camera_tf.transform.rotation.z = q[2]
        base_camera_tf.transform.rotation.w = q[3]
        self.dynamic_broadcaster.sendTransform(base_camera_tf)

    def image_callback(self, img_msg):
        """Update each frame by publishing the position of the markers."""
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return
        self.get_logger().info('Getting image...')
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

            # Publish the frame of the markers to the tf tree:
            self.update_markerframes(markers, len(pose_array.poses))

            # Calibrate the Camera to the Robot and publish markers to the tf_tree:
            if len(pose_array.poses) >= 2:
                self.calibrate(markers, len(pose_array.poses))


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
