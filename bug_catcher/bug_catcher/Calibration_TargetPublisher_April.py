"""


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
    #TODO: Add other parameters I have added.
"""

import cv2
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
import rclpy.node
from scipy.spatial.transform import Rotation as R
import tf2_ros
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_matrix
from enum import auto, Enum


class State(Enum):
    """
    Current state of the system.

    Determines what the main timer function should be doing on each
        iteration for the physics of the brick.
    """

    CALIBRATING = auto()
    PUBLISHING = auto()


class CalibrationNode(rclpy.node.Node):

    def __init__(self):
        super().__init__('calibration_node')
        # Declare tag calibration parameters:
        self.declare_parameter('calibration.tags.tag_1.x', -0.1143)
        self.declare_parameter('calibration.tags.tag_1.y', -0.4572)
        self.declare_parameter('calibration.tags.tag_2.x', -0.1143)
        self.declare_parameter('calibration.tags.tag_2.y', 0.4064)
        self.declare_parameter('calibration.tags.tag_3.x', 0.6858)
        self.declare_parameter('calibration.tags.tag_3.y', 0.4064)
        self.declare_parameter('calibration.tags.tag_4.x', 0.6858)
        self.declare_parameter('calibration.tags.tag_4.y', -0.4572)
        # Set the tag parameter values:
        self.tag_params = {
            1: (self.get_parameter('calibration.tags.tag_2.x').get_parameter_value().double_value,
                self.get_parameter('calibration.tags.tag_2.y').get_parameter_value().double_value),
            2: (self.get_parameter('calibration.tags.tag_3.x').get_parameter_value().double_value,
                self.get_parameter('calibration.tags.tag_3.y').get_parameter_value().double_value),
            3: (self.get_parameter('calibration.tags.tag_4.x').get_parameter_value().double_value,
                self.get_parameter('calibration.tags.tag_4.y').get_parameter_value().double_value),
            4: (self.get_parameter('calibration.tags.tag_1.x').get_parameter_value().double_value,
                self.get_parameter('calibration.tags.tag_1.y').get_parameter_value().double_value),
        }
        # SUBSCRIPTIONS:

        # PUBLISHERS:

        # Timer:
        self.timer_update = self.create_timer(0.1, self.calibrate_target_publisher)

        # Listeners:
        # The buffer stores received tf frames:
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        # Broadcasters:
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.dynamic_broadcaster = TransformBroadcaster(self)

        # # Establish the transforms of Robot to Tags that we already know.
        # #   These are established from environment set up (Designed to be constant/known)
        # for marker_id, (x, y) in self.tag_params.items():
        #     static_tf = TransformStamped()
        #     static_tf.header.stamp = self.get_clock().now().to_msg()
        #     static_tf.header.frame_id = 'base'
        #     static_tf.child_frame_id = f'tag_{marker_id}'
        #     static_tf.transform.translation.x = x
        #     static_tf.transform.translation.y = y
        #     static_tf.transform.translation.z = 0.025    # Z offset of base on table top.
        #     static_tf.transform.rotation.x = 0.0
        #     static_tf.transform.rotation.y = 0.0
        #     static_tf.transform.rotation.z = 0.0
        #     static_tf.transform.rotation.w = 1.0
        #     self.static_broadcaster.sendTransform(static_tf)

        # Create and save the matrix version of the base to marker trasforms for static recall:
        self.base_tag = {}
        for marker_id, (x, y) in self.tag_params.items():
            # Translation
            t = np.array([x, y, 0.025])  # Z offset
            # Rotation quaternion:
            q = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w
            # Convert quaternion to 4x4 rotation matrix
            mat = quaternion_matrix(q)
            # Set translation:
            mat[0:3, 3] = t
            # Store in dictionary for easy lookup:
            self.base_tag[marker_id] = mat

        # Establish a temporary transform between base and camera:
        # Create a dynamic transform to connect the tf trees.
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base'
        msg.child_frame_id = 'camera_link'
        # Rough initial pose (Set above robot):
        msg.transform.translation.x = 0.3
        msg.transform.translation.y = 0.0
        msg.transform.translation.z = 1.0
        msg.transform.rotation.x = 0.0
        msg.transform.rotation.y = 0.0
        msg.transform.rotation.z = 0.0
        msg.transform.rotation.w = 1.0
        self.dynamic_broadcaster.sendTransform(msg)

        # # Retrieve the Camera_Link to Camera_Optical Tf:
        # try:
        #     tf_msg = self.buffer.lookup_transform(
        #         'camera_link', 'camera_color_optical_frame', self.get_clock().now().to_msg()
        #     )
        #     # Convert the transform message to a matrix and store.
        #     t = tf_msg.transform.translation
        #     q = tf_msg.transform.rotation
        #     Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        #     # Save this transform statically in the node:  
        #     self.optical_link = np.eye(4)
        #     self.optical_link[:3, :3] = Rm
        #     self.optical_link[:3, 3] = [t.x, t.y, t.z]
        # except (tf2_ros.LookupException,
        #         tf2_ros.ExtrapolationException,
        #         tf2_ros.ConnectivityException) as e:
        #     print(f'Transform for Camera not yet available: {e}')

        # Establish Calibration Averaging Variables:
        self.num_april_tags = 4
        self.calibration_done = False
        self.calibration_frames = []
        self.max_calibration_frames = 150    # Average over 150 frames (5 seconds)
        self.state = State.CALIBRATING

    def average_transforms(self, T_list):
        """
        Average a list of 4x4 transforms (base->camera).

        Args:
            T_list (list of np.ndarray): Each element is a 4x4 transform matrix to average.

        Returns:
            T_avg (np.ndarray): 4x4 averaged transform.
        """
        if not T_list:
            return None

        # Average the translations:
        translations = np.array([T[:3, 3] for T in T_list])
        avg_translation = translations.mean(axis=0)

        # Average the rotation:
        rot_mats = np.array([T[:3, :3] for T in T_list])
        # #################### Begin_Citation [NK2] ###################
        # Compute rotation average via SVD:
        M = rot_mats.sum(axis=0)
        U, _, Vt = np.linalg.svd(M)
        R_avg = U @ Vt
        # Fix possible reflection
        if np.linalg.det(R_avg) < 0:
            U[:, -1] *= -1
            R_avg = U @ Vt
        # ################### End_Citation [NK2] #######################

        # Construct Transform Matrix of average:
        T_base_camera_avg = np.eye(4)
        T_base_camera_avg[:3, :3] = R_avg
        T_base_camera_avg[:3, 3] = avg_translation

        return T_base_camera_avg

    def invert_tf(self, T):
        """Inverse of 4x4 homogeneous transform (Calibration Helper Function)."""
        R_ = T[:3, :3]
        t = T[:3, 3]
        Tinv = np.eye(4)
        Tinv[:3, :3] = R_.T
        Tinv[:3, 3] = -R_.T @ t
        return Tinv

    def calibrateCamera_Aruco(self, num_tags):
        """
        Calculate an average of base to camera_link transforms detected in a frame from April tags.
        """
        # Get the position of the camera to marker, then invert:
        base_camera_tf = {}     # Initialize a dictionary to hold transforms from base to camera.
        optical_tag_tf = {}     # Initialize the transfrom of camera to tag
        tag_optical_tf = {}     # Initialize the transform of the tag to camera.

        for i in range(num_tags):
            # Listen and store the tf of base_marker seen by camera:
            try:
                tf_msg = self.buffer.lookup_transform(
                    'camera_color_optical_frame', f'tag_{i}', self.get_clock().now().to_msg()
                )
                # Convert the transform message to a matrix and store.
                t = tf_msg.transform.translation
                q = tf_msg.transform.rotation
                Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                T_optical_tag = np.eye(4)
                T_optical_tag[:3, :3] = Rm
                T_optical_tag[:3, 3] = [t.x, t.y, t.z]
                optical_tag_tf[i] = T_optical_tag
                self.get_logger().info(f' Base to Camera for Frame: {optical_tag_tf[i]}')

                # Get the inverse of the camera to tag:
                tag_optical_tf[i] = self.invert_tf(optical_tag_tf[i])

                # Multiply and store transform of base to camera to later average:
                base_camera_tf[i] = (self.base_tag[i] @ tag_optical_tf[i] @ self.optical_link)
            except (tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException) as e:
                print(f'Transform for April tag_{i} not available: {e}')

        # Average all base to camera translations and rotations to get better calibration.
        T_list = list(base_camera_tf.values())
        T_base_camera_avg = self.average_transforms(T_list)
        return T_base_camera_avg

    def calibrate_target_publisher(self):
        """
        Calibrate the system at start up, and switch to publishing bug locations.

        This function calibrates the system at start-up, then publishes a bool that its complete.
        Afterwards, it switches responsibility to managing Rviz views and tf info.

        """
        if self.state == State.CALIBRATING:
            # Perform a static callibration at Launch:
            # Compute per-frame avearge base->camera transform:
            T_base_camera_frame = self.calibrateCamera_April(self.num_april_tags)
            if T_base_camera_frame is not None:
                self.calibration_frames.append(T_base_camera_frame)
                self.get_logger().info(f' Base to Camera for Frame: {T_base_camera_frame}')

            if len(self.calibration_frames) >= self.max_calibration_frames:
                # Average transforms
                avg_tf = self.average_transforms(self.calibration_frames)

                # Create the publication message:
                # Convert to quaternion for publication message ([x, y, z, w]):
                q = R.from_matrix(avg_tf[:3, :3]).as_quat()

                # Publish TF: base -> camera_link
                msg = TransformStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'base'
                msg.child_frame_id = 'camera_link'
                # Store Averaged Translation:
                msg.transform.translation.x = avg_tf[:3, 3][0]
                msg.transform.translation.y = avg_tf[:3, 3][1]
                msg.transform.translation.z = avg_tf[:3, 3][2]
                self.get_logger().info(f'{msg.transform.translation}')
                # Store Averaged Quaternion:
                msg.transform.rotation.x = q[0]
                msg.transform.rotation.y = q[1]
                msg.transform.rotation.z = q[2]
                msg.transform.rotation.w = q[3]
                self.get_logger().info(f'{msg.transform.rotation}')
                # Publish the Averaged base to camera_link transform.
                self.static_broadcaster.sendTransform(msg)
                self.calibration_done = True
                self.get_logger().info('Static calibration complete.')

                # Calibration is complete, so switch to Publishing Task:
                self.state = State.PUBLISHING

        elif self.state == State.PUBLISHING:
            pass


def main():
    rclpy.init()
    node = CalibrationNode()
    rclpy.spin(node)
    # Destroy all nodes and windows:
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
