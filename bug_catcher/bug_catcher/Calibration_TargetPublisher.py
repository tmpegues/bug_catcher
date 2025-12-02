"""
This node locates Aruco markers and publishes their ids and poses and calibrates camera to robot.

The node also publishes a target color based on interpolated visual identification at the switch
position, as well as calibration for determining robot base to world enclosure. It will work with
another color visual node which will identify and publish bug locations.

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
    #TODO: Add other parameters I have added.
"""

from bug_catcher_interfaces.msg import ArucoMarkers
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from geometry_msgs.msg import TransformStamped
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import tf2_ros
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_from_matrix


class CalibrationNode(rclpy.node.Node):

    def __init__(self):
        super().__init__('calibration_node')
        # ################################### Begin_Citation[NK1] ##############################
        # Declare and read camera parameters:
        self.declare_parameter(
            name='marker_size',
            value=0.0625,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Size of the markers in meters.',
            ),
        )
        self.declare_parameter(
            name='aruco_dictionary_id',
            value='DICT_5X5_250',
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Dictionary that was used to generate markers.',
            ),
        )
        self.declare_parameter(
            name='image_topic',
            value='/camera/camera/color/image_raw',
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Image topic to subscribe to.',
            ),
        )
        self.declare_parameter(
            name='camera_info_topic',
            value='/camera/camera/color/camera_info',
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Camera info topic to subscribe to.',
            ),
        )
        self.declare_parameter(
            name='camera_frame',
            value='',
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Camera optical frame to use.',
            ),
        )
        # Declare tag calibration parameters:
        self.declare_parameter('calibration.tags.tag_1.x', -0.1143)
        self.declare_parameter('calibration.tags.tag_1.y', -0.4572)
        self.declare_parameter('calibration.tags.tag_2.x', -0.1143)
        self.declare_parameter('calibration.tags.tag_2.y', 0.4064)
        self.declare_parameter('calibration.tags.tag_3.x', 0.6858)
        self.declare_parameter('calibration.tags.tag_3.y', 0.4064)
        self.declare_parameter('calibration.tags.tag_4.x', 0.6858)
        self.declare_parameter('calibration.tags.tag_4.y', -0.4572)

        # Read and set the values of each parameter:
        self.marker_size = (
            self.get_parameter('marker_size').get_parameter_value().double_value
        )
        dictionary_id_name = (
            self.get_parameter('aruco_dictionary_id').get_parameter_value().string_value
        )
        image_topic = (self.get_parameter('image_topic').get_parameter_value().string_value)
        info_topic = (
            self.get_parameter('camera_info_topic').get_parameter_value().string_value
        )
        self.camera_frame = (self.get_parameter('camera_frame').get_parameter_value().string_value)
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

        # Make sure we have a valid dictionary id for Aruco markers:
        try:
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            if type(dictionary_id) is not type(cv2.aruco.DICT_5X5_100):
                raise AttributeError
        except AttributeError:
            self.get_logger().error(
                'bad aruco_dictionary_id: {}'.format(dictionary_id_name)
            )
            options = '\n'.join([s for s in dir(cv2.aruco) if s.startswith('DICT')])
            self.get_logger().error('valid options: {}'.format(options))

        # SUBSCRIPTIONS:
        # Camera_info subscription
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self.camera_info_callback, qos_profile_sensor_data
        )
        # Camera_image subscription
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data
        )

        # PUBLISHERS:
        # Aruco marker publisher: (ID,Pose)
        self.markers_pub = self.create_publisher(ArucoMarkers, 'aruco_markers', 10)
        # Aruco marker positions publisher: (Pose)
        self.poses_pub = self.create_publisher(PoseArray, 'aruco_poses', 10)

        # Listeners:
        # The buffer stores received tf frames:
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        # Broadcasters:
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.dynamic_broadcaster = TransformBroadcaster(self)

        # Establish the transforms of Robot to Tags that we already know.
        #   These are established from environment set up (Designed to be constant/known)
        for marker_id, (x, y) in self.tag_params.items():
            static_tf = TransformStamped()
            static_tf.header.stamp = self.get_clock().now().to_msg()
            static_tf.header.frame_id = 'base'
            static_tf.child_frame_id = f'aruco_{marker_id}'
            static_tf.transform.translation.x = x
            static_tf.transform.translation.y = y
            static_tf.transform.translation.z = 0.025    # Z offset of base on table top.
            static_tf.transform.rotation.x = 0.0
            static_tf.transform.rotation.y = 0.0
            static_tf.transform.rotation.z = 0.0
            static_tf.transform.rotation.w = 1.0
            self.static_broadcaster.sendTransform(static_tf)

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

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.aruco_dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self.bridge = CvBridge()

        # Establish Calibration Averaging Variables:
        self.calibration_done = False
        self.calibration_frames = []
        self.max_calibration_frames = 150    # Average over 150 frames (5 seconds)

        # Translate Y-coordinate to match ROS REP-103 frame as OpenCV has a flipped y orientation.
        # X_ros =  Z_cv, Y_ros = -X_cv, Z_ros = -Y_cv
        self.Tcv_to_ros = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

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

    def calibrateCamera_Aruco(self, markers, num_markers):
        """
        Calculate an average of base to camera_link transforms detected in a frame.

        Args_
            markers (MarkerArray): The markers identified by the camera.
            num_markers (int):  The number of markers identified.

        """
        # Get the position of the camera to marker, then invert:
        marker_camera_tf = {}   # Initialize a dictionary to hold transforms from marker to camera.
        base_marker_tf = {}     # Initialize a dictionary to hold transforms from base to marker.
        base_camera_tf = {}     # Initialize the transfrom of base to camera.

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

            # Rotate OpenCv interpretation to ROS:
            R_ros = self.Tcv_to_ros[:3, :3] @  cam_rot
            t_ros = self.Tcv_to_ros[:3, :3] @ cam_trans
            # Invert to get Tmarker->camera:
            cam_mark_matrix = np.eye(4)
            cam_mark_matrix[:3, :3] = R_ros
            cam_mark_matrix[:3, 3:4] = t_ros
            mark_cam_matrix = self.invert_tf(cam_mark_matrix)
            # Add identified frame in Ros orientation:
            marker_camera_tf[markers.marker_ids[i]] = mark_cam_matrix

            # Listen and store the tf of base_marker seen by camera:
            try:
                tf_msg = self.buffer.lookup_transform(
                    'base', f'aruco_{markers.marker_ids[i]}', rclpy.time.Time()
                )
                # Convert the transform message to a matrix and store.
                t = tf_msg.transform.translation
                q = tf_msg.transform.rotation
                Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = Rm
                T[:3, 3] = [t.x, t.y, t.z]
                base_marker_tf[markers.marker_ids[i]] = T
            except (tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException) as e:
                print(f'Transform for marker {markers.marker_ids[i]} not available: {e}')

            # Multiply and store transform of base to camera:
            base_camera_tf[markers.marker_ids[i]] = (base_marker_tf[markers.marker_ids[i]]
                                                     @ marker_camera_tf[markers.marker_ids[i]])

        # Average all base to camera translations and rotations to get better calibration.
        T_list = list(base_camera_tf.values())
        T_base_camera_avg = self.average_transforms(T_list)
        return T_base_camera_avg

    def image_callback(self, img_msg):
        """
        Update each frame by publishing the position of the markers.

        This callback will publish an average transform location of the base->camera
        after 10 successful frames of marker identification.

        Args_
            img_msg (Image): The image frame that comes from the camera.
        """
        if self.info_msg is None:
            self.get_logger().warn('No camera info has been received!')
            return
        # Convert the image message to cv2:
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='mono8')

        # Establish the Marker and position array:
        markers = ArucoMarkers()
        pose_array = PoseArray()

        # Set the camera frame:
        if self.camera_frame == '':
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
        if marker_ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.intrinsic_mat, self.distortion)
            # Draw detected markers if needed for debugging: (uncomment if needed)
            # cv2.aruco.drawDetectedMarkers(cv_image, corners, marker_ids)
            # for i in range(len(marker_ids)):
            #     cv2.drawFrameAxes(cv_image,
            #                       self.intrinsic_mat,
            #                       self.distortion,
            #                       rvecs[i],
            #                       tvecs[i],
            #                       length=0.1
            #                       )

            # Set marker max to ignore false identifications
            max_marker_id = 5
            for i, marker_id in enumerate(marker_ids):
                if int(marker_id) > max_marker_id:
                    self.get_logger().warn(f'Ignoring unknown marker ID {marker_id}')
                    continue
                # Set the position of each marker:
                pose = Pose()
                pose.position.x = float(tvecs[i][0][0])
                pose.position.y = float(tvecs[i][0][1])
                pose.position.z = float(tvecs[i][0][2])

                rot_matrix = np.eye(4)
                rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                quat = quaternion_from_matrix(rot_matrix)
                # Set the orientation of each marker.
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

            # Perform a static callibration at the start and avearge off the first 10 frames.
            if not self.calibration_done:
                # Compute per-frame avearge base->camera transform:
                T_base_camera_frame = self.calibrateCamera_Aruco(markers, len(pose_array.poses))
                if T_base_camera_frame is not None:
                    self.calibration_frames.append(T_base_camera_frame)

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
                    # Now calibrated, so destroy subscription:
                    self.destroy_subscription(self.image_sub)


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
