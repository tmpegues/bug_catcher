"""
Calibration Node for AprilTag Camera Extrinsics and Rviz Marker Publication.

This node performs initial camera extrinsic calibration using multiple AprilTags fixed in known
positions on the robots base frame. Over several hundred frames, the node computes an averaged
base to camera transform and publishes it as a static TF. After calibration, the node switches
modes to publish bug detections as RViz markers and update the MoveIt PlanningScene
with the currently targeted bug. This keeps it active in the ROS system after calibration.

Subscriptions
-------------
/bugs : bug_catcher_interfaces/BugArray
    Incoming detections of all tracked bugs: bug_id, pose, color, and target designation.

Published Topics
----------------
visualization_marker_array : visualization_msgs/MarkerArray
    RViz markers for all non-target bugs.

planning_scene : moveit_msgs/PlanningScene
    Planning scene updates containing or removing the target bug as a collision object.

TF Frames
---------
Publishes:
    base to camera_link (static)
    Additional dynamic transforms if required by future extensions
        (If we decide to use frames for bugs)

Parameters
----------
calibration.tags.tag_<i>.x : float
calibration.tags.tag_<i>.y : float
    Known AprilTag positions in the base frame.

"""

import cv2
import numpy as np
import rclpy
import tf2_ros
from bug_catcher.planningscene import PlanningSceneClass, Obstacle
from bug_catcher_interfaces import BugArray
from enum import Enum, auto
from geometry_msgs.msg import TransformStamped
from moveit_msgs.msg import PlanningScene
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_matrix
from visualization_msgs.msg import Marker, MarkerArray


class State(Enum):
    """
    Current state of the system.

    Determines what the main timer function should be doing on each
        iteration for the physics of the brick.
    """

    INITIALIZING = auto()
    CALIBRATING = auto()
    PUBLISHING = auto()


class CalibrationNode(Node):
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
            1: (
                self.get_parameter('calibration.tags.tag_2.x').get_parameter_value().double_value,
                self.get_parameter('calibration.tags.tag_2.y').get_parameter_value().double_value,
            ),
            2: (
                self.get_parameter('calibration.tags.tag_3.x').get_parameter_value().double_value,
                self.get_parameter('calibration.tags.tag_3.y').get_parameter_value().double_value,
            ),
            3: (
                self.get_parameter('calibration.tags.tag_4.x').get_parameter_value().double_value,
                self.get_parameter('calibration.tags.tag_4.y').get_parameter_value().double_value,
            ),
            4: (
                self.get_parameter('calibration.tags.tag_1.x').get_parameter_value().double_value,
                self.get_parameter('calibration.tags.tag_1.y').get_parameter_value().double_value,
            ),
        }

        # SUBSCRIPTIONS:
        # Subscription for the bug tracking info:
        self.bug_sub = self.create_subscription(BugArray, '/bugs', self.bug_callback, 10)

        # PUBLISHERS:
        markerQoS = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.mark_pub = self.create_publisher(
            MarkerArray, 'visualization_marker_array', markerQoS
        )
        self.planscene = self.node.create_publisher(
            PlanningScene,
            '/planning_scene',
            10
        )

        # Timer:
        self.timer_update = self.create_timer(0.05, self.calibrate_target_publisher)

        # Listeners:
        # The buffer stores received tf frames:
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        # Broadcasters:
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.dynamic_broadcaster = TransformBroadcaster(self)

        # Create and save the matrix version of the base to marker trasforms for static recall:
        self.base_tag = {}
        for marker_id, (x, y) in self.tag_params.items():
            # Translation
            t = np.array([x, y, 0.025])  # Z offset
            # Rotation quaternion:
            q = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w      # TODO: Does this need flipped for ROS?
            # Convert quaternion to 4x4 rotation matrix
            mat = quaternion_matrix(q)
            # Set translation:
            mat[0:3, 3] = t
            # Store in dictionary for easy lookup:
            self.base_tag[marker_id] = mat

        # Establish Calibration Averaging Variables:
        self.num_april_tags = 4
        self.calibration_done = False
        self.calibration_frames = []
        self.max_calibration_frames = 300  # Average over 300 frames (10 seconds)
        self.state = State.INITIALIZING

        # Translate Y-coordinate to match ROS REP-103 frame as OpenCV has a flipped y orientation.
        # X_ros =  Y_cv, Y_ros = -X_cv, Z_ros = Z_cv
        self.Tcv_to_ros = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Establish the required connections and trackers for updating the planningscene each call.
        # Save the last target bug for removal each update:
        self.ps = PlanningSceneClass(self)
        self.last_target_bug = None

    def average_transforms(self, T_list):
        """
        Compute the average of a list of 4x4 homogeneous transformation matrices.

        This function computes the mean transform from a list of rigid-body
        transformations. Translations are averaged component-wise, and rotations
        are averaged using Singular Value Decomposition (SVD) to ensure a valid rotation matrix.

        Parameters
        ----------
        T_list : list of np.ndarray
            A list of 4x4 homogeneous transformation matrices to average. Each matrix represents 
            a rigid body transform.

        Returns
        -------
        T_avg : np.ndarray or None
            The 4x4 homogeneous transformation matrix representing the averaged transform. 
            Returns None if the input list is empty.
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
        T_avg = np.eye(4)
        T_avg[:3, :3] = R_avg
        T_avg[:3, 3] = avg_translation

        return T_avg

    def invert_tf(self, T):
        """
        Compute the inverse of a 4x4 homogeneous transformation matrix.

        Parameters
        ----------
        T : np.ndarray
            A 4x4 homogeneous transformation matrix representing a rigid body transform.

        Returns
        -------
        Tinv : np.ndarray
            The 4x4 homogeneous transformation matrix representing the inverse transform.
        """
        R_ = T[:3, :3]
        t = T[:3, 3]
        Tinv = np.eye(4)
        Tinv[:3, :3] = R_.T
        Tinv[:3, 3] = -R_.T @ t
        return Tinv

    def calibrateCamera_April(self, num_tags):
        """
        Compute the averaged base-to-camera transform using observed AprilTags.

        This function listens to the transforms between the camera's optical frame
        and AprilTags in the scene,and then averages all detected base-to-camera transforms over 
        the tags seen in the current frame.

        Parameters
        ----------
        num_tags : int
            The number of AprilTags to consider for calibration. Tags are assumed
            to be named sequentially as 'tag_1', 'tag_2', ..., 'tag_{num_tags}'.

        Returns
        -------
        np.ndarray or None
            A 4x4 numpy array representing the averaged homogeneous transform
            from the robot base to the camera_link frame. Returns None if no
            tags are successfully observed.
        """
        # Get the position of the camera to marker, then invert:
        base_camera_tf = {}  # Initialize a dictionary to hold transforms from base to camera.
        optical_tag_tf = {}  # Initialize the transfrom of camera to tag
        tag_optical_tf = {}  # Initialize the transform of the tag to camera.

        # Retrieve the Camera_Link to Camera_Optical Tf:

        for i in range(1, num_tags + 1):
            self.get_logger().info(f'number of tags: {num_tags}')
            # Listen and store the tf of base_marker seen by camera:
            try:
                tf_msg = self.buffer.lookup_transform(
                    'camera_color_optical_frame',
                    f'tag_{i}',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0),
                )
                # Convert the transform message to a matrix and store.
                t = tf_msg.transform.translation
                q = tf_msg.transform.rotation
                Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                T_optical_tag = np.eye(4)
                T_optical_tag[:3, :3] = Rm
                T_optical_tag[:3, 3] = [t.x, t.y, t.z]
                optical_tag_tf[i] = T_optical_tag

                # Get the inverse of the camera to tag:
                tag_optical_tf[i] = self.invert_tf(optical_tag_tf[i])

                # Multiply and store transform of base to camera to later average:
                if type(self.optical_link) is type(None):
                    base_camera_tf[i] = self.base_tag[i] @ optical_tag_tf[i]
                else:
                    base_camera_tf[i] = self.base_tag[i] @ optical_tag_tf[i] @ self.optical_link
                self.get_logger().info(
                    f'tag_{i} seen.\nTcamera,tag = {optical_tag_tf[i]}\nTtag,\
                    camera = {tag_optical_tf[i]}\nTbase_camera = {base_camera_tf[i]}'
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException,
            ) as e:
                self.get_logger().info(f'Transform for April tag_{i} not available: {e}')
        # Average all base to camera translations and rotations to get better calibration.
        T_list = list(base_camera_tf.values())
        T_base_camera_avg = self.average_transforms(T_list)
        return T_base_camera_avg

    def calibrate_target_publisher(self):
        """
        Manages system calibration and transitions to publishing bug locations in Rviz.

        This function is intended to run periodically in a timer callback and performs the
        following tasks based on it's state:

        1. INITIALIZING:
            - Waits for camera_link to camera_color_optical_frame transform to become available.
            - Converts the transform to a 4x4 matrix and applies the CV to ROS frame adjustment.
            - Stores the optical_link transform and transitions to CALIBRATING state.

        2. CALIBRATING:
            - Collects base to camera transforms using AprilTags observed in the current frame.
            - Accumulates transforms over multiple frames.
            - Once enough frames are collected, averages the transforms.
            - Publishes the averaged base to camera_link transform as a static TF.
            - Marks calibration as complete and transitions to PUBLISHING state.

        3. PUBLISHING:
            - Updates RViz markers for all tracked bugs updated in the subscriber callback.
            - Publishes markers to the visualization_marker_array topic.
        """
        match self.state:
            case State.INITIALIZING:
                try:
                    tf_msg = self.buffer.lookup_transform(
                        'camera_color_optical_frame',
                        'camera_link',
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=1.0),
                    )
                    # Convert the transform message to a matrix and store.
                    t = tf_msg.transform.translation
                    q = tf_msg.transform.rotation
                    Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                    # Save this transform statically in the node:
                    optical_link = np.eye(4)
                    optical_link[:3, :3] = Rm
                    optical_link[:3, 3] = [t.x, t.y, t.z]

                    # Transform the Camera_Link to Ros:
                    optical_link = self.Tcv_to_ros @ optical_link
                    self.optical_link = optical_link
                    self.state = State.CALIBRATING
                    self.get_logger().info('Camera tf available')
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException,
                ) as e:
                    self.get_logger().info(f'Transform for camera still unavailable: {e}')
            case State.CALIBRATING:
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
            case State.PUBLISHING:
                pass
                # Update Rviz markers for all colored bugs:
                self.marker_array.markers = self.markers
                self.mark_pub.publish(self.marker_array)

    def bug_callback(self, bug_msg):
        """
        Callback for updating detected bug positions and the planning scene.

        Parameters
        ----------
        bug_msg :
            A list/array of bug detection messages. Each bug contains:
            - id: int8
                The unique identifier for that colored bug.
            - is_target : bool
                True if the bug should be treated as the active collision target.
            - pose : geometry_msgs/PoseStamped (or similar)
                The estimated pose of the bug in the camera/base frame.
            - color : str
                The bug's color label (e.g., 'red', 'blue', ...).
        """
        # Remove the last target bug if it exists:
        if self.last_target_bug is None:
            pass
        else:
            self.ps.remove_obstacle(self.last_target_bug)
        # Create an array of markers to build the arena:
        self.marker_array = MarkerArray()
        self.markers = []

        # Break apart each bug message: is_target: 1-> CollisionObject | 0-> Marker
        for i, bug in enumerate(bug_msg):
            # Check if the bug is the target or not:
            if bug.is_target is True:
                # Set the bug as a collision object and republish planning scene.
                size = {'type': 1, 'dimensions': [0.01, 0.01, 0.01]}  # Set to a box for all bugs.
                current_target_bug = Obstacle('target', bug.pose.pose, size)
                self.ps.add_obstacle(current_target_bug)
                self.last_target_bug = current_target_bug
            else:
                # The bug is not a current target, just track it as a colored marker and publish.
                marker = Marker()
                marker.header.frame_id = 'base'
                marker.header.stamp = bug.pose.stamp        # The bug gets a time stamp.
                marker.ns = 'bug_markers'
                marker.id = i           # Assumes that the ColorDetect sets a unique number.
                marker.type = Marker.CUBE
                marker.action = Marker.ADD

                # Set the location of the bug:
                marker.pose.position.x = bug.pose.x
                marker.pose.position.y = bug.pose.y
                marker.pose.position.z = bug.pose.z
                marker.pose.orientation.x = bug.pose.orientation.x
                marker.pose.orientation.y = bug.pose.orientation.y
                marker.pose.orientation.z = bug.pose.orientation.z
                marker.pose.orientation.w = bug.pose.orientation.w
                marker.scale.x = 0.01
                marker.scale.y = 0.01
                marker.scale.z = 0.01
                marker.lifetime.sec = 0.02
                marker.lifetime.nanosec = 0

                if bug.color == 'red':
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                elif bug.color == 'green':
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                elif bug.color == 'blue':
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                    marker.color.a = 1.0
                elif bug.color == 'orange':
                    marker.color.r = 1.0
                    marker.color.g = 0.5
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                elif bug.color == 'purple':
                    marker.color.r = 0.5
                    marker.color.g = 0.0
                    marker.color.b = 0.5
                    marker.color.a = 1.0
                self.markers.append(marker)


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
