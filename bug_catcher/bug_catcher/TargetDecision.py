"""
The script implements the 'target_decision_node' within the 'bug_catcher' package.

It acts as the central vision system processing two camera streams:
1. Sky Cam ("bug_god"): Detects ALL bugs, transforms them to the Base Frame,
   identifies the target based on color and proximity to the gripper, and publishes
   a BugArray containing all seen bugs.
2. Wrist Cam: Focuses purely on the specific target color for precise servoing
   and publishes a single target BugInfo.

Publishers
----------
+ /bug_god/bug_array (bug_catcher_interfaces.msg.BugArray): All detected bugs in Base Frame.
+ /bug_god/debug_view (sensor_msgs.msg.Image): Sky cam visualization with all bugs.
+ /bug_god/mask_view (sensor_msgs.msg.Image): Sky cam binary mask.
Subscribers
-----------
+ /camera/bug_god/color/image_raw (sensor_msgs.msg.Image): Sky Cam Feed.
+ /target_color (std_msgs.msg.String): Receives commands to switch the detection target.

Parameters
----------
+ default_color (string, default='blue'): The initial color to detect upon startup.
+ base_frame (string, default='base'): The robot's root frame ID.
+ gripper_frame (string, default='fer_hand_tcp'): The end-effector frame ID.
calibration.tags.tag_<i>.x : float
calibration.tags.tag_<i>.y : float
    Known AprilTag positions in the base frame.

"""

import os
from enum import Enum, auto

from ament_index_python.packages import get_package_share_directory

# from bug_catcher.sort import Sort
from bug_catcher.vision import Vision

from bug_catcher_interfaces.msg import BasePose, BasePoseArray, BugArray, BugInfo
# from bug_catcher_interfaces.srv import Sort

import cv2

from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import CameraInfo, Image

from std_msgs.msg import String

import tf2_ros
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener

from tf_transformations import quaternion_matrix


class State(Enum):
    """
    Current state of the system.

    Determines what the main timer function should be doing on each
        iteration for the physics of the brick.
    """

    INITIALIZING = auto()
    CALIBRATING_SKYCAM = auto()
    CALIBRATING_DROPS = auto()
    PUBLISHING = auto()


class TargetDecision(Node):
    """
    Manages camera calibration and bug visualization for the bug catcher robot.

    This node first calibrates the camera's extrinsic parameters by observing AprilTags
    with known positions in the robot's base frame. It averages the computed transforms
    over several frames and publishes the result as a static transform.

    After calibration, the node transitions to a publishing mode. In this mode, it
    subscribes to bug detection data, visualizes non-target bugs as markers in RViz,
    and updates the MoveIt planning scene with the current target bug as a
    collision object.
    """

    def __init__(self):
        """Initialize the TargetDecision node."""
        super().__init__('target_decision_node')

        # ==================================
        # 1. Parameters & Setup
        # ==================================
        # Declare the config parameters:
        self.declare_parameter('default_color', 'blue')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('gripper_frame', 'fer_hand_tcp')
        self.declare_parameter('color_path_sky_cam', '')
        self.declare_parameter('pad_start', 5)
        self.declare_parameter('pad_end', 10)

        # Declare tag calibration parameters:
        self.declare_parameter('calibration.tags.tag_1.x', -0.1143)
        self.declare_parameter('calibration.tags.tag_1.y', -0.4572)
        self.declare_parameter('calibration.tags.tag_2.x', -0.1143)
        self.declare_parameter('calibration.tags.tag_2.y', 0.4064)
        self.declare_parameter('calibration.tags.tag_3.x', 0.6858)
        self.declare_parameter('calibration.tags.tag_3.y', 0.4064)
        self.declare_parameter('calibration.tags.tag_4.x', 0.6858)
        self.declare_parameter('calibration.tags.tag_4.y', -0.4572)
        # Set the color switch location (w.r.t base):
        self.declare_parameter('color_switch_x', 0.76835)
        self.declare_parameter('color_switch_y', 0.00000)
        self.declare_parameter('color_switch_z', 0.1524)

        # Set the tag and config parameter values:
        self.target_color = self.get_parameter('default_color').value
        self.base_frame = self.get_parameter('base_frame').value
        self.gripper_frame = self.get_parameter('gripper_frame').value
        self.color_path_sky_cam = self.get_parameter('color_path_sky_cam').value
        self.pad_start = self.get_parameter('pad_start').get_parameter_value().integer_value
        self.pad_end = self.get_parameter('pad_end').get_parameter_value().integer_value
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
        self.color_switch_x = (
            self.get_parameter('color_switch_x').get_parameter_value().double_value
        )
        self.color_switch_y = (
            self.get_parameter('color_switch_y').get_parameter_value().double_value
        )
        self.color_switch_z = (
            self.get_parameter('color_switch_z').get_parameter_value().double_value
        )

        # TF Buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Vision Setup
        self.sky_cam_vision = Vision()
        self.bridge = CvBridge()
        self._load_calibrated_colors()

        self.sky_intrinsics = None

        # ==================================
        # 2. Sky Cam (bug_god) Setup
        # ==================================
        # PUBLISHERS:
        self.bug_array_pub = self.create_publisher(BugArray, '/bug_god/bug_array', 10)
        self.sky_debug_pub = self.create_publisher(Image, '/bug_god/debug_view', 10)
        self.sky_mask_pub = self.create_publisher(Image, '/bug_god/mask_view', 10)
        self.target_switch_pub = self.create_publisher(String, '/target_color', 10)
        self.switch_debug_pub = self.create_publisher(Image, '/bug_god/switch_view', 10)
        # Create a publisher for the drop locations:
        self.drop_pub = self.create_publisher(BasePoseArray, 'drop_locs', 10)

        # SUBSCRIBERS:
        self.service_target_sub = self.create_subscription(String,
                                                           '/service/target_color',
                                                           self.service_switch_callback,
                                                           10)

        # ==================================
        # 5. Initial System Integration Setup:
        # ==================================
        # Timer:
        self.timer_update = self.create_timer(0.05, self.calibrate_target_publisher)

        # Broadcasters:
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.dynamic_broadcaster = TransformBroadcaster(self)

        # Create and save the matrix version of the base to marker trasforms for static recall:
        self.base_tag = {}
        for marker_id, (x, y) in self.tag_params.items():
            # Translation
            t = np.array([x, y, 0.0762])  # Z offset
            # Rotation quaternion:
            q = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w
            # Convert quaternion to 4x4 rotation matrix
            mat = quaternion_matrix(q)
            # Set translation:
            mat[0:3, 3] = t
            # Store in dictionary for easy lookup:
            self.base_tag[marker_id] = mat

        # Translate Y-coordinate to match ROS REP-103 frame as OpenCV has a flipped y orientation.
        # X_ros =  Y_cv, Y_ros = -X_cv, Z_ros = Z_cv
        self.Tcv_to_ros = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.Tros_to_cv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Establish Calibration Averaging Variables:
        self.num_april_tags = 4
        self.calibration_done = False
        self.calibration_frames = []
        self.max_calibration_frames = 300  # Average over 300 frames (10 seconds)
        self.state = State.INITIALIZING

        # Set the switch mask to remove detection of color task switcher:
        self.switch_mask = None

        # Store the latest sky target:
        self.latest_sky_target = None
        self.last_sky_update_time = self.get_clock().now()

        self.get_logger().info(f'Node started. Current Target: [{self.target_color}]')

    # -----------------------------------------------------------------
    # Helper Functions
    # -----------------------------------------------------------------
    def _load_calibrated_colors(self):
        """Load calibrated colors from YAML file."""
        pkg_share = get_package_share_directory('bug_catcher')
        default_path = os.path.join(pkg_share, 'config', 'sky_cam_colors.yaml')
        # --- Load Sky Cam ---
        if not self.color_path_sky_cam:
            self.color_path_sky_cam = default_path
            self.get_logger().info(f'No sky path param. Using default: {default_path}')

        try:
            self.get_logger().info(f'Loading Sky Calibration: {self.color_path_sky_cam}')
            self.sky_cam_vision.load_calibration(self.color_path_sky_cam)
        except (IOError, FileNotFoundError) as e:
            self.get_logger().error(f'Failed to load Sky calibration: {e}')

        # Verify Target Color:
        if self.target_color not in self.sky_cam_vision.colors:
            self.get_logger().warn(f"Target '{self.target_color}' missing in Sky config!")

    def _pixel_to_pose(self, u, v, intrinsics, height):
        """
        Project a 2D pixel coordinate into 3D space using the Pinhole Camera Model.

        Assumes the object lies on a plane at a fixed distance 'height' from the camera.

        Args:
        ----
        u (int): Pixel x-coordinate.
        v (int): Pixel y-coordinate.
        intrinsics (np.ndarray): 3x3 Camera Intrinsic Matrix (K).
        height (float): The distance along the Z-axis from camera to object.

        Returns
        -------
        pose (geometry_msgs.msg.Pose): The computed 3D pose in the camera frame.

        """
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        X = (u - cx) * height / fx
        Y = (v - cy) * height / fy
        Z = height

        pose = Pose()
        pose.position.x = float(X)
        pose.position.y = float(Y)
        pose.position.z = float(Z)
        pose.orientation.w = 1.0

        return pose

    def _create_pose_stamped(self, pose_data, timestamp):
        """
        Create a PoseStamped message from raw pose data and timestamp.

        Args:
        ----
        pose_data (geometry_msgs.msg.Pose): The pose data.
        timestamp (builtin_interfaces.msg.Time): The timestamp for the header.

        Returns
        -------
        pose_stamped_msg (geometry_msgs.msg.PoseStamped): The constructed PoseStamped message.

        """
        pose_stamped_msg = PoseStamped()

        # 1. Header (Explicitly cast frame_id to string)
        pose_stamped_msg.header.frame_id = str(self.base_frame)
        pose_stamped_msg.header.stamp = timestamp
        # 2. Pose (Explicitly cast everything to float)
        pose_stamped_msg.pose.position.x = float(pose_data.position.x)
        pose_stamped_msg.pose.position.y = float(pose_data.position.y)
        pose_stamped_msg.pose.position.z = float(pose_data.position.z)

        pose_stamped_msg.pose.orientation.x = float(pose_data.orientation.x)
        pose_stamped_msg.pose.orientation.y = float(pose_data.orientation.y)
        pose_stamped_msg.pose.orientation.z = float(pose_data.orientation.z)
        pose_stamped_msg.pose.orientation.w = float(pose_data.orientation.w)

        return pose_stamped_msg

    def get_cam_height_and_transform(self, camera_frame):
        """
        Calculate camera height relative to base and return the transform object.

        Args:
        ----
        camera_frame (str): The frame ID of the camera optical frame.

        Returns
        -------
        cam_height (float): The Z-height of the camera relative to base.
        cam_tf (TransformStamped): The transform object from base to camera.

        """
        try:
            # Check if transform exists first to avoid spamming errors
            if not self.tf_buffer.can_transform(self.base_frame, camera_frame, rclpy.time.Time()):
                return None, None

            cam_tf = self.tf_buffer.lookup_transform(
                self.base_frame, camera_frame, rclpy.time.Time(seconds=0)
            )
            cam_height = cam_tf.transform.translation.z
            return cam_height, cam_tf
        except tf2_ros.TransformException:
            # Silent failure (normal during startup)
            return None, None

    def apply_transform(self, pose, transform):
        """
        Apply a transform (Rotation + Translation) to a Pose.

        This method performs manual matrix multiplication to apply a TF transform
        to a geometry_msgs/Pose. It is used to bypass potential type-check errors
        in the `tf2_geometry_msgs.do_transform_pose` library function.

        Args:
        ----
        pose (geometry_msgs.msg.Pose): The input pose in the source frame.
        transform (geometry_msgs.msg.TransformStamped): The transform to apply.

        Returns
        -------
        (geometry_msgs.msg.Pose): The transformed pose in the target frame.

        """
        # 1. Extract Translation
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        tz = transform.transform.translation.z

        # 2. Extract Rotation (Quaternion)
        rx = transform.transform.rotation.x
        ry = transform.transform.rotation.y
        rz = transform.transform.rotation.z
        rw = transform.transform.rotation.w

        # 3. Input Point
        px = pose.position.x
        py = pose.position.y
        pz = pose.position.z

        # 4. Quaternion Rotation Formula: v' = q * v * q_conjugate
        # Calculate q * v
        ix = rw * px + ry * pz - rz * py
        iy = rw * py + rz * px - rx * pz
        iz = rw * pz + rx * py - ry * px
        iw = -rx * px - ry * py - rz * pz

        # Calculate result * q_conjugate
        # q_conjugate = [-rx, -ry, -rz, rw]
        x_rot = ix * rw + iw * -rx + iy * -rz - iz * -ry
        y_rot = iy * rw + iw * -ry + iz * -rx - ix * -rz
        z_rot = iz * rw + iw * -rz + ix * -ry - iy * -rx

        # 5. Apply Translation
        new_pose = Pose()
        new_pose.position.x = float(x_rot + tx)
        new_pose.position.y = float(y_rot + ty)
        new_pose.position.z = float(z_rot + tz)

        # 6. Rotation (Keep original orientation for point bugs)
        new_pose.orientation = pose.orientation

        return new_pose

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

        # Loop through tags 1 to 4 for camera calibration:
        for i in range(1, num_tags + 1):
            # Listen and store the tf of base_marker seen by camera:
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    'bug_god_color_optical_frame',
                    f'tag_{i}',
                    rclpy.time.Time(),
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

    # -----------------------------------------------------------------
    # Timer Callback for state system:
    # -----------------------------------------------------------------
    def calibrate_target_publisher(self):
        """
        Manage system calibration and transitions to publishing bug locations in Rviz.

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
                    tf_msg = self.tf_buffer.lookup_transform(
                        'bug_god_color_optical_frame',
                        'bug_god_link',
                        rclpy.time.Time(),
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
                    self.state = State.CALIBRATING_SKYCAM
                    self.get_logger().info('Camera tf available')
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException,
                ) as e:
                    self.get_logger().info(f'Transform for camera still unavailable: {e}')
            case State.CALIBRATING_SKYCAM:
                # Perform a static callibration at Launch:
                # Compute per-frame avearge base->camera transform:
                T_base_camera_frame = self.calibrateCamera_April(self.num_april_tags)
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
                    msg.child_frame_id = 'bug_god_link'
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
                    self.state = State.CALIBRATING_DROPS

                    # Create the subscribers to start listening for bug detection:
                    sky_cb_group = MutuallyExclusiveCallbackGroup()
                    self.sky_image_sub = self.create_subscription(
                        Image,
                        '/camera/bug_god/color/image_raw',
                        self.sky_image_cb,
                        10,
                        callback_group=sky_cb_group,
                    )
                    self.sky_info_sub = self.create_subscription(
                        CameraInfo,
                        '/camera/bug_god/color/camera_info',
                        self.sky_info_cb,
                        10,
                        callback_group=sky_cb_group,
                    )

            case State.PUBLISHING:
                pass
                # All publishing will take place in the image callbacks

    def sky_info_cb(self, msg):
        """Store Sky Camera intrinsics."""
        if self.sky_intrinsics is None:
            self.sky_intrinsics = np.array(msg.k).reshape(3, 3)
            self.get_logger().info('Sky Camera (Bug God) Intrinsics Received.')

    def sky_image_cb(self, msg):
        """
        Process Sky Camera image feed.

        Detects all colors, builds a BugArray of all bugs in the Base Frame,
        calculates distance to the gripper, and marks the closest matching
        bug as the target.

        Args:
        ----
        msg (sensor_msgs.msg.Image): The raw image from Sky Cam.

        """
        if self.sky_intrinsics is None:
            return

        match self.state:
            case State.CALIBRATING_DROPS:
                # Loop through Tags 5-10 and set the location of the drop pads:
                # Establish the locations for the drop off pads:
                self.drop_locs = {}
                self.get_logger().info(f'pad_start: {self.pad_start}')
                for i in range(self.pad_start, self.pad_end + 1):
                    # Listen and store the tf of base_marker seen by camera:
                    try:
                        tf_msg = self.tf_buffer.lookup_transform(
                            'base',
                            f'tag_{i}',
                            rclpy.time.Time(),
                            timeout=rclpy.duration.Duration(seconds=1.0),
                        )
                        # Convert the transform message to a matrix and store.
                        pose = Pose()
                        pose.position.x = float(tf_msg.transform.translation.x)
                        pose.position.y = float(tf_msg.transform.translation.y)
                        pose.position.z = float(tf_msg.transform.translation.z)
                        pose.orientation.w = 1.0
                        self.get_logger().info(f'i: {i}')
                        self.get_logger().info(f'pad_start: {self.pad_start}')
                        self.get_logger().info(f' pose is {pose}')
                        # Set the color string of the id:
                        if i == self.pad_start:
                            self.drop_locs['pink'] = pose
                        elif i == self.pad_start + 1:
                            self.drop_locs['green'] = pose
                        elif i == self.pad_start + 2:
                            self.drop_locs['blue'] = pose
                        elif i == self.pad_start + 3:
                            self.drop_locs['orange'] = pose
                        elif i == self.pad_start + 4:
                            self.drop_locs['purple'] = pose
                        elif i == self.pad_start + 5:
                            self.drop_locs['yellow'] = pose
                    except (
                        tf2_ros.LookupException,
                        tf2_ros.ExtrapolationException,
                        tf2_ros.ConnectivityException,
                    ) as e:
                        self.get_logger().info(f'Unable to find drop location for tag_{i}: {e}')
                # Publish the locations of the drop points:
                base_poses = BasePoseArray()
                for color, pose in self.drop_locs.items():
                    pair = BasePose()
                    pair.color = color
                    pair.pose = pose
                    base_poses.base_poses.append(pair)
                self.drop_pub.publish(base_poses)
                # Switch to the Publication state:
                self.state = State.PUBLISHING
            case State.PUBLISHING:
                try:
                    frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    mask = np.zeros(frame.shape[:2], dtype='uint8')
                    # Mask image to crop out unwanted filtering regions. (Tags will be established)
                    # ####################### Begin_Citation [NK3] ###################
                    world_mask = np.zeros(frame.shape[:2], dtype='uint8')

                    # Get the pixel locations of the top left/bottom right of the mask positions
                    #   This will map to markers 2 and 4.
                    tf_msg = self.tf_buffer.lookup_transform(
                        'bug_god_color_optical_frame',
                        'tag_2',
                        rclpy.time.Time(),
                    )
                    # Convert the transform message to a matrix and store.
                    t_left = tf_msg.transform.translation
                    tf_msg = self.tf_buffer.lookup_transform(
                        'bug_god_color_optical_frame',
                        'tag_4',
                        rclpy.time.Time(),
                    )
                    # Convert the transform message to a matrix and store.
                    t_right = tf_msg.transform.translation

                    # Calculate the 3D Position:
                    # Camera intrinsics matrix
                    K = self.sky_intrinsics
                    fx = K[0, 0]
                    fy = K[1, 1]
                    cx = K[0, 2]
                    cy = K[1, 2]
                    # 3D point in camera frame (meters)
                    X_L, Y_L, Z_L = t_left.x, t_left.y, t_left.z
                    X_R, Y_R, Z_R = t_right.x, t_right.y, t_right.z
                    # Left Tag location:
                    u_L = int(fx * (X_L / Z_L) + cx) - 60
                    v_L = int(fy * (Y_L / Z_L) + cy) - 60
                    # Right Tag Location:
                    u_R = int(fx * (X_R / Z_R) + cx) + 60
                    v_R = int(fy * (Y_R / Z_R) + cy) + 60

                    # Draw the mask to be within the tags:
                    cv2.rectangle(world_mask, (u_L, v_L), (u_R, v_R), 255, -1)

                    frame = cv2.bitwise_and(frame, frame, mask=world_mask)
                    # if switch mask is not none, apply it:
                    if self.switch_mask is not None:
                        inverted_mask = cv2.bitwise_not(self.switch_mask)
                        frame = cv2.bitwise_and(frame, frame, mask=inverted_mask)
                    # ####################### End_Citation [NK3] #####################
                except CvBridgeError as e:
                    self.get_logger().error(f'CV Bridge Error: {e}, unable to extract corners!')
                    return

                # 1. Get Transform Once per Frame
                cam_frame_id = msg.header.frame_id
                cam_height, transform_stamped = self.get_cam_height_and_transform(cam_frame_id)

                if cam_height is None:
                    cam_height = 1.0  # Fallback height if TF is not ready

                # 2. Get Gripper Position
                gripper_pos_base = None

                try:
                    gripper_tf = self.tf_buffer.lookup_transform(
                        self.base_frame, self.gripper_frame, rclpy.time.Time(seconds=0)
                    )
                    gripper_pos_base = gripper_tf.transform.translation
                except tf2_ros.TransformException:
                    pass

                bug_array = BugArray()
                bug_array.header.stamp = msg.header.stamp
                bug_array.header.frame_id = self.base_frame

                final_debug_frame = frame.copy()
                closest_dist = float('inf')
                target_bug_index = -1
                all_detected_bugs = []

                # 3. Iterate colors
                for color_name in self.sky_cam_vision.colors.keys():
                    detections, _, _ = self.sky_cam_vision.detect_objects(frame, color_name)
                    results = []
                    for i, det in enumerate(detections):
                        x, y, w, h = cv2.boundingRect(det)
                        u = int(x + w / 2)
                        v = int(y + h / 2)

                        cv2.drawContours(final_debug_frame, [det], -1, (0, 255, 0), 2)

                        cv2.putText(
                            final_debug_frame,
                            f'{color_name} {i}',
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                        obj_id = i
                        results.append((obj_id, u, v))

                    for obj_id, u, v in results:
                        # Pixel -> Camera 3D
                        pose_cam = self._pixel_to_pose(u, v, self.sky_intrinsics, cam_height)

                        # Camera 3D -> Base 3D
                        if transform_stamped:
                            try:
                                pose_base = self.apply_transform(pose_cam, transform_stamped)
                            except (AttributeError, TypeError) as e:
                                self.get_logger().warn(f'Failed to apply transform: {e}')
                                continue
                        else:
                            continue

                        # Build PoseStamped
                        try:
                            correct_pose_stamped = self._create_pose_stamped(
                                pose_base, msg.header.stamp
                            )
                        except (AttributeError, TypeError) as e:
                            self.get_logger().error(f'Error creating safe pose: {e}')
                            continue

                        # Build BugInfo
                        bug_info = BugInfo()
                        bug_info.id = int(obj_id)
                        bug_info.color = color_name
                        bug_info.pose = correct_pose_stamped
                        bug_info.target = False

                        cv2.putText(
                            final_debug_frame,
                            f'{color_name}',
                            (u, v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                        # Determine Closest Target Bug Logic can be improved:
                        # Set the target bug here:
                        if color_name == self.target_color and gripper_pos_base:
                            dx = pose_base.position.x - gripper_pos_base.x
                            dy = pose_base.position.y - gripper_pos_base.y
                            dist = dx**2 + dy**2

                            if dist < closest_dist:
                                closest_dist = dist
                                target_bug_index = len(all_detected_bugs)
                                bug_info.target = True

                        all_detected_bugs.append(bug_info)

                # 4. Mark Target Bug
                if target_bug_index != -1:
                    all_detected_bugs[target_bug_index].target = True
                    cv2.putText(
                        final_debug_frame,
                        f'TARGET: {self.target_color}',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                bug_array.bugs = all_detected_bugs
                self.bug_array_pub.publish(bug_array)
                self.sky_debug_pub.publish(
                    self.bridge.cv2_to_imgmsg(final_debug_frame, encoding='bgr8')
                )

                if mask is not None:
                    self.sky_mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding='mono8'))

                # Create a mask of the image at the color_switch point to change the color when
                # the block has been changed. (Pre_existing mask exists on image)
                switch_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                # save the switch mask to actively apply it to the color detection masks.
                #   This will ensure we don't detect the color block as a bug.
                self.switch_mask = np.zeros(switch_frame.shape[:2], dtype='uint8')
                # Set the current location in base:
                vec = np.array(
                    [self.color_switch_x, self.color_switch_y, self.color_switch_z, 1.0]
                )

                # Transform this to the camera frame:
                try:
                    tf_msg = self.tf_buffer.lookup_transform(
                        'bug_god_color_optical_frame',
                        'bug_god_link',
                        rclpy.time.Time(),
                    )
                    # Convert the transform message to a matrix and store.
                    t = tf_msg.transform.translation
                    q = tf_msg.transform.rotation
                    Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                    # Save this transform statically in the node:
                    transform = np.eye(4)
                    transform[:3, :3] = Rm
                    transform[:3, 3] = [t.x, t.y, t.z]
                    vec_cam_ros = transform @ vec
                    vec_cam_cv = vec_cam_ros @ self.Tros_to_cv
                    x, y, z, _ = vec_cam_cv
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException,
                ) as e:
                    self.get_logger().info(f'Transform  lookup failed: {e}')

                # 3D point in camera frame (meters)
                X_L, Y_L, Z_L = x, y, z
                X_R, Y_R, Z_R = x, y, z
                # Left Tag location:
                u_L = int(fx * (X_L / Z_L) + cx) - 40
                v_L = int(fy * (Y_L / Z_L) + cy) - 40
                # Right Tag Location:
                u_R = int(fx * (X_R / Z_R) + cx) + 40
                v_R = int(fy * (Y_R / Z_R) + cy) + 40

                # Draw the mask to be within the tags:
                cv2.rectangle(self.switch_mask, (u_L, v_L), (u_R, v_R), 255, -1)
                switch_frame = cv2.bitwise_and(switch_frame, switch_frame, mask=self.switch_mask)

                # Loop through colors and if it returns a position, then switch the target.
                switch_debug_frame = switch_frame.copy()
                for color_name in self.sky_cam_vision.colors.keys():
                    detections, _, _ = self.sky_cam_vision.detect_objects(
                        switch_debug_frame, color_name
                    )
                    # If a color is detected publish that color to the topic:
                    if len(detections) != 0:
                        self.target_switch_pub.publish(String(data=color_name))
                        self.target_color = color_name

                self.switch_debug_pub.publish(
                    self.bridge.cv2_to_imgmsg(switch_debug_frame, encoding='bgr8')
                )

    def service_switch_callback(self, switch_msg):
        """Color Switch Subscription for User Service to start Sort with Different Choice."""
        self.target_color = switch_msg.data


def main(args=None):
    """Run the main function for the ColorDetection node."""
    rclpy.init()
    node = TargetDecision()
    rclpy.spin(node)
    # Destroy all nodes and windows:
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
