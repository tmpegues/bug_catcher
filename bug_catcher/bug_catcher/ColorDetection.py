"""
The script implements the 'color_detection_node' within the 'bug_catcher' package.

It acts as the central vision system processing two camera streams:
1. Sky Cam ("BugGod"): Detects ALL bugs, transforms them to Base Frame,
   identifies the target based on color and proximity to gripper, and publishes a BugArray.
2. Wrist Cam: Focuses on the specific target color for precise servoing.

Publishers
----------
+ /bug_god/bug_array (bug_catcher_interfaces.msg.BugArray): All detected bugs in Base Frame.
+ /bug_god/debug_view (sensor_msgs.msg.Image): Sky cam visualization with all bugs.
+ /bug_god/mask_view (sensor_msgs.msg.Image): Sky cam binary mask.
+ /wrist_camera/bug_poses (geometry_msgs.msg.PoseArray): Targets seen by wrist cam in Base Frame.
+ /wrist_camera/debug_view (sensor_msgs.msg.Image): Wrist cam visualization.
+ /wrist_camera/mask_view (sensor_msgs.msg.Image): Wrist cam binary mask.

Subscribers
-----------
+ /camera/buggod/color/image_raw: Sky Cam Feed.
+ /camera/wrist_camera/color/image_raw: Wrist Cam Feed.
+ /target_color (std_msgs.msg.String): Receives commands to switch the detection target.

Parameters
----------
+ default_color (string, default='red'): The initial color to detect upon startup.

"""

import os

from ament_index_python.packages import get_package_share_directory

from bug_catcher.sort import Sort
from bug_catcher.vision import Vision

from bug_catcher_interfaces.msg import BugArray, BugInfo

import cv2

from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Pose, PoseArray, PoseStamped

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, Image

from std_msgs.msg import String

from tf2_geometry_msgs import do_transform_pose

import tf2_ros


class ColorDetection(Node):
    """The Color Detection Node."""

    def __init__(self):
        """Initialize the ColorDetection node."""
        super().__init__('color_detection_node')

        # ==================================
        # 1. Parameters & Setup
        # ==================================
        self.declare_parameter('default_color', 'red')
        self.target_color = self.get_parameter('default_color').value

        self.base_frame = 'base'

        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Vision System Setup
        self.vision = Vision()
        self.bridge = CvBridge()

        # Load Color Calibration
        pkg_share = get_package_share_directory('bug_catcher')
        yaml_path = os.path.join(pkg_share, 'config', 'calibrated_colors.yaml')

        if not os.path.exists(yaml_path):
            self.get_logger().warn(f'Config not found at {yaml_path}, using local file.')
            yaml_path = 'calibrated_colors.yaml'

        try:
            self.vision.load_calibration(yaml_path)
            # Verify if the default target color exists in the loaded config
            if self.target_color not in self.vision.colors:
                self.get_logger().warn(
                    f"Default color '{self.target_color}' not found in YAML! "
                    f'Waiting for command...'
                )
        except (FileNotFoundError, ValueError, KeyError, IOError) as e:
            self.get_logger().error(f'Failed to load calibration: {e}')

        # Store intrinsics separately
        self.sky_intrinsics = None
        self.wrist_intrinsics = None

        # ==================================
        # 2. Communication - BugGod (SkyCam)
        # ==================================
        sky_cb_group = MutuallyExclusiveCallbackGroup()
        self.sky_image_sub = self.create_subscription(
            Image,
            '/camera/buggod/color/image_raw',
            self.sky_image_cb,
            10,
            callback_group=sky_cb_group,
        )
        self.sky_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/buggod/color/camera_info',
            self.sky_info_cb,
            10,
            callback_group=sky_cb_group,
        )

        self.bug_array_pub = self.create_publisher(BugArray, '/bug_god/bug_array', 10)
        self.sky_debug_pub = self.create_publisher(Image, '/bug_god/debug_view', 10)
        self.sky_mask_pub = self.create_publisher(Image, '/bug_god/mask_view', 10)

        # ==================================
        # 3. Communication - Wrist Cam
        # ==================================
        wrist_cb_group = MutuallyExclusiveCallbackGroup()
        self.wrist_image_sub = self.create_subscription(
            Image,
            '/camera/wrist_camera/color/image_raw',
            self.wrist_image_cb,
            10,
            callback_group=wrist_cb_group,
        )
        self.wrist_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/wrist_camera/color/camera_info',
            self.wrist_info_cb,
            10,
            callback_group=wrist_cb_group,
        )

        self.wrist_pose_pub = self.create_publisher(PoseArray, '/wrist_camera/bug_poses', 10)
        self.wrist_debug_pub = self.create_publisher(Image, '/wrist_camera/debug_view', 10)
        self.wrist_mask_pub = self.create_publisher(Image, '/wrist_camera/mask_view', 10)

        # ==================================
        # 4. Target Color Command Subscriber
        # ==================================
        self.command_sub = self.create_subscription(
            String, '/target_color', self.command_callback, 10
        )

        self.get_logger().info(f'Node started. Current Target: [{self.target_color}]')

    # -----------------------------------------------------------------
    # Helper Functions
    # -----------------------------------------------------------------
    def _pixel_to_pose(self, u, v, intrinsics, height):
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

    def get_cam_height_and_transform(self, camera_frame):
        """Calculate camera height relative to base, and return the transform object."""
        try:
            cam_tf = self.tf_buffer.lookup_transform(
                self.base_frame, camera_frame, rclpy.time.Time()
            )
            cam_height = cam_tf.transform.translation.z
            return cam_height, cam_tf
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            self.get_logger().warn(
                f'Failed to get transform from {camera_frame} to {self.base_frame}.'
            )
            return None, None

    def transform_pose_to_base(self, pose_in_cam, transform_msg):
        """Transform a Pose from a given frame to the base frame."""
        if transform_msg is None:
            return None

        pose_stamped = PoseStamped()
        pose_stamped.pose = pose_in_cam

        try:
            out_pose_stamped = do_transform_pose(pose_stamped, transform_msg)
            return out_pose_stamped.pose
        except tf2_ros.TransformException as e:
            self.get_logger().error(f'Pose transformation failed: {e}')
            return None

    # -----------------------------------------------------------------
    # Callback Functions
    # -----------------------------------------------------------------
    def command_callback(self, msg):
        """
        Handle the callback for target color switching.

        Updates the detection target and resets the SORT tracker to prevent
        ID conflicts between different colors.
        """
        new_color = msg.data.lower().strip()

        # If the color hasn't changed, ignore
        if new_color == self.target_color:
            return

        # Check if the requested color exists in calibration data
        if new_color in self.vision.colors:
            self.get_logger().info(f'Switching Target Color: {self.target_color} -> {new_color}')
            self.target_color = new_color

            # This ensures IDs from the previous color do not persist.
            self.vision.tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.1)
        else:
            self.get_logger().warn(
                f"Received command '{new_color}', but calibration data not found!"
            )

    def sky_info_cb(self, msg):
        """Sky Camera Info callback to retrieve intrinsics."""
        if self.sky_intrinsics is None:
            self.sky_intrinsics = np.array(msg.k).reshape(3, 3)
            self.get_logger().info('Sky Camera (Bug God) Intrinsics Received.')

    def wrist_info_cb(self, msg):
        """Wrist Camera Info callback to retrieve intrinsics."""
        if self.wrist_intrinsics is None:
            self.wrist_intrinsics = np.array(msg.k).reshape(3, 3)
            self.get_logger().info('Wrist Camera Intrinsics Received.')

    def sky_image_cb(self, msg):
        """Process Sky Camera Images: Detect all colors, build BugArray, and determine targets."""
        if self.sky_intrinsics is None:
            return

        # Convert ROS Image to OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return

        cam_frame_id = msg.header.frame_id
        cam_height, transform_stamped = self.get_cam_height_and_transform(cam_frame_id)

        if cam_height is None:
            cam_height = 1.0  # Default height if transform fails

        gripper_pos_base = None

        # 1. Get the gripper position in the sky camera frame
        # TODO: Use get_ee_pose from robotstate.py instead?
        try:
            gripper_tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                'fer_hand_tcp',  # TODO: Change the frame name when the new gripper is available
                rclpy.time.Time(),
            )
            gripper_pos_base = gripper_tf.transform.translation
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            pass

        bug_array_msg = BugArray()
        bug_array_msg.header.stamp = msg.header.stamp
        bug_array_msg.header.frame_id = self.base_frame  # Outputting in Base Frame

        # Canvas for drawing all bugs
        final_debug_frame = frame.copy()

        closest_dist = float('inf')
        target_bug_index = -1
        all_detected_bugs = []

        # 2. Iterate over all calibrated colors
        for color_name in self.vision.colors.keys():
            detections, _, _ = self.vision.detect_objects(frame, color_name)
            results, _ = self.vision.update_tracker(detections, final_debug_frame)

            for obj_id, u, v in results:
                # 3D Projection
                # TODO: Make changes to match Nolan's new msg type
                pose_cam = self._pixel_to_pose(u, v, self.sky_intrinsics, cam_height)

                if transform_stamped is None:
                    continue

                pose_base = self.transform_pose_to_base(pose_cam, transform_stamped)

                if pose_base is None:
                    continue

                pose_stamped_base = PoseStamped()
                pose_stamped_base.header.frame_id = self.base_frame
                pose_stamped_base.header.stamp = msg.header.stamp
                pose_stamped_base.pose = pose_base

                # Build BugInfo
                bug_info = BugInfo()
                bug_info.id = int(obj_id)
                bug_info.color = color_name
                bug_info.pose = pose_stamped_base
                bug_info.target = False

                # Draw on debug frame
                cv2.putText(
                    final_debug_frame,
                    f'{color_name}',
                    (u, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # Check if this is a candidate for Target (Color match + Closest to gripper)
                if color_name == self.target_color and gripper_pos_base:
                    # Calculate distance to gripper
                    dx = pose_stamped_base.pose.position.x - gripper_pos_base.x
                    dy = pose_stamped_base.pose.position.y - gripper_pos_base.y
                    dist = dx**2 + dy**2  # Squared distance for now, TODO: Do we need Z?

                    if dist < closest_dist:
                        closest_dist = dist
                        target_bug_index = len(all_detected_bugs)

                all_detected_bugs.append(bug_info)

        # 3. Mark the target bug
        if target_bug_index != -1:
            all_detected_bugs[target_bug_index].target = True

            cv2.putText(
                final_debug_frame,
                f'TARGET LOCKED: {self.target_color.upper()}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                final_debug_frame,
                f'SEARCHING: {self.target_color.upper()}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        # 4. Publish BugArray
        bug_array_msg.bugs = all_detected_bugs
        self.bug_array_pub.publish(bug_array_msg)
        self.sky_debug_pub.publish(self.bridge.cv2_to_imgmsg(final_debug_frame, encoding='bgr8'))

    def wrist_image_cb(self, msg):
        """Process Wrist Camera Images: Only looks for the assigned TARGET COLOR."""
        if self.wrist_intrinsics is None:
            return

        # Convert ROS Image to OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return

        cam_frame_id = msg.header.frame_id
        cam_height, _ = self.get_cam_height_and_transform(cam_frame_id)
        if cam_height is None:
            cam_height = 0.5  # Default height if transform fails

        # 1. Detect ONLY the target color
        detections, debug_frame, mask = self.vision.detect_objects(frame, self.target_color)
        tracked_results, final_frame = self.vision.update_tracker(detections, debug_frame)

        # 2. Build PoseArray
        bug_poses = PoseArray()
        bug_poses.header.stamp = msg.header.stamp
        bug_poses.header.frame_id = self.base_frame  # Output in Base Frame

        for obj_id, u, v in tracked_results:
            pose_cam = self._pixel_to_pose(u, v, self.wrist_intrinsics, cam_height)
            pose_stamped_cam = PoseStamped()
            pose_stamped_cam.header = msg.header
            pose_stamped_cam.pose = pose_cam

            try:
                pose_stamped_base = self.tf_buffer.transform(
                    pose_stamped_cam, self.base_frame, timeout=rclpy.duration.Duration(seconds=0.1)
                )
                bug_poses.poses.append(pose_stamped_base.pose)
            except tf2_ros.TransformException as e:
                self.get_logger().error(f'Failed to transform wrist pose to base: {e}')
                continue

        # Annotate
        cv2.putText(
            final_frame,
            f'WRIST TARGET: {self.target_color.upper()}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        # 3. Publish
        self.wrist_pose_pub.publish(bug_poses)
        self.wrist_debug_pub.publish(self.bridge.cv2_to_imgmsg(final_frame, encoding='bgr8'))

        if mask is not None:
            self.wrist_mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding='mono8'))


def main(args=None):
    """Run the main function for the ColorDetection node."""
    try:
        rclpy.init(args=args)
        node = ColorDetection()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
