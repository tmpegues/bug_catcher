"""
The script implements the 'color_detection_node' within the 'bug_catcher' package.

It serves as the vision processing unit for the project, performing object detection
based on color masking, tracking objects using the SORT algorithm, and projecting
2D pixel coordinates into 3D world coordinates for robotic grasping.

Publishers
----------
+ processed_image (sensor_msgs.msg.Image): Publishes debug visualization frames.
+ bug_poses (geometry_msgs.msg.PoseArray): Publishes 3D poses of tracked bugs.

Subscribers
-----------
+ /camera/camera/color/image_raw (sensor_msgs.msg.Image): Receives RGB camera feed.
+ /camera/camera/color/camera_info (sensor_msgs.msg.CameraInfo): Receives camera intrinsics.
+ /target_color (std_msgs.msg.String): Receives commands to switch the detection target.

Parameters
----------
+ default_color (string, default='red'): The initial color to detect upon startup.
+ camera_height (float, default=0.5): Distance from camera lens to the table surface (meters).

"""

import os

from ament_index_python.packages import get_package_share_directory

from bug_catcher.sort import Sort
from bug_catcher.vision import Vision

import cv2

from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Pose, PoseArray
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, Image

from std_msgs.msg import String


class ColorDetection(Node):
    """The Color Detection Node."""

    def __init__(self):
        """Initialize the ColorDetection node."""
        super().__init__('color_detection_node')

        # Parameters
        self.declare_parameter('default_color', 'red')
        self.target_color = self.get_parameter('default_color').value
        self.declare_parameter('camera_height', 0.5)
        self.camera_height = self.get_parameter('camera_height').value

        # Vision System Setup
        self.vision = Vision()

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

        # Utilities
        self.bridge = CvBridge()
        self.intrinsics = None

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_cb, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.info_cb, 10
        )
        self.command_sub = self.create_subscription(
            String, '/target_color', self.command_callback, 10
        )

        # Publishers
        self.vis_pub = self.create_publisher(Image, 'processed_image', 10)
        self.pose_pub = self.create_publisher(PoseArray, 'bug_poses', 10)

        self.get_logger().info(f'Node started. Current Target: [{self.target_color}]')

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

            # CRITICAL: Reset the tracker when switching colors.
            # This ensures IDs from the previous color do not persist.
            self.vision.tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.1)
        else:
            self.get_logger().warn(
                f"Received command '{new_color}', but calibration data not found!"
            )

    def info_cb(self, msg):
        """Camera Info callback to retrieve intrinsics."""
        if self.intrinsics is None:
            self.intrinsics = np.array(msg.k).reshape(3, 3)
            self.get_logger().info('Camera Intrinsics Received.')

    def image_cb(self, msg):
        """
        Handle the main image processing callback.

        This handles conversion, detection, tracking, 3D projection, and publishing.
        """
        if self.intrinsics is None:
            return

        # Convert ROS Image to OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return

        # 1. Vision Processing (Detection & Tracking)
        # detect_objects returns bounding boxes and a frame with debug drawings
        detections, debug_frame = self.vision.detect_objects(frame, self.target_color)

        # Update SORT tracker to get persistent IDs
        tracked_results, final_frame = self.vision.update_tracker(detections, debug_frame)

        # Annotate target on screen
        cv2.putText(
            final_frame,
            f'TARGET: {self.target_color.upper()}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        # 2. Coordinate Conversion (Pixel -> Camera Frame)
        bug_poses = PoseArray()
        bug_poses.header = msg.header

        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]

        # Z is assumed constant based on camera height
        Z = self.camera_height

        # Process tracked objects
        if len(tracked_results) > 0:
            for obj_id, u, v in tracked_results:
                # Pinhole Camera Model Projection
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy

                pose = Pose()
                pose.position.x = float(X)
                pose.position.y = float(Y)
                pose.position.z = float(Z)
                # Orientation is standard identity
                pose.orientation.w = 1.0

                bug_poses.poses.append(pose)

            # Publish poses
            self.pose_pub.publish(bug_poses)

        # 3. Publish Visualization
        self.vis_pub.publish(self.bridge.cv2_to_imgmsg(final_frame, encoding='bgr8'))


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
