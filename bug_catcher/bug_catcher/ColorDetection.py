"""Camera Node."""

from bug_catcher.vision import Vision

from enum import auto, Enum

from cv_bridge import CvBridge

from geometry_msgs.msg import Pose, PoseArray
from rclpy.qos import qos_profile_sensor_data
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class State(Enum):
    """
    Current state of the system.

    Determines what the main timer function should be doing on each
        iteration for the physics of the brick.
    """

    WAITING = auto()
    DETECTING = auto()


class ColorDetection(Node):
    """
    Color Detection node for the bug catcher.

    This node subscribes to the camera, applies filters from vision.py onto the video
    frames, publishes the video frame to a camera window and tracks the filtered object.
    It also pubishes the center of the filtered objects

    """

    def __init__(self):
        """Initialize the camera node."""
        super().__init__('color_detection_node')

        # Initializing the camera from config file:
        self.declare_parameter(name='image_topic', value='/camera/camera/color/image_raw')
        self.declare_parameter(name='camera_info_topic', value='/camera/camera/color/camera_info')
        self.declare_parameter(name='camera_frame', value='')
        # Set the parameter values:
        image_topic = (self.get_parameter('image_topic').get_parameter_value().string_value)
        info_topic = (
            self.get_parameter('camera_info_topic').get_parameter_value().string_value
        )
        self.camera_frame = (self.get_parameter('camera_frame').get_parameter_value().string_value)

        # camera number for webcam = 0, can be changed depending on device
        # self.camera_device_number = 0
        # self.camera = cv2.VideoCapture(self.camera_device_number)

        # Initializing the cvbridge. cvbrige converts between ROS Image messages and OpenCV images.
        self.bridge = CvBridge()

        # Timer:
        self.timer_update = self.create_timer(0.01, self.bug_updater)

        # Publishers:
        self.camera_publisher = self.create_publisher(Image, 'color_filter_image', 10)
        self.bug_poses_publisher = self.create_publisher(PoseArray, 'bug_poses', 10)

        # Subscriptions:
        # Camera_info subscription
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self.camera_info_callback, qos_profile_sensor_data
        )
        # Camera image subscription
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data
        )
        # Calibration Success Subscription'
        self.calibrate_sub = self.create_subscription(
            Bool, 'calibrate', self.calibrate_callback, 10
        )
        # Target Color Position for ROI in Color Filtering:
        self.targetROI_sub = self.create_subscription(Pose, 'roi_pose', self.roi_pose_callback, 10)

        # Initializing Vision class from vision.py
        self.vision = Vision()

        # Declare the filename from the launchfile:
        self.declare_parameter('file', 'bug_color_hsv.yaml')
        filename = self.get_parameter('file').value
        self.vision.load_color(filename=filename)

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None
        # Set up calibration bool:
        self.calibrated = False
        # Set the current cv Frame:
        self.cv_image = None
        self.image_header = None
        # Set the target pose for ROI:
        self.target_pose = None
        # Set an initial state:
        self.state = State.WAITING

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

    # def image_callback(self, msg: Image):
    #     """Convert ros2 msg to opencv image data (np.ndarray)."""
    #     opencv_img_msg = self.bridge_object.imgmsg_to_cv2(msg)

    #     # display video window
    #     cv2.imshow('Video', opencv_img_msg)
    #     cv2.waitKey(1)

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
        # Convert the current frame to cv2:
        self.cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        # Set image header: 
        self.image_header = img_msg.header.stamp

    def bug_updater(self):
        """
        Apply filter on the video frame and track and publish the bug Position.

        This function applies the filters from the Vision class onto the video
        frame and tracks the center of the bugs. It publishes the video
        frame and the center of the bugs

        """
        if self.state == State.WAITING:
            # Wait to recieve a message from Calibration node that the system is calibrated.
            if self.calibrated is True:
                self.state = State.DETECTING
        elif self.state == State.DETECTING:
            # # capturing and reading the video frame
            # success, frame = self.camera.read()

            if self.cv_image is None:
                self.get_logger().warn('No camera frame has been recieved yet!')
                return

            # Set the curret frame:
            frame = self.cv_image
            # # resizing the video window
            # frame = cv2.resize(frame, (800, 800), interpolation=cv2.INTER_CUBIC)

            # Detect color in input ROI and switch filter:
            #   Switch the target pose to a pixel coord:
            # Camera intrinsics:
            fx = self.intrinsic_mat[0, 0]
            fy = self.intrinsic_mat[1, 1]
            cx0 = self.intrinsic_mat[0, 2]
            cy0 = self.intrinsic_mat[1, 2]

            # Extract 3D target position in camera frame
            Xtarg = self.target_pose.position.x
            Ytarg = self.target_pose.position.y
            Ztarg = self.target_pose.position.z

            # Project 3D point to pixel coordinates
            u = int(fx * Xtarg / Ztarg + cx0)
            v = int(fy * Ytarg / Ztarg + cy0)

            target_pixel = (u, v)

            self.vision.detect_input_and_switch_filter(frame, target_pixel, roi_size=10)

            # Get the frame after the filter the mask have been applied.
            frame_after_filter, frame_threshold = self.vision.processing_video_frame(
                frame, self.vision.current_low_hsv, self.vision.current_high_hsv
            )

            # Find the contour around the detected bug, draw the contour and track the center
            contour, heirarchy = self.vision.add_contour(frame_threshold)
            detections = self.vision.draw_bounding_box(contours=contour, frame=frame_after_filter)
            tracked = self.vision.tracker.update(detections)
            centers = self.vision.track_results(tracked, frame_after_filter)

            # applying border around ROI
            frame_with_border_on_ROI = self.vision.apply_border_on_ROI(frame_after_filter, frame)

            # Convert the opencv np.ndarray msg to ros2 msg and publish
            ros2_img_msg = self.bridge.cv2_to_imgmsg(frame_with_border_on_ROI)
            self.camera_publisher.publish(ros2_img_msg)

            # Initialize PoseArray
            bug_poses = PoseArray()
            bug_poses.header.stamp = self.get_clock().now().to_msg()

            # Set frame ID
            if self.camera_frame == '':
                bug_poses.header.frame_id = self.info_msg.header.frame_id
            else:
                bug_poses.header.frame_id = self.camera_frame

            # Loop through detected bug centers
            if centers:
                for i, (cx, cy) in enumerate(centers):
                    # TODO: This is in Camera frame not robot base, so we need to convert them before publishing.
                    # Convert the centers into CV positions:
                    Z = 0.00125      # Constant for base of board on table (NOT CORRECT)
                    X = (cx - cx0) * Z / fx
                    Y = (cy - cy0) * Z / fy

                    # Create Pose for each bug identified for that color.
                    pose = Pose()
                    pose.position.x = float(X)
                    pose.position.y = float(Y)
                    pose.position.z = float(Z)
                    # Orientation always identity quaternion for our purpose.
                    pose.orientation.x = 0.0
                    pose.orientation.y = 0.0
                    pose.orientation.z = 0.0
                    pose.orientation.w = 1.0

                    # Append to PoseArray.
                    bug_poses.poses.append(pose)
                    # TODO: Subscribe to these poses, and publish them in the Calibration_TargetPublisher node.
                    # We may need to have a new message type for publication of bugs with their name and position.

            # Publish one message containing all bug poses
            self.bug_poses_publisher.publish(bug_poses)

    def calibrate_callback(self, success_msg):
        # If the subscription gets a bool of True, set the parameter to True
        if success_msg.data is True:
            self.calibrated = True

    def roi_pose_callback(self, pose_msg):
        # Set the target location of the color filter for target ID:
        self.target_pose = pose_msg


def main(args=None):
    """Entry point for the Camera Node."""
    rclpy.init(args=args)
    node = ColorDetection()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
