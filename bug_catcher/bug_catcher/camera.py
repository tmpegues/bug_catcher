"""Camera Node."""

from bug_catcher.vision import Vision

import cv2

from cv_bridge import CvBridge

from geometry_msgs.msg import Point

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
# TODO use custom message to publish the bug centers as a list without duplication


class Camera(Node):
    """
    Camera node for the bug catcher.

    This node subscribes to the camera, applies filters from vision.py onto the video
    frames, publishes the video frame to a camera window and tracks the filtered object.
    It also pubishes the center of the filtered objects

    """

    def __init__(self):
        """Initialize the camera node."""
        super().__init__('camera')

        # Initializing the camera device number and capturing video from that camera
        # camera number for webcam = 0, can be changed depending on device
        self.camera_device_number = 0
        self.camera = cv2.VideoCapture(self.camera_device_number)

        # Initializing the cvbridge. cvbrige converts between ROS Image messages
        # and OpenCV images
        self.bridge_object = CvBridge()

        # publishers:
        self.camera_publisher = self.create_publisher(Image, 'camera_image', 10)
        self.bug_centres_publisher = self.create_publisher(Point, 'bug_centers', 10)

        # timer:
        self.timer_update = self.create_timer(0.01, self.update)

        # subscriber:
        self.camera_subscriber = self.create_subscription(
            Image, 'camera_image', self.camera_image, 10
        )

        # Initializing Vision class from vision.py
        self.vision = Vision()

        # Declare the filename from the launchfile:
        self.declare_parameter('file', 'bug_color_hsv.yaml')
        filename = self.get_parameter('file').value
        self.vision.load_color(filename=filename)

    def camera_image(self, msg: Image):
        """Convert ros2 msg to opencv image data (np.ndarray)."""
        opencv_img_msg = self.bridge_object.imgmsg_to_cv2(msg)

        # display video window
        cv2.imshow('Video', opencv_img_msg)
        cv2.waitKey(1)

    def update(self):
        """
        Apply filter on the video frame and track the bug center.

        This function applies the filters from the Vision class onto the video
        frame and tracks the center of the bugs. It publishes the video
        frame and the center of the bugs

        """
        # capturing and reading the video frame
        success, frame = self.camera.read()

        # resizing the video window
        frame = cv2.resize(frame, (800, 800), interpolation=cv2.INTER_CUBIC)

        # detect color in input ROI and switch filter
        self.vision.detect_input_and_switch_filter(frame)

        # get the frame after the filter has been applied and the mask
        frame_after_filter, frame_threshold = self.vision.processing_video_frame(
            frame, self.vision.current_low_hsv, self.vision.current_high_hsv
        )

        # find the contour around the detected bug, draw the contour and track the center
        contour, heirarchy = self.vision.add_contour(frame_threshold)
        detections = self.vision.draw_bounding_box(contours=contour, frame=frame_after_filter)
        tracked = self.vision.tracker.update(detections)
        center = self.vision.track_results(tracked, frame_after_filter)

        # applying border around ROI
        frame_with_border_on_ROI = self.vision.apply_border_on_ROI(frame_after_filter, frame)

        # initializing point
        center_point = Point()

        # if the camera.read has read the frame succesfully
        if success:
            # converting the opencv np.ndarray msg to ros2 msg and publishing
            ros2_img_msg = self.bridge_object.cv2_to_imgmsg(frame_with_border_on_ROI)
            self.camera_publisher.publish(ros2_img_msg)
        # publishing the center of the bug
        if center is not None:
            center_point.x = float(center[0])
            center_point.y = float(center[1])
            self.bug_centres_publisher.publish(center_point)


def main(args=None):
    """Entry point for the Camera Node."""
    rclpy.init(args=args)
    node = Camera()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
