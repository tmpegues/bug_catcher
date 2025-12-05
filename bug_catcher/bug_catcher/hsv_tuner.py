"""
The script implements the 'ros_hsv_tuner' node.

It is a utility tool designed to help calibrate HSV (Hue, Saturation, Value)
thresholds for computer vision tasks. It subscribes to a live camera feed,
provides an interactive GUI with trackbars to adjust thresholds in real-time,
and saves the resulting configuration to a YAML file for use by other nodes.

Subscribers
-----------
+ /camera/camera/color/image_raw (sensor_msgs.msg.Image): Receives the raw RGB camera feed.

Outputs
-------
+ calibrated_colors.yaml: A YAML file generated in the working directory containing
  the tuned values.

"""

import cv2

from cv_bridge import CvBridge

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

import yaml


def nothing():
    """Provide a dummy callback function for OpenCV trackbars."""
    pass


class RosHSVTuner(Node):
    """
    ROS Node for interactive HSV Threshold Tuning.

    Attributes
    ----------
    bridge (CvBridge): Converts ROS Image messages to OpenCV format.
    window_name (str): Name of the OpenCV GUI window.
    image_topic (str): The topic name to subscribe to.
    current_color_name (str): The label of the color currently being tuned.
    colors_data (list): List storing the calibrated data to be saved.

    """

    def __init__(self):
        """Initialize the RosHSVTuner node."""
        super().__init__('ros_hsv_tuner_node')

        self.bridge = CvBridge()
        self.window_name = 'ROS HSV Tuner'

        self.image_topic = '/camera/camera/color/image_raw'

        self.get_logger().info(f'Subscribing to Image Topic: {self.image_topic}')
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)

        self.current_frame = None
        self.colors_data = []
        self.current_color_name = 'blue'  # Start tuning for 'blue' by default

        self._setup_trackbars()

        # Timer to handle GUI updates and logic (approx 30 FPS)
        # This decouples the GUI loop from the ROS callback to prevent freezing
        self.timer = self.create_timer(0.03, self.timer_callback)

    def _setup_trackbars(self):
        """Initialize the OpenCV window and create HSV trackbars."""
        cv2.namedWindow(self.window_name)

        # Create trackbars for Lower and Upper HSV limits
        cv2.createTrackbar('Low H', self.window_name, 0, 180, nothing)
        cv2.createTrackbar('High H', self.window_name, 180, 180, nothing)
        cv2.createTrackbar('Low S', self.window_name, 0, 255, nothing)
        cv2.createTrackbar('High S', self.window_name, 255, 255, nothing)
        cv2.createTrackbar('Low V', self.window_name, 0, 255, nothing)
        cv2.createTrackbar('High V', self.window_name, 255, 255, nothing)

        print('------------------------------------------------')
        print(f'Currently Tuning: {self.current_color_name}')
        print('Controls:')
        print('Adjust trackbars to isolate the object.')
        print("Press 's' to save the current color and move to the next.")
        print("Press 'q' to quit and generate the YAML file.")
        print('------------------------------------------------')

    def image_callback(self, msg):
        """
        Handle the camera image topic.

        Only handles data conversion to ensure the callback remains lightweight.

        Args:
        ----
        msg (sensor_msgs.msg.Image): The incoming image message.

        """
        try:
            # Convert ROS Image message to OpenCV BGR format
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_frame = self.current_frame[50:650, 350:975]
        except (FileNotFoundError, ValueError, KeyError, IOError) as e:
            self.get_logger().error(f'CvBridge error: {e}')

    def timer_callback(self):
        """
        Run the main processing loop triggered by the timer.

        Handle image processing, masking, GUI updates, and user input.
        """
        if self.current_frame is None:
            return

        # Work on a copy of the frame
        frame = self.current_frame.copy()

        # Retrieve current positions of all trackbars
        l_h = cv2.getTrackbarPos('Low H', self.window_name)
        h_h = cv2.getTrackbarPos('High H', self.window_name)
        l_s = cv2.getTrackbarPos('Low S', self.window_name)
        h_s = cv2.getTrackbarPos('High S', self.window_name)
        l_v = cv2.getTrackbarPos('Low V', self.window_name)
        h_v = cv2.getTrackbarPos('High V', self.window_name)

        lower_hsv = np.array([l_h, l_s, l_v])
        upper_hsv = np.array([h_h, h_s, h_v])

        # --- Image Processing ---
        # 1. Apply Gaussian Blur to reduce noise
        frame_blur = cv2.GaussianBlur(frame, (11, 11), 0)

        # 2. Convert to HSV color space
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        # 3. Create the mask based on trackbar values
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # 4. Apply mask to original frame for visualization
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Overlay text status
        cv2.putText(
            result,
            f'Editing: {self.current_color_name}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Show windows
        cv2.imshow(self.window_name, result)
        cv2.imshow('Mask', mask)

        # Handle Keyboard Input
        # WaitKey is required here for OpenCV GUI to function properly
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            self._save_color_entry(l_h, h_h, l_s, h_s, l_v, h_v)

        elif key == ord('q'):
            self._save_to_yaml()
            rclpy.shutdown()
            cv2.destroyAllWindows()

    def _save_color_entry(self, l_h, h_h, l_s, h_s, l_v, h_v):
        """
        Store the current HSV parameters in memory.

        Args:
        ----
        l_h (int): Lower bound value for Hue.
        l_s (int): Lower bound value for Saturation.
        l_v (int): Lower bound value for Value.
        h_h (int): Upper bound value for Hue.
        h_s (int): Upper bound value for Saturation.
        h_v (int): Upper bound value for Value.

        """
        print(f'--> Saving parameters for color [{self.current_color_name}]...')

        entry = {
            'color': self.current_color_name,
            'hsv': {'low': [int(l_h), int(l_s), int(l_v)], 'high': [int(h_h), int(h_s), int(h_v)]},
        }
        self.colors_data.append(entry)

        # Print current list of saved colors
        saved_names = [c['color'] for c in self.colors_data]
        print(f'--> Current saved colors: {saved_names}')

        # Prompt user for the next color name
        new_color = input("Enter the name for the next color (e.g., 'green'): ")
        if new_color:
            self.current_color_name = new_color
        else:
            print('Continuing to tune the current color...')

    def _save_to_yaml(self):
        """Write all stored color calibration data to a YAML file."""
        filename = 'calibrated_colors.yaml'

        # Save to the current working directory
        with open(filename, 'w') as f:
            yaml.dump(self.colors_data, f)

        self.get_logger().info(f'Successfully saved all color parameters to {filename}')


def main(args=None):
    """Run the main function for the RosHSVTuner node."""
    rclpy.init(args=args)
    node = RosHSVTuner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
