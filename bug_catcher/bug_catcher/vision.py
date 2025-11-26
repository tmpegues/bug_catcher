"""
Vision processing for the bug catcher.

Captures webcam frames, applies HSV masking and contour detection, and tracks detections
using the SORT tracker.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from bug_catcher.sort import Sort

import cv2

import numpy as np

import yaml


class Vision:
    """
    Vision processing for the bug catcher.

    Captures webcam feed, applies color masking, detects contours, and tracks
    objects with persistent IDs. Supports multiple color profiles.
    """

    def __init__(self):
        """
        Initialize.

        Setup the blur value, the object tracker and the current low and high hsv values

        """
        # Blur Value
        self.blur = 15

        # Current Value
        self.current_low_hsv = (0, 0, 0)
        self.current_high_hsv = (180, 255, 255)

        # Tracker
        # #################### Begin_Citation [13] ##################
        self.tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.0)
        # #################### End_Citation [13] ##################
        self.detections = []

    def load_color(self, filename):
        """Load the colors from the config file."""
        # Read through the yaml file and get the hsv values for the respective color:
        pkg_share = get_package_share_directory('bug_catcher')
        file = Path(pkg_share) / 'config' / filename
        with open(file, 'r') as f:
            data = yaml.safe_load(f)

        for entry in data:
            # Get the color hsv
            if entry['color'].lower() == 'blue':
                self.blue_low_hsv = np.array(entry['hsv']['low'])
                self.blue_high_hsv = np.array(entry['hsv']['high'])
            elif entry['color'].lower() == 'green':
                self.green_low_hsv = np.array(entry['hsv']['low'])
                self.green_high_hsv = np.array(entry['hsv']['high'])
            elif entry['color'].lower() == 'orange':
                self.orange_low_hsv = np.array(entry['hsv']['low'])
                self.orange_high_hsv = np.array(entry['hsv']['high'])
            elif entry['color'].lower() == 'purple':
                self.purple_low_hsv = np.array(entry['hsv']['low'])
                self.purple_high_hsv = np.array(entry['hsv']['low'])

    def processing_video_frame(self, frame, low_hsv, high_hsv):
        """
        Apply a blur filter and color filter on the video frame, apply a mask excluding the ROI.

        This function first applies a blur filter on the input video frame. The blurred
        frame is then converted to HSV, and the given lower and higher HSV thresholds
        are used to create a binary mask. This mask is applied to the
        blurred frame so that only the areas within the selected HSV range remain
        visible. Finally all the pixels region of interest (the area within the frame
        that recieves the input color for the filter) is replaced by black [0, 0, 0].

        Args
        ----
        frame : np.ndarray
            Video frame on which the filters need to be applied.
        low_hsv : tuple[int, int, int]
            Lower HSV value of the color filter.
        high_hsv : tuple[int, int, int]
            Higher HSV value of the color filter.

        Returns
        -------
        frame_after_mask_except_ROI : np.ndarray
            Blurred video frame with the color filter applied and only the areas
            inside the HSV range visible, except the ROI which is black.
        frame_threshold : np.ndarray
            Binary mask where the filtered color is white(1) and everything
            else is black(0).
        """
        frame_blur = cv2.GaussianBlur(frame, (self.blur, self.blur), 0)
        frame_HSV = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, low_hsv, high_hsv)

        # replacing all the pixels in the ROI to black so that the tracker
        # does not pick it up
        frame_after_mask = cv2.bitwise_and(frame, frame, mask=frame_threshold)
        frame_after_mask_except_ROI = frame_after_mask
        frame_after_mask_except_ROI[50:200, 50:200] = [0, 0, 0]
        return frame_after_mask_except_ROI, frame_threshold

    def detect_input_and_switch_filter(self, frame):
        """
        Switch the color based on the HSV value in the input region of interest.

        Args
        ----
        frame : np.ndarray
            Video frame from which the input hsv values are detected in the ROI

        Return
        ------
        None

        """
        color = frame[50:200, 50:200]
        hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        if np.any(cv2.inRange(hsv_color, self.blue_low_hsv, self.blue_high_hsv)):
            print('switch to blue filter')
            self.current_low_hsv = self.blue_low_hsv
            self.current_high_hsv = self.blue_high_hsv
        elif np.any(cv2.inRange(hsv_color, self.green_low_hsv, self.green_high_hsv)):
            print('switch to green filter')
            self.current_low_hsv = self.green_low_hsv
            self.current_high_hsv = self.green_high_hsv
        elif np.any(cv2.inRange(hsv_color, self.purple_low_hsv, self.purple_high_hsv)):
            print('switch to purple filter')
            self.current_low_hsv = self.purple_low_hsv
            self.current_high_hsv = self.purple_high_hsv
        elif np.any(cv2.inRange(hsv_color, self.orange_low_hsv, self.orange_high_hsv)):
            print('switch to orange filter')
            self.current_low_hsv = self.orange_low_hsv
            self.current_high_hsv = self.orange_high_hsv

    def apply_border_on_ROI(self, frame_after_filter, frame):
        """
        Apply a border around the input region of interest and replace the pixel.

        This function applies a border around the input region of interest and replaces
        the pixels inside with the pixels in the original frame. This is done so that
        the input region of interest is always visible and is not included in the
        tracking

        Args
        ----
        frame_after_filter : np.ndarray
           Video frame on which the border needs to be applied.

        Return
        ------
        frame_after_filter : np.ndarray
           Video frame after border has been applied to ROI and pixels have been replaced

        """
        YELLOW = [0, 255, 255]
        frame_after_filter[50:200, 50:200] = frame[50:200, 50:200]
        cv2.rectangle(frame_after_filter, (50, 50), (200, 200), YELLOW, 5)
        return frame_after_filter

    def add_contour(self, mask=None):
        """
        Clean the given mask and extract its external contours.

        Args
        ----
        mask : np.ndarray
            Binary mask from which contours will be extracted.

        Returns
        -------
        contours : list
            List of detected external contours.
        hierarchy : np.ndarray
            Contour hierarchy information.

        """
        kernel = np.ones((3, 3), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(
            mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours, hierarchy

    def draw_bounding_box(self, contours, frame):
        """
        Draw a bounding box around the given contour on the video frame.

        Args
        ----
        contours : np.ndarray
            Contour for which the bounding box will be drawn.
        frame : np.ndarray
            Video frame where the bounding box will be rendered.

        Returns
        -------
        frame : np.ndarray
            Video frame with the drawn bounding box.

        """
        if contours is None or len(contours) == 0:
            return np.empty((0, 5))  # no contours found
        area = 100
        detections = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < area:
                continue

            if len(contour) < 5:
                continue
            # #################### Begin_Citation [13] ##################
            x, y, w, h = cv2.boundingRect(contour)

            if w < 5 or h < 5:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), -1)
            detections.append([x, y, x + w, y + h, 1.0])

        if len(detections) > 0:
            return np.array(detections, dtype=float)
        else:
            return np.empty((0, 5))

    def track_results(self, tracked, frame):
        """
        Draw tracking results on the video frame.

        Args
        ----
        tracked : (np.ndarray)
            Array of tracked bounding boxes with IDs.
        frame : (np.ndarray)
            Video frame on which tracking results will be drawn.

        Returns
        -------
        centre : tuple[int, int]
            Center of the tracked object in the camera coordinate frame (pixels)

        """
        center = None
        for x1, y1, x2, y2, track_id in tracked:
            if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            center = (cx, cy)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f'ID {int(track_id)}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        if center is not None:
            return center

    # #################### End_Citation [13] ##################
