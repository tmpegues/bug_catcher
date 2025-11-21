"""
Vision processing for the bug catcher.

Captures webcam frames, applies HSV masking and contour detection, and tracks detections
using the SORT tracker.
"""

import cv2

import numpy as np

from sort import Sort


class Vision:
    """
    Vision processing for the bug catcher.

    Captures webcam feed, applies color masking, detects contours, and tracks
    objects with persistent IDs. Supports multiple color profiles.
    """

    def __init__(self):
        """
        Initialize.

        Set up window, video capture, HSV threshold defaults for different colors,
        blur value and the object tracker.
        """
        self.window_tracking = 'Tracking'

        self.cap = cv2.VideoCapture(0)

        cv2.namedWindow(self.window_tracking)

        # store current video frame
        self.current_frame = None

        # Blue
        self.blue_low_hsv = (100, 150, 0)
        self.blue_high_hsv = (140, 225, 225)

        # Purple
        self.purple_low_hsv = (113, 40, 20)
        self.purple_high_hsv = (160, 225, 225)

        # Orange
        self.orange_low_hsv = (0, 150, 120)
        self.orange_high_hsv = (180, 225, 225)

        # Green
        self.green_low_hsv = (35, 82, 80)
        self.green_high_hsv = (85, 225, 225)

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

    def blur_frame(self, frame):
        """
        Apply a blur filter on the video frame.

        Args
        ----
        frame : np.ndarray
           Video frame on which the filter needs to be applied.

        Returns
        -------
        frame_blur : np.ndarray
            Video frame with the blur filter applied.

        """
        frame_blur = cv2.GaussianBlur(frame, (self.blur, self.blur), 0)
        return frame_blur

    def color_filter(self, frame, low_hsv, high_hsv):
        """
        Apply a color filter on the video frame.

        Args
        ----
        frame : np.ndarray
           Video frame on which the filter needs to be applied.
        low_hsv : tuple[int, int, int]
           Lower HSV value of the filter
        high_hsv : tuple[int, int, int]
            Higher HSV value of the filter

        Returns
        -------
        frame_threshold : np.ndarray
            Video frame with the color filter applied

        """
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, low_hsv, high_hsv)
        return frame_threshold

    def only_rgb(self, frame, mask):
        """
        Apply a mask on the video frame.

        Apply a mask on the video frame to keep only the pixels within
        the selected color range.

        Args
        ----
        frame : np.ndarray
           Video frame on which the  needs to be applied.
        mask : np.ndarray
           A binary mask indicating which pixels to keep (non-zero) or hide (zero).

        Returns
        -------
        only_rgb : np.ndarray
            Video frame with only un-masked areas visible.

        """
        only_rgb = cv2.bitwise_and(frame, frame, mask=mask)
        return only_rgb

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
        None

        """
        for x1, y1, x2, y2, track_id in tracked:
            if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

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

    # #################### End_Citation [13] ##################

    def run(self):
        """Run the main application loop."""
        if not self.cap.isOpened():
            print('Error: Could not open camera')
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.current_frame = frame

            # blur frame
            frame_blur = self.blur_frame(frame)

            # input key
            key = cv2.waitKey(1) & 0xFF

            # shift to blue color filter by pressing the 'b' button
            if key == ord('b'):
                self.current_low_hsv = self.blue_low_hsv
                self.current_high_hsv = self.blue_high_hsv
                print('switch to blue')

            # shift to orange color filter by pressing the 'o' button
            elif key == ord('o'):
                self.current_low_hsv = self.orange_low_hsv
                self.current_high_hsv = self.orange_high_hsv
                print('switch to orange')

            # shift to purple color filter by pressing the 'p' button
            elif key == ord('p'):
                self.current_low_hsv = self.purple_low_hsv
                self.current_high_hsv = self.purple_high_hsv
                print('switch to purple')

            # shift to green color filter by pressing the 'g' button
            elif key == ord('g'):
                self.current_low_hsv = self.green_low_hsv
                self.current_high_hsv = self.green_high_hsv
                print('switch to green')

            # quitting by pressing the 'q' button
            elif key == ord('q'):
                print('closing')
                break

            # applying the color filter
            frame_threshold = self.color_filter(
                frame=frame_blur, low_hsv=self.current_low_hsv, high_hsv=self.current_high_hsv
            )

            # bitwise mask on the blurred video frame
            only_rgb = self.only_rgb(frame=frame_blur, mask=frame_threshold)

            # detecting contours
            contour, heirarchy = self.add_contour(frame_threshold)

            # drawing a bounding box around the contour
            detections = self.draw_bounding_box(contours=contour, frame=only_rgb)

            # tracking the contour detected
            tracked = self.tracker.update(detections)

            # display the tracked results
            self.track_results(tracked, only_rgb)

            # display the window
            cv2.imshow(self.window_tracking, only_rgb)

            # allow quitting by pressing the window close button
            if cv2.getWindowProperty(self.window_tracking, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Vision().run()
