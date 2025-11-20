import cv2
import numpy as np
from sort import Sort

class Vision:
    def __init__(self):
        self.window_name = "Webcam"
        self.window_mask_name = "Mask"
        self.window_only_rgb = "Bug"

        self.cap = cv2.VideoCapture(0)

        cv2.namedWindow(self.window_name)

        # store current frame
        self.current_frame = None

        self.max_value = 255
        self.max_value_H = 360//2
        self.low_H = 0
        self.low_S = 0
        self.low_V = 0
        self.high_H = self.max_value_H
        self.high_S = self.max_value
        self.high_V = self.max_value
        self.low_H_name = 'Low H'
        self.low_S_name = 'Low S'
        self.low_V_name = 'Low V'
        self.high_H_name = 'High H'
        self.high_S_name = 'High S'
        self.high_V_name = 'High V'

        self.tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.0)
        self.detections = []

        # create trackbar (blur+update combined)
        
        cv2.createTrackbar(self.low_H_name, self.window_name, self.low_H, self.max_value_H, self.on_low_H_thresh_trackbar)
        cv2.createTrackbar(self.high_H_name,self.window_name , self.high_H, self.max_value_H, self.on_high_H_thresh_trackbar)
        cv2.createTrackbar(self.low_S_name, self.window_name, self.low_S, self.max_value, self.on_low_S_thresh_trackbar)
        cv2.createTrackbar(self.high_S_name,self.window_name , self.high_S, self.max_value, self.on_high_S_thresh_trackbar)
        cv2.createTrackbar(self.low_V_name, self.window_name, self.low_V, self.max_value, self.on_low_V_thresh_trackbar)
        cv2.createTrackbar(self.high_V_name,self.window_name , self.high_V, self.max_value, self.on_high_V_thresh_trackbar)
    
    def on_low_H_thresh_trackbar(self, val):        
        self.low_H = val
        self.low_H = min(self.high_H-1, self.low_H)
        cv2.setTrackbarPos(self.low_H_name, self.window_name, self.low_H)

    def on_high_H_thresh_trackbar(self, val):        
        self.high_H = val
        self.high_H = max(self.high_H, self.low_H+1)
        cv2.setTrackbarPos(self.high_H_name, self.window_name, self.high_H)

    def on_low_S_thresh_trackbar(self, val):        
        self.low_S = val
        self.low_S = min(self.high_S-1, self.low_S)
        cv2.setTrackbarPos(self.low_S_name, self.window_name, self.low_S)

    def on_high_S_thresh_trackbar(self, val):        
        self.high_S = val
        self.high_S = max(self.high_S, self.low_S+1)
        cv2.setTrackbarPos(self.high_S_name, self.window_name, self.high_S)

    def on_low_V_thresh_trackbar(self, val):        
        self.low_V = val
        self.low_V = min(self.high_V-1, self.low_V)
        cv2.setTrackbarPos(self.low_V_name, self.window_name, self.low_V)

    def on_high_V_thresh_trackbar(self, val):        
        self.high_V = val
        self.high_V = max(self.high_V, self.low_V+1)
        cv2.setTrackbarPos(self.high_V_name, self.window_name, self.high_V)

    def add_contour(self,mask=None):
        # Morphological closing (fills small gaps inside the pen mask)
        kernel = np.ones((3,3), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)        
        mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        #cv2.imshow("Closed Mask", mask_closed)
        contours, hierarchy = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy
        
    def draw_bounding_box(self, contours, frame):
        if contours is None or len(contours) == 0:
            return np.empty((0,5))  # no contours found   
        area = 100
        detections = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area<area:
                continue     

            if len(contour) <5:
                continue

            x, y, w, h = cv2.boundingRect(contour) 

            if w < 5 or h <5:
                continue

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), -1)
            detections.append([x, y, x + w, y + h, 1.0])

            

        """# Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # fitRectangle needs at least 4 points
        if len(largest_contour) < 4:
            return np.empty((0, 5))

        x, y, w, h = cv2.boundingRect(largest_contour) 
        if w == 0 or h == 0:
            return np.empty((0, 5))"""
        if len(detections)>0:
            return np.array(detections, dtype=float)
        else:
            return np.empty((0,5))
    
    def track_results(self, tracked, frame):
        for x1, y1, x2, y2, track_id in tracked:
            if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
        
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)
            cv2.putText(frame, f"ID {int(track_id)}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    
    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_threshold = cv2.inRange(frame_HSV, (self.low_H, self.low_S, self.low_V), (self.high_H, self.high_S, self.high_V))
            only_rgb = cv2.bitwise_and(frame, frame, mask=frame_threshold)
            
            

            self.current_frame = frame

            contour, heirarchy = self.add_contour(frame_threshold)
            detections = self.draw_bounding_box(contours=contour, frame=only_rgb)
            
            tracked = self.tracker.update(detections)
            print(f"Detections: {detections}, Tracked: {tracked}")
            
            self.track_results(tracked, only_rgb)


            cv2.imshow(self.window_name, frame)
            cv2.imshow(self.window_mask_name, frame_threshold)
            cv2.imshow(self.window_only_rgb, only_rgb)

            

            # --- allow quitting with Q ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # --- allow quitting by pressing the window close button ---
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1 or \
               cv2.getWindowProperty(self.window_mask_name, cv2.WND_PROP_VISIBLE) < 1 or \
               cv2.getWindowProperty(self.window_only_rgb, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.cap.release()
        cv2.destroyAllWindows()


# run it
if __name__ == "__main__":
    Vision().run()
