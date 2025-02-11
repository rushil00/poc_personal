import cv2
import numpy as np
import argparse
import os

from utils import configure_camera, draw_quadrilateral, draw_regions

class MotionDetection:
    def __init__(self, fps=30, regions=None):
        self.last_frame = None
        self.no_motion_frame_limit = fps * 1.5
        self.consecutive_no_motion_frames = 0
        self.motion_detected = False
        self.motion_frame_count = 0
        self.fps = fps

        self.adaptive_threshold = 500.0
        self.adaptive_multiplier = 1.5
        self.increase_alpha = 0.25
        self.decrease_alpha = 0.125
        self.regions_detected = {}
        self.motion_mask = None

        if regions is None:
            self.regions = [np.array([[0, 0], [640, 0], [640, 480], [0, 480]])]
        else:
            self.regions = regions

        self.region_motion_counts = {}
        self.current_active_region = None
        self.region_labels = {i: f"Region {i+1}" for i in range(len(self.regions))}

    def update_adaptive_threshold(self, areas):
        if not areas:
            return

        candidate_threshold = np.mean(areas) * self.adaptive_multiplier
        alpha = self.increase_alpha if candidate_threshold > self.adaptive_threshold else self.decrease_alpha
        self.adaptive_threshold = (1 - alpha) * self.adaptive_threshold + alpha * candidate_threshold
        self.adaptive_threshold = max(500, self.adaptive_threshold)

    def detect_motion(self, frame):
        """
        Detect motion by comparing the current frame with the previous frame.
        
        Parameters:
        frame (np.ndarray): Current frame.
        """
        try:
            if self.last_frame is None:
                self.last_frame = frame
                return

            if self.last_frame.shape != frame.shape:
                frame = cv2.resize(frame, (self.last_frame.shape[1], self.last_frame.shape[0]))

            gray_last = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            gray_last = cv2.GaussianBlur(gray_last, (15, 15), 0)
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_current = cv2.GaussianBlur(gray_current, (15, 15), 0)
            
            # Compute difference between frames
            frame_difference = cv2.absdiff(gray_last, gray_current)
            _, threshold_image = cv2.threshold(frame_difference, 25, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            threshold_image = cv2.dilate(threshold_image, kernel, iterations=2)

            # Create a mask for the regions
            region_mask = np.zeros_like(threshold_image)
            for region in self.regions:
                cv2.fillPoly(region_mask, [region], 255)

            # Apply the region mask to the threshold image
            self.motion_mask = cv2.bitwise_and(threshold_image, region_mask)

            contours, _ = cv2.findContours(self.motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Debug: Show current adaptive threshold value
            print(f"Adaptive Threshold: {self.adaptive_threshold:.2f}")

            # 🔹 Filter contours using adaptive threshold
            filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= self.adaptive_threshold]
            print(f"Filtered contours count: {len(filtered_contours)}")

            total_motion_area = sum(cv2.contourArea(c) for c in filtered_contours)
            noise_areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0]
            self.update_adaptive_threshold(noise_areas)

            if total_motion_area < self.adaptive_threshold * 2.4:
                self.consecutive_no_motion_frames += 1
                if self.consecutive_no_motion_frames >= self.no_motion_frame_limit:
                    self.motion_detected = False
                    self.motion_frame_count = 0  # Reset motion frame count

            else:  # No Motion 
                self.consecutive_no_motion_frames = 0  # Reset no-motion counter
                self.motion_frame_count += 1
                if self.motion_frame_count > 3:
                    self.motion_detected = True

            # Reset region motion counts
            self.region_motion_counts = {i: 0 for i in range(len(self.regions))}
            self.regions_detected.clear()

            # Determine regions of motion based on user-defined regions
            for contour in filtered_contours:
                if cv2.contourArea(contour) > self.adaptive_threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    mask_roi = self.motion_mask[y:y+h, x:x+w]
                    cluster_pixels = cv2.findNonZero(mask_roi)
                    if cluster_pixels is not None and len(cluster_pixels) > 200:
                        avg_x = int(cluster_pixels[:, 0, 0].mean()) + x
                        avg_y = int(cluster_pixels[:, 0, 1].mean()) + y
                        
                        # Check which region contains this point
                        for region_idx, region in enumerate(self.regions):
                            if cv2.pointPolygonTest(region, (avg_x, avg_y), False) >= 0:
                                self.region_motion_counts[region_idx] += 1

            # Determine most active region
            if self.region_motion_counts:
                max_activity_region = max(self.region_motion_counts.items(), key=lambda x: x[1])
                if max_activity_region[1] > 0:  # If there's any motion
                    self.current_active_region = max_activity_region[0]
                    print(f"Most motion in {self.region_labels[self.current_active_region]}")
                else:
                    self.current_active_region = None

            # Store current frame as last frame for next iteration
            self.last_frame = frame

        except Exception as e:
            print(f"Motion detection failed due to: {e}")
            raise e

    def update_motion_status(self, frame, mask, fps):
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        self.detect_motion(masked_frame)
        return self.motion_mask

    def process_frame(self, frame, fps):
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255
        motion_mask = self.update_motion_status(frame, mask, fps)
        
        frame_with_regions = frame.copy()
        for idx, region in enumerate(self.regions):
            color = (0, 255, 0) if idx == self.current_active_region else (0, 165, 255)
            cv2.polylines(frame_with_regions, [region], True, color, 2)
            if idx == self.current_active_region:
                overlay = frame_with_regions.copy()
                cv2.fillPoly(overlay, [region], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame_with_regions, 0.7, 0, frame_with_regions)

        if motion_mask is not None:
            motion_mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        else:
            motion_mask_bgr = np.zeros_like(frame)

        combined_frame = cv2.hconcat([frame_with_regions, motion_mask_bgr])
        return cv2.resize(combined_frame, (1280, 480))

def iterate_main(main_dir='captures/videos'):
    video_files = [f for f in os.listdir(main_dir) if f.endswith('.mp4')]
    for videopath in video_files[-5:]:
        videopath = os.path.join(main_dir, videopath)
        cap = cv2.VideoCapture(videopath)
        if not cap.isOpened():
            print(f"Error: Could not open video {videopath}.")
            continue

        ret, first_frame = cap.read()
        if not ret:
            print(f"Error: Could not read first frame from {videopath}")
            continue

        first_frame = cv2.resize(first_frame, (640, 480))
        num = input("How many regions do you want?")
        regions = draw_regions(first_frame, int(num))
        print(f"REGIONS: {regions}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        motion_detector = MotionDetection(fps, regions)
        frame_count = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            frame_count += 1
            if frame_count % 1 == 0:
                processed_frame = motion_detector.process_frame(frame, fps)
                if motion_detector.motion_detected:
                    cv2.putText(processed_frame, "Motion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(processed_frame, "No Motion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Motion Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("*"*80)
        print("*"*80)
        cap.release()
        cv2.destroyAllWindows()

def main(vidpath):
    cap = cv2.VideoCapture(vidpath)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    num = input("How many regions do you want?\n")
    regions = draw_regions(first_frame, int(num))
    print(f"REGIONS: {regions}")
    motion_detector = MotionDetection(fps, regions)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 1 == 0:
            processed_frame = motion_detector.process_frame(frame, fps)
            if motion_detector.motion_detected:
                cv2.putText(processed_frame, "Motion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(processed_frame, "No Motion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow("Region Detection", processed_frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iterate_main('videos_representative')
    # main(0)