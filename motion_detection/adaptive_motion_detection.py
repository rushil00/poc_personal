import cv2
import numpy as np
import argparse
import os

from utils import configure_camera, draw_quadrilateral

class MotionDetection:
    def __init__(self, fps=30, regions=None):
        """
        Motion detection with an adaptive contour area threshold that only decays slowly.

        Parameters:
        no_motion_frame_limit (int): Number of consecutive frames without motion before setting motion status to False.
        """
        self.last_frame = None  # Store the last frame for comparison
        self.no_motion_frame_limit = fps * 1.5
        self.consecutive_no_motion_frames = 0
        # self.previous_motion_detected = False
        self.motion_detected = False
        self.motion_frame_count = 0
        self.fps = fps

        # Adaptive threshold parameters
        self.adaptive_threshold = 500.0  # Starting threshold value
        self.adaptive_multiplier = 1.5  # Multiplier to scale the average noise contour area
        self.increase_alpha = 0.25  # Fast update when the candidate threshold is higher
        self.decrease_alpha = 0.125  # Slow decay when the candidate threshold is lower
        self.regions_detected = {}  # To store detected regions and their counts
        self.motion_mask = None #np.ones(frame.shape[:2], dtype="uint8") * 255
        # Default region as the corners of the whole frame
        if regions is None:
            self.regions = [np.array([[0, 0], [640, 0], [640, 480], [0, 480]])]
        else:
            self.regions = regions

        self.region_motion_counts = {}  # Track motion counts per region
        self.current_active_region = None  # Store currently most active region
        self.region_labels = {i: f"Region {i+1}" for i in range(len(self.regions))}  # Add region labels

    def update_adaptive_threshold(self, areas):
        """
        Update the adaptive threshold using an exponential moving average strategy.
        """
        if not areas:
            return  # Nothing to update if no areas

        # Compute the candidate threshold from the current frame
        candidate_threshold = np.mean(areas) * self.adaptive_multiplier

        # If the candidate is greater than the current threshold, update quickly
        if candidate_threshold > self.adaptive_threshold:
            alpha = self.increase_alpha
        else:
            alpha = self.decrease_alpha

        # Update the adaptive threshold using EMA
        print("EMA", alpha)
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
                return  # Skip first frame
            
            if self.last_frame.shape != frame.shape:
                frame = cv2.resize(frame, (self.last_frame.shape[1], self.last_frame.shape[0]))

            # Convert frames to grayscale and apply Gaussian blur
            gray_last = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            gray_last = cv2.GaussianBlur(gray_last, (15, 15), 0)
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_current = cv2.GaussianBlur(gray_current, (15, 15), 0)
            
            # Compute difference between frames
            frame_difference = cv2.absdiff(gray_last, gray_current)
            _, threshold_image = cv2.threshold(frame_difference, 25, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            threshold_image = cv2.dilate(threshold_image, kernel, iterations=2)
            self.motion_mask = threshold_image
            # Find contours
            contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Debug: Show current adaptive threshold value
            print(f"Adaptive Threshold: {self.adaptive_threshold:.2f}")

            # 🔹 Filter contours using adaptive threshold
            filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= self.adaptive_threshold]
            print(f"Filtered contours count: {len(filtered_contours)}")

            # # Motion detection logic
            total_motion_area = sum(cv2.contourArea(c) for c in filtered_contours)

            noise_areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0]
            self.update_adaptive_threshold(noise_areas)

            if total_motion_area < self.adaptive_threshold * 5: # motion
                self.consecutive_no_motion_frames += 1
                if self.consecutive_no_motion_frames >= self.no_motion_frame_limit:
                    self.motion_detected = False
                    self.motion_frame_count = 0  # Reset motion frame count

                # 🔹 Update adaptive threshold with the areas from the current (no-motion) frame.
                # Optionally, you might filter out very small areas that are just sensor noise.
                # noise_areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0]
                # self.update_adaptive_threshold(noise_areas)

            else:  # No Motion 
                self.consecutive_no_motion_frames = 0  # Reset no-motion counter
                self.motion_frame_count += 1
                if self.motion_frame_count > 3:  # Require multiple frames of motion before confirming
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
        """
        Update motion detection with a new frame.
        
        Parameters:
        frame (np.ndarray): New frame to process.
        mask: Binary mask to apply.
        """

        # print(f"FPS: {fps}")
        # self.no_motion_frame_limit = fps * 1.5
        # self.previous_motion_detected = self.motion_detected
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Directly detect motion on the masked frame
        self.detect_motion(masked_frame)
        return self.motion_mask

    def process_frame(self, frame, fps):
        """
        Process a single frame for motion detection.
        Shows original frame with region tracings on left, full motion mask on right
        """
        # Create mask and process motion
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255
        motion_mask = self.update_motion_status(frame, mask, fps)
        
        # Draw regions on original frame
        frame_with_regions = frame.copy()
        
        # Draw all regions with their numbers
        for idx, region in enumerate(self.regions):
            # Draw region outline
            color = (0, 255, 0) if idx == self.current_active_region else (0, 165, 255)
            cv2.polylines(frame_with_regions, [region], True, color, 2)
            
            # Add region number
            # M = cv2.moments(region)
            # if M['m00'] != 0:
            #     cx = int(M['m10'] / M['m00'])
            #     cy = int(M['m01'] / M['m00'])
            #     cv2.putText(frame_with_regions, f"Region {idx+1}", (cx-20, cy), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Highlight active region
            if idx == self.current_active_region:
                overlay = frame_with_regions.copy()
                cv2.fillPoly(overlay, [region], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame_with_regions, 0.7, 0, frame_with_regions)

        # Convert motion mask to BGR for display
        if motion_mask is not None:
            motion_mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        else:
            motion_mask_bgr = np.zeros_like(frame)

        # Combine frames side by side
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

        # Read and display first frame for region selection
        ret, first_frame = cap.read()
        if not ret:
            print(f"Error: Could not read first frame from {videopath}")
            continue

        # Resize first frame for display
        first_frame = cv2.resize(first_frame, (640, 480))
        num = input("How many regions do you want?")
        regions = draw_quadrilateral(first_frame, int(num))
        print(f"REGIONS: {regions}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        motion_detector = MotionDetection(fps, regions)
        frame_count = 0

        # Reset video to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            frame_count += 1
            if frame_count % 1 == 0:  # Process every 10th frame
                processed_frame = motion_detector.process_frame(frame, fps)
                cv2.imshow("Motion Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("*"*80)
        print("*"*80)
        cap.release()
        cv2.destroyAllWindows()

def main(vidpath):
    # vidpath = "captures/videos/video0.mp4"
    # cap = cv2.VideoCapture(vidpath)
    cap = cv2.VideoCapture(vidpath)
    # cap = configure_camera(cap, width=1280, height=720, fps=90, codec="MJPG")    

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    num = input("How many regions do you want?\n")
    regions = draw_quadrilateral(first_frame, int(num))
    print(f"REGIONS: {regions}")
    motion_detector = MotionDetection(fps, regions)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 1 == 0:  # Process every frame
            processed_frame = motion_detector.process_frame(frame, fps)
            cv2.imshow("Region Detection", processed_frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    iterate_main('videos_representative')
    # main(0)
