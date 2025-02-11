import cv2
import numpy as np
import os
from utils import configure_camera, draw_quadrilateral

class MotionDetectionSparse:
    def __init__(self, fps=30, regions=None):
        """
        Motion detection using sparse optical flow (Lucas-Kanade method).
        """
        self.last_frame = None  # Store the last frame for comparison
        self.no_motion_frame_limit = 30
        self.consecutive_no_motion_frames = 0
        self.previous_motion_detected = False
        self.motion_detected = False
        self.motion_frame_count = 0

        # Parameters for sparse optical flow
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.track_points = None  # Points to track
        self.track_colors = np.random.randint(0, 255, (100, 3))  # Random colors for visualization

        # Adaptive threshold parameters
        self.adaptive_threshold = 5.3  # Starting threshold for motion magnitude
        self.increase_alpha = 0.25  # Fast update when the candidate threshold is higher
        self.decrease_alpha = 0.1  # Slow decay when the candidate threshold is lower
        self.regions_detected = {}  # To store detected regions and their counts
        self.motion_mask = None  # Motion mask for visualization
        self.motion_magnitudes = None

        # Default region as the corners of the whole frame
        if regions is None:
            self.regions = [np.array([[0, 0], [640, 0], [640, 480], [0, 480]])]
        else:
            self.regions = regions

        self.region_motion_counts = {}  # Track motion counts per region
        self.current_active_region = None  # Store currently most active region
        self.region_labels = {i: f"Region {i+1}" for i in range(len(self.regions))}  # Add region labels

    def update_adaptive_threshold(self, motion_magnitudes):
        """
        Update the adaptive threshold using a more robust statistical approach.
        """
        if motion_magnitudes is None or len(motion_magnitudes) == 0:
            return

        # Calculate statistical measures
        mean_magnitude = np.mean(motion_magnitudes)
        std_magnitude = np.std(motion_magnitudes)
        
        # Use median and percentile for more robust thresholding
        median_magnitude = np.median(motion_magnitudes)
        percentile_75 = np.percentile(motion_magnitudes, 75)
        
        # Compute dynamic multiplier based on motion variance
        dynamic_multiplier = 1.0 + (std_magnitude / (mean_magnitude + 1e-6))
        
        # Calculate new threshold using multiple factors
        candidate_threshold = median_magnitude * dynamic_multiplier
        
        # Add noise resistance by considering the 75th percentile
        candidate_threshold = max(candidate_threshold, percentile_75 * 0.5)
        
        # Smooth threshold update with dynamic learning rate
        alpha = self.increase_alpha if candidate_threshold > self.adaptive_threshold else self.decrease_alpha
        alpha = min(alpha * (1 + std_magnitude/10), 0.5)  # Adjust learning rate based on motion variance
        
        # Update threshold with bounds
        # self.adaptive_threshold = (1 - alpha) * self.adaptive_threshold + alpha * candidate_threshold
        # self.adaptive_threshold = np.clip(self.adaptive_threshold, 0.5, 50.0)  # Set reasonable bounds

    def detect_motion(self, frame):
        """
        Detect motion using sparse optical flow.
        """
        try:
            if self.last_frame is None:
                self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.track_points = cv2.goodFeaturesToTrack(self.last_frame, mask=None, **self.feature_params)
                return  # Skip first frame
            # Apply Gaussian blur to the last frame and the current frame
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(self.last_frame, gray_current, self.track_points, None, **self.lk_params)

            # Select good points
            good_new = new_points[status == 1]
            good_old = self.track_points[status == 1]

            # Compute motion magnitudes
            self.motion_magnitudes = np.linalg.norm(good_new - good_old, axis=1)

            # Update adaptive threshold
            self.update_adaptive_threshold(self.motion_magnitudes)

            # Filter motion vectors based on adaptive threshold
            significant_motion = self.motion_magnitudes > self.adaptive_threshold
            significant_points = good_new[significant_motion]
            significant_magnitudes = self.motion_magnitudes[significant_motion]

            # Reset region motion counts
            self.region_motion_counts = {i: 0 for i in range(len(self.regions))}
            self.regions_detected.clear()

            # Update motion detection status
            if len(significant_points) > 0:
                self.motion_detected = True
                self.consecutive_no_motion_frames = 0

                # Check each point against defined regions
                for point, magnitude in zip(significant_points, significant_magnitudes):
                    x, y = point.ravel()
                    point_in_region = False
                    
                    # Check which region contains this point
                    for region_idx, region in enumerate(self.regions):
                        if cv2.pointPolygonTest(region, (x, y), False) >= 0:
                            self.region_motion_counts[region_idx] += 1
                            point_in_region = True
                            
                    # Only store point if it's in a defined region
                    if point_in_region:
                        print(f"Motion in region with magnitude: {magnitude:.2f}")

                # Determine most active region
                if self.region_motion_counts:
                    max_activity_region = max(self.region_motion_counts.items(), key=lambda x: x[1])
                    if max_activity_region[1] > 0:  # If there's any motion
                        self.current_active_region = max_activity_region[0]
                        print(f"Most motion in {self.region_labels[self.current_active_region]}")
                    else:
                        self.current_active_region = None
            else:
                self.consecutive_no_motion_frames += 1
                if self.consecutive_no_motion_frames >= self.no_motion_frame_limit:
                    self.motion_detected = False
                    self.regions_detected = {}
                self.current_active_region = None

            # Update track points for the next frame
            self.track_points = cv2.goodFeaturesToTrack(gray_current, mask=None, **self.feature_params)
            self.last_frame = gray_current

            # Visualize motion vectors
            self.motion_mask = np.zeros_like(frame)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.motion_mask = cv2.line(self.motion_mask, (int(a), int(b)), (int(c), int(d)), self.track_colors[i].tolist(), 2)
                self.motion_mask = cv2.circle(self.motion_mask, (int(a), int(b)), 5, self.track_colors[i].tolist(), -1)

        except Exception as e:
            print(f"Motion detection failed due to: {e}")
            raise e

    def update_motion_status(self, frame, mask, fps):
        """
        Update motion detection with a new frame.
        """
        self.no_motion_frame_limit = fps * 1.5
        self.previous_motion_detected = self.motion_detected
        # Apply Gaussian blur to the frame
        frame = cv2.GaussianBlur(frame, (19, 19), 0)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Directly detect motion on the masked frame
        self.detect_motion(masked_frame)
        return self.motion_mask

    def process_frame(self, frame, fps):
        """
        Process a single frame for motion detection.
        Shows original frame with region tracings on left, motion vectors on right
        """
        # Create blank frame for motion visualization
        motion_viz = np.zeros_like(frame)
        
        # Draw the regions and handle motion detection
        frame_with_regions = frame.copy()
        
        # Draw all regions with their numbers
        for idx, region in enumerate(self.regions):
            # Draw region outline
            color = (0, 255, 0) if idx == self.current_active_region else (0, 165, 255)
            cv2.polylines(frame_with_regions, [region], True, color, 2)
            
            # Add region number
            M = cv2.moments(region)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(frame_with_regions, f"Region {idx+1}", (cx-20, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Highlight active region
            if idx == self.current_active_region:
                overlay = frame_with_regions.copy()
                cv2.fillPoly(overlay, [region], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame_with_regions, 0.7, 0, frame_with_regions)

        # Create mask and process motion
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255
        self.update_motion_status(frame, mask, fps)

        # Draw motion vectors on motion visualization
        if (self.track_points is not None and self.motion_magnitudes is not None and 
            len(self.track_points) > 0 and len(self.motion_magnitudes) > 0):
            for i, (point, magnitude) in enumerate(zip(self.track_points, self.motion_magnitudes)):
                x, y = point.ravel()
                if magnitude > self.adaptive_threshold:
                    color = self.track_colors[i % len(self.track_colors)].tolist()
                    motion_viz = cv2.circle(motion_viz, (int(x), int(y)), 5, color, -1)

        # Combine frames side by side
        combined_frame = cv2.hconcat([frame_with_regions, motion_viz])
        return cv2.resize(combined_frame, (1280, 480))


def iterate_main(main_dir='captures/videos'):
    video_files = [f for f in os.listdir(main_dir) if f.endswith('.mp4')]
    for videopath in video_files[-9:]:
        videopath = os.path.join(main_dir, videopath)
        cap = cv2.VideoCapture(videopath)
        if not cap.isOpened():
            print(f"Error: Could not open video {videopath}.")
            continue

        motion_detector = MotionDetectionSparse()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 1 == 0:  # Process every 10th frame
                processed_frame = motion_detector.process_frame(frame, fps)
                color = (0, 0, 255) if not motion_detector.motion_detected else (50, 255, 50)
                cv2.putText(processed_frame, "No Motion" if not motion_detector.motion_detected else "Motion", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
                processed_frame = cv2.resize(processed_frame,(640,480))

                # Display motion regions on the frame
                if motion_detector.regions_detected:
                    most_motion_region = max(motion_detector.regions_detected, key=motion_detector.regions_detected.get)
                    regions_text = f"Most motion in: {most_motion_region}"
                    cv2.putText(processed_frame, regions_text, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Display magnitudes of motion points
                if motion_detector.motion_detected:
                    for point, magnitude in zip(motion_detector.track_points, motion_detector.motion_magnitudes):
                        x, y = point.ravel()
                        cv2.putText(processed_frame, f"{magnitude:.2f}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow("Motion Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    num = input("How many regions do you want?\n")
    regions = draw_quadrilateral(first_frame, int(num))
    fps = cap.get(cv2.CAP_PROP_FPS)
    motion_detector = MotionDetectionSparse(fps, regions)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 1 == 0:  # Process every frame
            processed_frame = motion_detector.process_frame(frame, fps)
            
            # Add motion information overlay
            if motion_detector.current_active_region is not None:
                region_num = motion_detector.current_active_region
                motion_text = f"Motion detected in Region {region_num + 1}"
                cv2.putText(processed_frame, motion_text, (40, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show motion counts for active region
                count_text = f"Motion count: {motion_detector.region_motion_counts[region_num]}"
                cv2.putText(processed_frame, count_text, (40, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Optical Flow Detection", processed_frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # iterate_main('videos_new')
    main()
