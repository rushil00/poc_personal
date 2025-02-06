import cv2
import numpy as np
import os

class MotionDetectionSparse:
    def __init__(self):
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
        self.adaptive_threshold = 10.0  # Starting threshold for motion magnitude
        self.adaptive_multiplier = 1.5  # Multiplier to scale the average motion magnitude
        self.increase_alpha = 0.25  # Fast update when the candidate threshold is higher
        self.decrease_alpha = 0.1  # Slow decay when the candidate threshold is lower
        self.regions_detected = {}  # To store detected regions and their counts
        self.motion_mask = None  # Motion mask for visualization
        self.motion_magnitudes = None

    def update_adaptive_threshold(self, motion_magnitudes):
        """
        Update the adaptive threshold using an exponential moving average strategy.
        """
        if motion_magnitudes is None:
            return  # Nothing to update if no motion magnitudes

        # Compute the candidate threshold from the current frame
        candidate_threshold = np.mean(motion_magnitudes) * self.adaptive_multiplier

        # If the candidate is greater than the current threshold, update quickly
        if candidate_threshold > self.adaptive_threshold:
            alpha = self.increase_alpha
        else:
            alpha = self.decrease_alpha

        # Update the adaptive threshold using EMA
        self.adaptive_threshold = (1 - alpha) * self.adaptive_threshold + alpha * candidate_threshold
        self.adaptive_threshold = max(10, self.adaptive_threshold)  # Ensure a minimum threshold

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

            # Update motion detection status
            if len(significant_points) > 0:
                self.motion_detected = True
                self.consecutive_no_motion_frames = 0
                # Print the coordinates and magnitudes of the pixels where motion is detected
                for point, magnitude in zip(significant_points, significant_magnitudes):
                    print(f"Motion detected at pixel: {point}, Magnitude: {magnitude}")

                # Determine the region of motion
                H, W = frame.shape[:2]
                regions = {}
                for point in significant_points:
                    x, y = point.ravel()
                    if y < H // 2:
                        h_region = 'top'
                    else:
                        h_region = 'bottom'
                    if x < W // 3:
                        v_region = 'left'
                    elif x < 2 * W // 3:
                        v_region = 'middle'
                    else:
                        v_region = 'right'
                    region = (h_region, v_region)
                    if region in regions:
                        regions[region] += 1
                    else:
                        regions[region] = 1

                self.regions_detected = regions
                print("Motion detected in regions:", self.regions_detected)

            else:
                self.consecutive_no_motion_frames += 1
                if self.consecutive_no_motion_frames >= self.no_motion_frame_limit:
                    self.motion_detected = False
                    self.regions_detected = {}

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
        """
        H, W = frame.shape[:2]
        # Draw intersecting lines to divide the regions
        cv2.line(frame, (W // 3, 0), (W // 3, H), (255, 0, 0), 2)
        cv2.line(frame, (2 * W // 3, 0), (2 * W // 3, H), (255, 0, 0), 2)
        cv2.line(frame, (0, H // 2), (W, H // 2), (255, 0, 0), 2)

        # Create a mask (for simplicity, using a full frame mask here)
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255

        # Update motion status
        motion_mask = self.update_motion_status(frame, mask, fps)
        color = (0, 0, 255) if not self.motion_detected else (50, 255, 50)
        cv2.putText(frame, "No Motion" if not self.motion_detected else "Motion", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)

        # Display motion regions on the frame
        if self.regions_detected:
            most_motion_region = max(self.regions_detected, key=self.regions_detected.get)
            regions_text = f"Most motion in: {most_motion_region}"
            cv2.putText(frame, regions_text, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display magnitudes of motion points
        if self.motion_detected:
            for point, magnitude in zip(self.track_points, self.motion_magnitudes):
                x, y = point.ravel()
                cv2.putText(frame, f"{magnitude:.2f}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if motion_mask is not None:
            combined_frame = cv2.addWeighted(frame, 0.8, motion_mask, 1, 0)
            return combined_frame
        else:
            return frame


def iterate_main(main_dir='captures/videos'):
    video_files = [f for f in os.listdir(main_dir) if f.endswith('.mp4')]
    for videopath in video_files[-2:]:
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
            if frame_count % 10 == 0:  # Process every 10th frame
                processed_frame = motion_detector.process_frame(frame, fps)
                cv2.imshow("Motion Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    vidpath = "captures/videos/video0.mp4"
    motion_detector = MotionDetectionSparse()

    cap = cv2.VideoCapture(vidpath)
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 1 == 0:  # Process every frame
                processed_frame = motion_detector.process_frame(frame, fps)
                cv2.imshow("Motion Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    iterate_main('videos_representative')
    # main()
