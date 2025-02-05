import cv2
import numpy as np
import argparse

class MotionDetection:
    def __init__(self):
        """
        Motion detection with an adaptive contour area threshold that only decays slowly.
        
        Parameters:
        no_motion_frame_limit (int): Number of consecutive frames without motion before setting motion status to False.
        """
        self.last_frame = None  # Store the last frame for comparison
        self.no_motion_frame_limit = 30
        self.consecutive_no_motion_frames = 0
        self.previous_motion_detected = False
        self.motion_detected = False
        self.motion_frame_count = 0

        # Adaptive threshold parameters
        self.adaptive_threshold = 500.0  # Starting threshold value
        self.adaptive_multiplier = 1.5  # Multiplier to scale the average noise contour area
        self.increase_alpha = 0.25  # Fast update when the candidate threshold is higher
        self.decrease_alpha = 0.1  # Slow decay when the candidate threshold is lower
        self.regions_detected = {}  # To store detected regions and their counts
        self.motion_mask = None #np.ones(frame.shape[:2], dtype="uint8") * 255

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
            gray_last = cv2.GaussianBlur(gray_last, (21, 21), 0)
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)
            
            # Compute difference between frames
            frame_difference = cv2.absdiff(gray_last, gray_current)
            _, threshold_image = cv2.threshold(frame_difference, 25, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            threshold_image = cv2.dilate(threshold_image, kernel, iterations=2)
            self.motion_mask = threshold_image
            # Find contours
            contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Debug: Show current adaptive threshold value
            print(f"Adaptive Threshold: {self.adaptive_threshold:.2f}")

            # 🔹 Filter contours using adaptive threshold
            filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= self.adaptive_threshold]
            print(f"Filtered contours count: {len(filtered_contours)}")

            # Motion detection logic
            total_motion_area = sum(cv2.contourArea(c) for c in filtered_contours)
            if total_motion_area < self.adaptive_threshold * 5:
                self.consecutive_no_motion_frames += 1
                if self.consecutive_no_motion_frames >= self.no_motion_frame_limit:
                    self.motion_detected = False
                    self.motion_frame_count = 0  # Reset motion frame count

                # 🔹 Update adaptive threshold with the areas from the current (no-motion) frame.
                # Optionally, you might filter out very small areas that are just sensor noise.
                noise_areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0]
                self.update_adaptive_threshold(noise_areas)

            else:  # Motion detection based on contours count
                self.consecutive_no_motion_frames = 0  # Reset no-motion counter
                self.motion_frame_count += 1
                if self.motion_frame_count > 3:  # Require multiple frames of motion before confirming
                    self.motion_detected = True

            # Determine regions of motion
            height, width = frame.shape[:2]
            horizontal_step = height // 2
            vertical_step = width // 3
            self.regions_detected.clear()
            for contour in filtered_contours:
                if cv2.contourArea(contour) > self.adaptive_threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    mask_roi = self.motion_mask[y:y+h, x:x+w]
                    cluster_pixels = cv2.findNonZero(mask_roi)
                    if cluster_pixels is not None and len(cluster_pixels) > 200:
                        avg_x = int(cluster_pixels[:, 0, 0].mean()) + x
                        avg_y = int(cluster_pixels[:, 0, 1].mean()) + y
                        # Horizontal region (up/down)
                        h_region = 'up' if avg_y < horizontal_step else 'down'
                        # Vertical region (left/middle/right)
                        if avg_x < vertical_step:
                            v_region = 'left'
                        elif avg_x < 2 * vertical_step:
                            v_region = 'middle'
                        else:
                            v_region = 'right'
                        region = (h_region, v_region)
                        if region in self.regions_detected:
                            self.regions_detected[region] += 1
                        else:
                            self.regions_detected[region] = 1
            
            # Print detected regions
            if self.regions_detected:
                print(f"Motion detected in regions: {self.regions_detected}")
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

        print(f"FPS: {fps}")
        self.no_motion_frame_limit = fps * 1.5
        self.previous_motion_detected = self.motion_detected
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Directly detect motion on the masked frame
        self.detect_motion(masked_frame)
        return self.motion_mask


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Motion Detection")
    # parser.add_argument("--video", type=str, help="Path to the video file or camera index (0 for default camera)", default=0)
    # args = parser.parse_args()
    vidpath = "captures/videos/video0.mp4"
    # Initialize video capture
    cap = cv2.VideoCapture(vidpath)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Initialize motion detection
    motion_detector = MotionDetection()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:  # Process every 10th frame
            # Create a mask (for simplicity, using a full frame mask here)
            mask = np.ones(frame.shape[:2], dtype="uint8") * 255

            # Update motion status
            fps = cap.get(cv2.CAP_PROP_FPS)
            motion_mask = motion_detector.update_motion_status(frame, mask, fps)
            regions = motion_detector.regions_detected

            if motion_mask is not None:
                motion_mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                combined_frame = cv2.hconcat([frame, motion_mask_bgr])
                combined_frame = cv2.resize(combined_frame, (1280, 480))
                # Display region with maximum motion
                if regions:
                    max_region = max(regions, key=regions.get)
                    text = f"Max Motion Region: ({max_region[0]}, {max_region[1]})"
                    cv2.putText(combined_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Motion Detection", combined_frame)
            else:
                resized_frame = cv2.resize(frame, (1280, 480))
            
            # cv2.imshow("Motion Detection", resized_frame if resized_frame else combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
