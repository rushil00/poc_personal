import cv2
import numpy as np
import os

# class MotionDetection:
#     def __init__(self):
#         """
#         Motion detection using background subtraction (MOG2).
#         """
#         self.no_motion_frame_limit = 30
#         self.consecutive_no_motion_frames = 0
#         self.previous_motion_detected = False
#         self.motion_detected = False
#         self.motion_frame_count = 0

#         # Background subtractor
#         self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

#         # Adaptive threshold parameters
#         self.adaptive_threshold = 500.0  # Starting threshold value
#         self.adaptive_multiplier = 1.5  # Multiplier to scale the average noise contour area
#         self.increase_alpha = 0.25  # Fast update when the candidate threshold is higher
#         self.decrease_alpha = 0.1  # Slow decay when the candidate threshold is lower
#         self.regions_detected = {}  # To store detected regions and their counts
#         self.motion_mask = None

#     def update_adaptive_threshold(self, areas):
#         """
#         Update the adaptive threshold using an exponential moving average strategy.
#         """
#         if not areas:
#             return  # Nothing to update if no areas

#         # Compute the candidate threshold from the current frame
#         candidate_threshold = np.mean(areas) * self.adaptive_multiplier

#         # If the candidate is greater than the current threshold, update quickly
#         if candidate_threshold > self.adaptive_threshold:
#             alpha = self.increase_alpha
#         else:
#             alpha = self.decrease_alpha

#         # Update the adaptive threshold using EMA
#         self.adaptive_threshold = (1 - alpha) * self.adaptive_threshold + alpha * candidate_threshold
#         self.adaptive_threshold = max(500, self.adaptive_threshold)

#     def detect_motion(self, frame):
#         """
#         Detect motion using background subtraction.
        
#         Parameters:
#         frame (np.ndarray): Current frame.
#         """
#         try:
#             # Apply background subtraction
#             fgmask = self.fgbg.apply(frame)
            
#             # Thresholding to remove shadows (127 is shadow value)
#             _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            
#             # Noise removal using morphological operations
#             kernel = np.ones((5, 5), np.uint8)
#             thresh = cv2.erode(thresh, kernel, iterations=1)
#             thresh = cv2.dilate(thresh, kernel, iterations=2)
            
#             # Find contours of moving objects
#             contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             # Debug: Show current adaptive threshold value
#             print(f"Adaptive Threshold: {self.adaptive_threshold:.2f}")

#             # Filter contours using adaptive threshold
#             filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= self.adaptive_threshold]
#             print(f"Filtered contours count: {len(filtered_contours)}")

#             # Motion detection logic
#             total_motion_area = sum(cv2.contourArea(c) for c in filtered_contours)
#             if total_motion_area < self.adaptive_threshold * 5:
#                 self.consecutive_no_motion_frames += 1
#                 if self.consecutive_no_motion_frames >= self.no_motion_frame_limit:
#                     self.motion_detected = False
#                     self.motion_frame_count = 0  # Reset motion frame count

#                 # Update adaptive threshold with the areas from the current (no-motion) frame.
#                 noise_areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0]
#                 self.update_adaptive_threshold(noise_areas)

#             else:  # Motion detection based on contours count
#                 self.consecutive_no_motion_frames = 0  # Reset no-motion counter
#                 self.motion_frame_count += 1
#                 if self.motion_frame_count > 3:  # Require multiple frames of motion before confirming
#                     self.motion_detected = True

#             # Determine regions of motion
#             height, width = frame.shape[:2]
#             horizontal_step = height // 2
#             vertical_step = width // 3
#             self.regions_detected.clear()
#             for contour in filtered_contours:
#                 if cv2.contourArea(contour) > self.adaptive_threshold:
#                     x, y, w, h = cv2.boundingRect(contour)
#                     mask_roi = thresh[y:y+h, x:x+w]
#                     cluster_pixels = cv2.findNonZero(mask_roi)
#                     if cluster_pixels is not None and len(cluster_pixels) > 200:
#                         avg_x = int(cluster_pixels[:, 0, 0].mean()) + x
#                         avg_y = int(cluster_pixels[:, 0, 1].mean()) + y
#                         # Horizontal region (up/down)
#                         h_region = 'top' if avg_y < horizontal_step else 'bottom'
#                         # Vertical region (left/middle/right)
#                         if avg_x < vertical_step:
#                             v_region = 'left'
#                         elif avg_x < 2 * vertical_step:
#                             v_region = 'middle'
#                         else:
#                             v_region = 'right'
#                         region = (h_region, v_region)
#                         self.regions_detected[region] = self.regions_detected.get(region, 0) + 1

#             # Set motion_detected based on regions_detected
#             if self.regions_detected:
#                 self.motion_detected = True
#             else:
#                 self.motion_detected = False

#             # Print detected regions
#             if self.regions_detected:
#                 print(f"Motion detected in regions: {self.regions_detected}")

#             # Store motion mask for visualization
#             self.motion_mask = thresh

#         except Exception as e:
#             print(f"Motion detection failed due to: {e}")
#             raise e

#     def update_motion_status(self, frame, mask, fps):
#         """
#         Update motion detection with a new frame.
        
#         Parameters:
#         frame (np.ndarray): New frame to process.
#         mask: Binary mask to apply.
#         """
#         print(f"FPS: {fps}")
#         self.no_motion_frame_limit = fps * 1.5
#         self.previous_motion_detected = self.motion_detected
#         masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
#         # Directly detect motion on the masked frame
#         self.detect_motion(masked_frame)
#         return self.motion_mask

#     def process_frame(self, frame, fps):
#         """
#         Process a single frame for motion detection.
        
#         Parameters:
#         frame (np.ndarray): Frame to process.
#         fps (float): Frames per second of the video.
#         """
#         H, W = frame.shape[:2]
#         # Draw intersecting lines to divide the regions
#         cv2.line(frame, (W // 3, 0), (W // 3, H), (255, 0, 0), 2)
#         cv2.line(frame, (2 * W // 3, 0), (2 * W // 3, H), (255, 0, 0), 2)
#         cv2.line(frame, (0, H // 2), (W, H // 2), (255, 0, 0), 2)

#         # Create a mask (for simplicity, using a full frame mask here)
#         mask = np.ones(frame.shape[:2], dtype="uint8") * 255

#         # Update motion status
#         motion_mask = self.update_motion_status(frame, mask, fps)
#         regions = self.regions_detected
#         color = (0, 0, 255) if not self.motion_detected else (50, 255, 50)
#         if motion_mask is not None:
#             motion_mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
#             combined_frame = cv2.hconcat([frame, motion_mask_bgr])
#             combined_frame = cv2.resize(combined_frame, (1280, 480))
#             # Display region with maximum motion
#             if regions:
#                 max_region = max(regions, key=regions.get)
#                 text = f"Max Motion Region: ({max_region[0]}, {max_region[1]})"
#             return combined_frame
#         else:
#             resized_frame = cv2.resize(frame, (1280, 480))
#             return resized_frame




class MotionDetection2:
    def __init__(self):
        """
        Motion detection using MOG2 background subtraction with adaptive thresholding
        """
        # Background subtractor with optimized parameters
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        # Adaptive threshold parameters
        self.adaptive_threshold = 500.0  # Starting threshold value
        self.adaptive_multiplier = 1.5
        self.increase_alpha = 0.25
        self.decrease_alpha = 0.1
        
        # Motion tracking parameters
        self.no_motion_frame_limit = 30
        self.consecutive_no_motion_frames = 0
        self.motion_detected = False
        self.motion_frame_count = 0
        self.regions_detected = {}
        self.motion_mask = None
        
        # Morphological kernels
        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    
    def update_motion_status(self, frame, mask, fps):
        """
        Update motion detection with a new frame.
        
        Parameters:
        frame (np.ndarray): New frame to process.
        mask: Binary mask to apply.
        fps: Frames per second of video
        """
        self.no_motion_frame_limit = int(fps * 1.5)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        self.detect_motion(masked_frame)
        return self.motion_mask
    

    def update_adaptive_threshold(self, areas):
        """Update adaptive threshold using exponential moving average"""
        if not areas:
            return

        candidate_threshold = np.mean(areas) * self.adaptive_multiplier
        alpha = self.increase_alpha if candidate_threshold > self.adaptive_threshold else self.decrease_alpha
        self.adaptive_threshold = (1 - alpha) * self.adaptive_threshold + alpha * candidate_threshold
        self.adaptive_threshold = max(500, self.adaptive_threshold)

    def detect_motion(self, frame):
        """Detect motion using background subtraction"""
        try:
            # Apply background subtraction
            fgmask = self.fgbg.apply(frame)
            
            # Process mask
            _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, self.erode_kernel, iterations=1)
            thresh = cv2.dilate(thresh, self.dilate_kernel, iterations=2)
            
            self.motion_mask = thresh
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            filtered_contours = []
            noise_areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Ignore very small areas
                    noise_areas.append(area)
                    if area >= self.adaptive_threshold:
                        filtered_contours.append(contour)
            
            # Update adaptive threshold
            self.update_adaptive_threshold(noise_areas)
            
            # Motion status logic
            total_motion_area = sum(cv2.contourArea(c) for c in filtered_contours)
            if total_motion_area < self.adaptive_threshold * 5:
                self.consecutive_no_motion_frames += 1
                if self.consecutive_no_motion_frames >= self.no_motion_frame_limit:
                    self.motion_detected = False
                    self.motion_frame_count = 0
            else:
                self.consecutive_no_motion_frames = 0
                self.motion_frame_count += 1
                self.motion_detected = self.motion_frame_count > 3

            # Detect regions
            self.regions_detected.clear()
            if filtered_contours:
                height, width = frame.shape[:2]
                h_step = height // 2
                v_step = width // 3
                
                for contour in filtered_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    mask_roi = self.motion_mask[y:y+h, x:x+w]
                    
                    if cv2.countNonZero(mask_roi) > 200:
                        M = cv2.moments(mask_roi)
                        if M["m00"] != 0:
                            avg_x = int(M["m10"] / M["m00"]) + x
                            avg_y = int(M["m01"] / M["m00"]) + y
                            
                            h_region = 'top' if avg_y < h_step else 'bottom'
                            v_region = 'left' if avg_x < v_step else 'middle' if avg_x < 2*v_step else 'right'
                            
                            region = (h_region, v_region)
                            self.regions_detected[region] = self.regions_detected.get(region, 0) + 1

            # Final motion status
            self.motion_detected = bool(self.regions_detected)

        except Exception as e:
            print(f"Motion detection error: {e}")
            raise

    def process_frame(self, frame, fps):
        """Process frame and return visualization"""
        self.no_motion_frame_limit = int(fps * 1.5)
        self.detect_motion(frame)
        
        # Visualization
        H, W = frame.shape[:2]
        cv2.line(frame, (W//3, 0), (W//3, H), (255,0,0), 2)
        cv2.line(frame, (2*W//3, 0), (2*W//3, H), (255,0,0), 2)
        cv2.line(frame, (0, H//2), (W, H//2), (255,0,0), 2)

        # Create output visualization
        if self.motion_mask is not None:
            motion_mask_bgr = cv2.cvtColor(self.motion_mask, cv2.COLOR_GRAY2BGR)
            combined = cv2.hconcat([frame, motion_mask_bgr])
            combined = cv2.resize(combined, (1280, 480))
            
            # Add status text
            # status_color = (0, 255, 0) if self.motion_detected else (0, 0, 255)
            # status_text = "Motion Detected" if self.motion_detected else "No Motion"
            # cv2.putText(combined, status_text, (10, 30), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # # Add region info
            # if self.regions_detected:
            #     main_region = max(self.regions_detected, key=self.regions_detected.get)
            #     cv2.putText(combined, f"Main Area: {main_region[0]} {main_region[1]}", (10, 70),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            return combined
        
        return frame



def iterate_main(main_dir='captures/videos'):
    video_files = [f for f in os.listdir(main_dir) if f.endswith('.mp4')]
    for videopath in video_files[-5:]:
        videopath = os.path.join(main_dir, videopath)
        cap = cv2.VideoCapture(videopath)
        if not cap.isOpened():
            print(f"Error: Could not open video {videopath}.")
            continue

        motion_detector = MotionDetection2()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 == 0:  # Process every 10th frame
                processed_frame = motion_detector.process_frame(frame, fps)
                # Add status text
                status_color = (0, 255, 0) if motion_detector.motion_detected else (0, 0, 255)
                status_text = "Motion Detected" if motion_detector.motion_detected else "No Motion"
                cv2.putText(processed_frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                # Add region info
                if motion_detector.regions_detected:
                    main_region = max(motion_detector.regions_detected, key=motion_detector.regions_detected.get)
                    cv2.putText(processed_frame, f"Main Area: {main_region[0]} {main_region[1]}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(motion_detector.regions_detected)
                cv2.imshow("Motion Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("*"*60)
        print("*"*60)
        cap.release()
        cv2.destroyAllWindows()



def main():
    vidpath = "captures/videos/video0.mp4"
    motion_detector = MotionDetection2()

    cap = cv2.VideoCapture(0)
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
                # Add status text
                status_color = (0, 255, 0) if motion_detector.motion_detected else (0, 0, 255)
                status_text = "Motion Detected" if motion_detector.motion_detected else "No Motion"
                cv2.putText(processed_frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                # Add region info
                if motion_detector.regions_detected:
                    main_region = max(motion_detector.regions_detected, key=motion_detector.regions_detected.get)
                    cv2.putText(processed_frame, f"Main Area: {main_region[0]} {main_region[1]}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                    cv2.imshow("Motion Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    iterate_main('videos_new')
    # main()