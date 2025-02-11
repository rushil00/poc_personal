import os
import cv2
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from adaptive_motion_detection import MotionDetection
from utils import draw_quadrilateral
from sparse_optical_flow import MotionDetectionSparse
from background_subtraction import MotionDetection2



class ConsensusMotionDetection:
    def __init__(self, fps=30, regions=None, queue_len=10, max_workers=1):
        # Default region as the corners of the whole frame if none provided
        if regions is None:
            self.regions = [np.array([[0, 0], [640, 0], [640, 480], [0, 480]])]
        else:
            self.regions = regions

        # Initialize detectors with regions
        self.motion_detector_contour = MotionDetection(fps, self.regions)
        self.motion_detector_sparse = MotionDetectionSparse(fps, self.regions)
        
        self.motion_detected = False
        self.regions_detected = {}
        self.region_motion_counts = {i: 0 for i in range(len(self.regions))}
        self.current_active_region = None
        self.region_labels = {i: f"Region {i+1}" for i in range(len(self.regions))}
        self.motion_mask = None
        self.frameQueue = deque(maxlen=queue_len)
        self.frame_counter = 0
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def update_motion_status(self, frame, mask, fps):
        """
        Update motion status using both detectors and create a consensus.
        """
        self.frameQueue.append(frame)
        self.frame_counter += 1
        if self.frame_counter % 15 == 0 and len(self.frameQueue) > 1:
            frame1 = self.frameQueue.popleft()
            frame2 = self.frameQueue.popleft()
            future_contour = self.executor.submit(self.motion_detector_contour.update_motion_status, frame1, mask, fps)
            future_sparse = self.executor.submit(self.motion_detector_sparse.update_motion_status, frame2, mask, fps)
            mask_contour = future_contour.result()
            mask_sparse = future_sparse.result()

            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]

            # Convert sparse mask from BGR to grayscale if needed
            if mask_sparse is not None:
                if len(mask_sparse.shape) == 3:
                    mask_sparse = cv2.cvtColor(mask_sparse, cv2.COLOR_BGR2GRAY)
                mask_sparse = cv2.resize(mask_sparse, (frame_width, frame_height))
            else:
                mask_sparse = np.zeros((frame_height, frame_width), dtype=np.uint8)

            # Convert contour mask
            if mask_contour is not None:
                if len(mask_contour.shape) == 3:
                    mask_contour = cv2.cvtColor(mask_contour, cv2.COLOR_BGR2GRAY)
                mask_contour = cv2.resize(mask_contour, (frame_width, frame_height))
            else:
                mask_contour = np.zeros((frame_height, frame_width), dtype=np.uint8)

            # Normalize masks to ensure binary values
            _, mask_contour = cv2.threshold(mask_contour, 1, 255, cv2.THRESH_TRIANGLE)
            _, mask_sparse = cv2.threshold(mask_sparse, 1, 255, cv2.THRESH_TRIANGLE)

            # Combine masks
            self.motion_mask = cv2.bitwise_or(mask_contour, mask_sparse)

            # Reset region motion counts
            self.region_motion_counts = {i: 0 for i in range(len(self.regions))}

            # Combine motion detection results
            self.motion_detected = (
                self.motion_detector_contour.motion_detected 
                or self.motion_detector_sparse.motion_detected
            )

            # Update region motion counts from both detectors
            if self.motion_detected:
                # Add counts from contour detector
                if hasattr(self.motion_detector_contour, 'region_motion_counts'):
                    for region_idx, count in self.motion_detector_contour.region_motion_counts.items():
                        self.region_motion_counts[region_idx] += count

                # Add counts from sparse detector
                if hasattr(self.motion_detector_sparse, 'region_motion_counts'):
                    for region_idx, count in self.motion_detector_sparse.region_motion_counts.items():
                        self.region_motion_counts[region_idx] += count

                # Determine most active region
                if self.region_motion_counts:
                    max_activity_region = max(self.region_motion_counts.items(), key=lambda x: x[1])
                    if max_activity_region[1] > 0:
                        self.current_active_region = max_activity_region[0]
                        print(f"Consensus: Most motion in Region {self.current_active_region + 1}")
                    else:
                        self.current_active_region = None
            else:
                self.current_active_region = None
                self.region_motion_counts = {i: 0 for i in range(len(self.regions))}

            print('-'*20)
            print("Region motion counts:", self.region_motion_counts)
            print("Current active region:", self.current_active_region)
            print('-'*20)

    def process_frame(self, frame, fps):
        """
        Process a single frame for motion detection using consensus.
        Shows original frame with region tracings and motion visualization.
        """
        # Draw regions on original frame
        frame_with_regions = frame.copy()
        
        # Draw all regions with their numbers and motion counts
        for idx, region in enumerate(self.regions):
            # Determine color based on activity
            if idx == self.current_active_region:
                color = (0, 255, 0)  # Green for most active
            else:
                # Scale color based on motion count
                motion_count = self.region_motion_counts.get(idx, 0)
                if motion_count > 0:
                    intensity = min(255, motion_count * 50)  # Scale the intensity
                    color = (0, intensity, intensity)  # Yellow-ish for active
                else:
                    color = (0, 165, 255)  # Orange for inactive

            # Draw region outline
            cv2.polylines(frame_with_regions, [region], True, color, 2)
            
            # Add region number and motion count
            M = cv2.moments(region)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(frame_with_regions, f"R{idx+1}: {self.region_motion_counts.get(idx, 0)}", 
                           (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Highlight active region
            if idx == self.current_active_region:
                overlay = frame_with_regions.copy()
                cv2.fillPoly(overlay, [region], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame_with_regions, 0.7, 0, frame_with_regions)

        # Create mask and update motion status
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255
        self.update_motion_status(frame, mask, fps)

        # Combine frame with motion mask
        if self.motion_mask is not None:
            motion_mask_bgr = cv2.cvtColor(self.motion_mask, cv2.COLOR_GRAY2BGR)
            combined_frame = cv2.hconcat([frame_with_regions, motion_mask_bgr])
            combined_frame = cv2.resize(combined_frame, (1280, 480))

            # Add motion information overlay if motion detected
            if self.current_active_region is not None:
                motion_text = f"Motion in Region {self.current_active_region + 1}"
                cv2.putText(combined_frame, motion_text, (40, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show consensus information
                consensus_text = "Consensus: Both Detectors Agree" if (
                    self.motion_detector_contour.current_active_region == 
                    self.motion_detector_sparse.current_active_region) else "Detectors Disagree"
                cv2.putText(combined_frame, consensus_text, (40, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            return combined_frame
        else:
            return frame_with_regions


def iterate_main(main_dir='captures/videos'):
    video_files = [f for f in os.listdir(main_dir) if f.endswith('.mp4')]
    for videopath in video_files:
        videopath = os.path.join(main_dir, videopath)
        cap = cv2.VideoCapture(videopath)
        if not cap.isOpened():
            print(f"Error: Could not open video {videopath}.")
            continue

        motion_detector = ConsensusMotionDetection()
        fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = motion_detector.process_frame(frame, fps)
            if motion_detector.regions_detected:
                most_motion_region = max(motion_detector.regions_detected, key=motion_detector.regions_detected.get)
                regions_text = f"Most motion in: {most_motion_region}"
                cv2.putText(processed_frame, regions_text, (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
            # Display motion status and regions
            color = (0, 0, 255) if not motion_detector.motion_detected else (50, 255, 50)
            cv2.putText(processed_frame, "No Motion" if not motion_detector.motion_detected else "Motion", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)

            cv2.imshow("Motion Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        print('*'*60)
        print('*'*60)
        cv2.destroyAllWindows()

def decode_fourcc(value):
    """Decode the FourCC codec value."""
    return "".join([chr((value >> 8 * i) & 0xFF) for i in range(4)])

def configure_camera(cap, width=1280, height=720, fps=90, codec="MJPG"):
    """Configure the camera with resolution, FPS, and codec."""
    if not cap or not cap.isOpened():
        return None

    fourcc = cv2.VideoWriter_fourcc(*codec)
    old_fourcc = decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))

    if cap.set(cv2.CAP_PROP_FOURCC, fourcc):
        print(f"Codec changed from {old_fourcc} to {decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))}")
    else:
        print(f"Error: Could not change codec from {old_fourcc}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    print(f"Camera configured with FPS: {cap.get(cv2.CAP_PROP_FPS)}, "
          f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, "
          f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    return cap
  
def main(vidpath=0):
    cap = cv2.VideoCapture(vidpath)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get first frame for region selection
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    # Get user input for regions
    num = input("How many regions do you want?\n")
    regions = draw_quadrilateral(first_frame, int(num))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize consensus detector with regions
    motion_detector = ConsensusMotionDetection(fps=fps, regions=regions)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = motion_detector.process_frame(frame, fps)
        cv2.imshow("Consensus Motion Detection", processed_frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    # main(0)