import os
import cv2
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from adaptive_motion_detection import MotionDetection
from sparse_optical_flow import MotionDetectionSparse


class ConsensusMotionDetection:
    def __init__(self, queue_len=10, max_workers=1):
        self.motion_detector_contour = MotionDetection()
        self.motion_detector_sparse = MotionDetectionSparse()
        self.motion_detected = False
        self.regions_detected = {}
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
        if self.frame_counter % 10 == 0 and len(self.frameQueue) > 1:
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

            # Combine motion status (logical OR)
            self.motion_detected = (
                self.motion_detector_contour.motion_detected or
                self.motion_detector_sparse.motion_detected
            )

            if not self.motion_detected:
                self.regions_detected.clear()

            # Combine regions detected (union of regions)
            regions_contour = self.motion_detector_contour.regions_detected
            regions_sparse = self.motion_detector_sparse.regions_detected

            # Combine regions detected (union of regions with summed values for common keys)
            self.regions_detected = regions_contour.copy()
            for key, value in regions_sparse.items():
                if key in self.regions_detected:
                    self.regions_detected[key] += value
                else:
                    self.regions_detected[key] = value

            print("CONSENSUS REGIONS", self.regions_detected)


    def process_frame(self, frame, fps):
        """
        Process a single frame for motion detection using consensus.
        """
        H, W = frame.shape[:2]
        # Draw intersecting lines to divide the regions
        cv2.line(frame, (W // 3, 0), (W // 3, H), (255, 0, 0), 2)
        cv2.line(frame, (2 * W // 3, 0), (2 * W // 3, H), (255, 0, 0), 2)
        cv2.line(frame, (0, H // 2), (W, H // 2), (255, 0, 0), 2)

        # Create a mask (for simplicity, using a full frame mask here)
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255

        # Update motion status using consensus
        self.update_motion_status(frame, mask, fps)

        # Display motion status and regions
        color = (0, 0, 255) if not self.motion_detected else (50, 255, 50)
        cv2.putText(frame, "No Motion" if not self.motion_detected else "Motion", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)

        if self.regions_detected:
            most_motion_region = max(self.regions_detected, key=self.regions_detected.get)
            regions_text = f"Most motion in: {most_motion_region}"
            cv2.putText(frame, regions_text, (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)

        # Combine frame with motion mask
        if self.motion_mask is not None:
            motion_mask_bgr = cv2.cvtColor(self.motion_mask, cv2.COLOR_GRAY2BGR)
            combined_frame = cv2.hconcat([frame, motion_mask_bgr])
            combined_frame = cv2.resize(combined_frame, (1280, 480))
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

        motion_detector = ConsensusMotionDetection()
        fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = motion_detector.process_frame(frame, fps)
            cv2.imshow("Motion Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
  

def main():
    vidpath = "captures/videos/video0.mp4"
    motion_detector = ConsensusMotionDetection()

    cap = cv2.VideoCapture(vidpath)
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
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
    # main()
    iterate_main('videos_new')