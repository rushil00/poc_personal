# import time
# import cv2
# from collections import deque
# from concurrent.futures import ThreadPoolExecutor
# import os
# import sys

# # Append the current script directory to the system path
# script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(script_dir)

# class MotionDetection:
#     def __init__(self, queue_len=10, max_workers=1, buffer=12):
#         """
#         Initialize the MotionDetection class.
#         Parameters:
#         queue_len (int): Maximum length of the frame queue.
#         max_workers (int): Maximum number of worker threads.
#         buffer (int): Number of frames to wait before setting motion status to false
#         """
#         self.motionValue = 1000
#         self.frameQueue = deque(maxlen=queue_len)
#         self.executor = ThreadPoolExecutor(max_workers=max_workers)
#         self.threshold = 5
#         self.buffer = buffer
#         self.no_motion_count = 0  # Counter for frames without motion
#         self.prev_motion_status = False
#         self.current_motion_status = False
#         self.motion_check_count = 0
#         self.motion_mask = None
#         self.ignorance_threshold = 3000

#     def set_ignorance_threshold(self, ignorance_threshold):
#         self.ignorance_theshold = ignorance_threshold

#     def motion_detect(self, frame1, frame2, log=True):
#             """
#             Detect motion by comparing two frames using Structural Similarity Index (SSIM).
#             Parameters:
#             frame1 (np.ndarray): First frame.
#             frame2 (np.ndarray): Second frame.
#             log (bool): Whether to log the SSIM index.
#             """
#             try:
#                 gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#                 gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#                 gray_frame1 = cv2.GaussianBlur(gray_frame1, (5, 5), 0)
#                 gray_frame2 = cv2.GaussianBlur(gray_frame2, (5, 5), 0)
                
#                 diff = cv2.absdiff(gray_frame1, gray_frame2)
#                 _, self.motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                
#                 # Remove noise using morphological operations
#                 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#                 self.motion_mask = cv2.morphologyEx(self.motion_mask, cv2.MORPH_OPEN, kernel)
#                 self.motion_mask = cv2.morphologyEx(self.motion_mask, cv2.MORPH_CLOSE, kernel)
                
#                 # Find contours in the motion mask
#                 contours, _ = cv2.findContours(self.motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 motion_pixels = []
#                 for contour in contours:
#                     if cv2.contourArea(contour) > self.ignorance_threshold:  # Ignore small contours to reduce noise
#                         x, y, w, h = cv2.boundingRect(contour)
#                         motion_pixels.append((x + w // 2, y + h // 2))
                
#                 # Draw regions on the frame
#                 H, W = frame1.shape[:2]
#                 regions = [
#                     ((0, 0), (W // 3, H // 2), "Up-Left"),
#                     ((W // 3, 0), (2 * W // 3, H // 2), "Up-Middle"),
#                     ((2 * W // 3, 0), (W, H // 2), "Up-Right"),
#                     ((0, H // 2), (W // 3, H), "Down-Left"),
#                     ((W // 3, H // 2), (2 * W // 3, H), "Down-Middle"),
#                     ((2 * W // 3, H // 2), (W, H), "Down-Right")
#                 ]
#                 for (start, end, label) in regions:
#                     cv2.rectangle(frame1, start, end, (255, 0, 0), 2)
#                     cv2.putText(frame1, label, (start[0] + 10, start[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
#                 # Determine the region where motion is happening
#                 if motion_pixels:
#                     for (x, y) in motion_pixels:
#                         for (start, end, label) in regions:
#                             if start[0] <= x < end[0] and start[1] <= y < end[1]:
#                                 print(f"Motion detected in region: {label}")
#                                 break
                
#             except Exception as e:
#                 print(f"Motion detection failed due to: {e}")
#                 raise e

#     # DEPRECATED:
#     def __motion_detect(self, frame1, frame2, log=True):
#         """
#         Detect motion by comparing two frames using Structural Similarity Index (SSIM).
#         Parameters:
#         frame1 (np.ndarray): First frame.
#         frame2 (np.ndarray): Second frame.
#         log (bool): Whether to log the SSIM index.
#         """
#         try:
#             gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#             gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#             gray_frame1 = cv2.GaussianBlur(gray_frame1, (3, 7), 0)
#             gray_frame2 = cv2.GaussianBlur(gray_frame2, (3, 7), 0)
            
#             diff = cv2.absdiff(gray_frame1, gray_frame2)
#             _, self.motion_mask = cv2.threshold(diff, 7, 255, cv2.THRESH_TRIANGLE)
            
#             # Remove noise using morphological operations
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#             self.motion_mask = cv2.morphologyEx(self.motion_mask, cv2.MORPH_OPEN, kernel)
#             self.motion_mask = cv2.morphologyEx(self.motion_mask, cv2.MORPH_CLOSE, kernel)
            
#             # Find contours in the motion mask
#             contours, _ = cv2.findContours(self.motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             motion_pixels = []
#             avg_x, avg_y = 0, 0
#             for contour in contours:
#                 if cv2.contourArea(contour) > self.ignorance_theshold:  # Ignore small contours to reduce noise
#                     x, y, w, h = cv2.boundingRect(contour)
#                     mask_roi = self.motion_mask[y:y+h, x:x+w]
#                     cluster_pixels = cv2.findNonZero(mask_roi)
#                     if cluster_pixels is not None and len(cluster_pixels) > 200:
#                         avg_x = int(cluster_pixels[:, 0, 0].mean()) + x
#                         avg_y = int(cluster_pixels[:, 0, 1].mean()) + y
#                         motion_pixels.append((avg_x, avg_y))
#             print("Motion detected at pixels:", motion_pixels)
        
#         except Exception as e:
#             print(f"Motion detection failed due to: {e}")
#             raise e

#     def motionUpdate(self, frame, mask):
#         """
#         Update the motion detection with a new frame.
#         Parameters:
#         frame (np.ndarray): New frame to update motion detection.
#         mask: Mask to apply to the frame
#         """
#         self.prev_motion_status = self.current_motion_status
#         frame = cv2.bitwise_and(frame, frame, mask=mask)
#         self.frameQueue.append(frame)
        
#         if len(self.frameQueue) > 1:
#             self.motion_detect(self.frameQueue.popleft(), self.frameQueue.popleft(), False)
#         return self.motion_mask


# def main():
#     # Initialize video capture
#     videopath = 'captures/videos/video0.mp4'
#     cap = cv2.VideoCapture(videopath)  # Use 0 for webcam
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     # Initialize motion detection
#     motion_detector = MotionDetection(queue_len=10, max_workers=1, buffer=12)

#     # Read the first frame to create a mask
#     ret, prev_frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         return
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#     mask = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#     mask[:] = 255  # Full frame mask
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Update motion detection
#         motion_mask = motion_detector.motionUpdate(frame, mask)

#         if motion_mask is not None:
#             motion_mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
#             combined_frame = cv2.hconcat([frame, motion_mask_bgr])
#             combined_frame = cv2.resize(combined_frame, (1280, 480))
#             cv2.imshow("Motion Detection", combined_frame)
#         else:
#             resized_frame = cv2.resize(frame, (1280, 480))
#             cv2.imshow("Motion Detection", resized_frame)

#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release video capture and close windows
#     cap.release()
#     cv2.destroyAllWindows()

# def iterate_main(main_dir:str='captures/videos', ignorance_threshold=2900):
#     video_files = [f for f in os.listdir(main_dir) if f.endswith('.mp4')]
#     for videopath in video_files[-5:]:
#         videopath = os.path.join(main_dir, videopath)
#         cap = cv2.VideoCapture(videopath)  # Use 0 for webcam
#         if not cap.isOpened():
#             print(f"Error: Could not open video {videopath}.")
#             continue

#         # Initialize motion detection
#         motion_detector = MotionDetection(queue_len=10, max_workers=1, buffer=12)
#         motion_detector.set_ignorance_threshold(ignorance_threshold)

#         # Read the first frame to create a mask
#         ret, prev_frame = cap.read()
#         if not ret:
#             print(f"Error: Could not read frame from {videopath}.")
#             cap.release()
#             continue
#         mask = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#         mask[:] = 255  # Full frame mask
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Update motion detection
#             motion_mask = motion_detector.motionUpdate(frame, mask)

#             H, W = frame.shape[:2]
#             regions = [
#                 ((0, 0), (W // 3, H // 2), "Up-Left"),
#                 ((W // 3, 0), (2 * W // 3, H // 2), "Up-Middle"),
#                 ((2 * W // 3, 0), (W, H // 2), "Up-Right"),
#                 ((0, H // 2), (W // 3, H), "Down-Left"),
#                 ((W // 3, H // 2), (2 * W // 3, H), "Down-Middle"),
#                 ((2 * W // 3, H // 2), (W, H), "Down-Right")
#             ]
#             if motion_mask is not None:
#                 motion_mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
#                 # Draw intersecting lines to divide the regions
#                 cv2.line(frame, (W // 3, 0), (W // 3, H), (255, 0, 0), 2)
#                 cv2.line(frame, (2 * W // 3, 0), (2 * W // 3, H), (255, 0, 0), 2)
#                 cv2.line(frame, (0, H // 2), (W, H // 2), (255, 0, 0), 2)
                
#                 combined_frame = cv2.hconcat([frame, motion_mask_bgr])
#                 combined_frame = cv2.resize(combined_frame, (1280, 480))
#                 cv2.imshow("Motion Detection", combined_frame)
#             else:
#                 resized_frame = cv2.resize(frame, (1280, 480))

#                 # Label the regions
#                 for (start, end, label) in regions:
#                     cv2.putText(resized_frame, label, (start[0] + 10, start[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#                 cv2.imshow("Motion Detection", resized_frame)

#             # Break the loop on 'q' key press
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         # Release video capture and close windows
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # main()
#     # iterate_main(main_dir="videos_representative",ignorance_threshold=500)
#     iterate_main(ignorance_threshold=500)
import time
import cv2
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

class MotionDetection:
    def __init__(self, queue_len=10, max_workers=1, buffer=12):
        self.motionValue = 1000
        self.frameQueue = deque(maxlen=queue_len)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.threshold = 5
        self.buffer = buffer
        self.no_motion_count = 0
        self.prev_motion_status = False
        self.current_motion_status = False
        self.motion_check_count = 0
        self.motion_mask = None
        self.ignorance_threshold = 3000  # Correct variable name
        self.regions_detected = {}  # To store detected regions and their counts
        self.frame_counter = 0  # Frame counter to process every 9th frame

    def set_ignorance_threshold(self, ignorance_threshold):
        self.ignorance_threshold = ignorance_threshold  # Fixed typo

    def motion_detect(self, frame1, frame2, log=True):
        try:
            gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gray_frame1 = cv2.GaussianBlur(gray_frame1, (3, 7), 0)
            gray_frame2 = cv2.GaussianBlur(gray_frame2, (3, 7), 0)
            
            diff = cv2.absdiff(gray_frame1, gray_frame2)
            _, self.motion_mask = cv2.threshold(diff, 7, 255, cv2.THRESH_TRIANGLE)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            self.motion_mask = cv2.morphologyEx(self.motion_mask, cv2.MORPH_OPEN, kernel)
            self.motion_mask = cv2.morphologyEx(self.motion_mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(self.motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_pixels = []
            H, W = frame1.shape[:2]  # Get frame dimensions
            for contour in contours:
                if cv2.contourArea(contour) > self.ignorance_threshold:  # Use corrected variable
                    x, y, w, h = cv2.boundingRect(contour)
                    mask_roi = self.motion_mask[y:y+h, x:x+w]
                    cluster_pixels = cv2.findNonZero(mask_roi)
                    if cluster_pixels is not None and len(cluster_pixels) > 200:
                        avg_x = int(cluster_pixels[:, 0, 0].mean()) + x
                        avg_y = int(cluster_pixels[:, 0, 1].mean()) + y
                        motion_pixels.append((avg_x, avg_y))
            
            # Determine regions
            horizontal_step = H // 2
            vertical_step = W // 3
            self.regions_detected.clear()
            for (x, y) in motion_pixels:
                # Horizontal region (up/down)
                h_region = 'up' if y < horizontal_step else 'down'
                # Vertical region (left/middle/right)
                if x < vertical_step:
                    v_region = 'left'
                elif x < 2 * vertical_step:
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
                print(f"Frame: {self.frame_counter}, Motion detected in regions: {self.regions_detected}")
        
        except Exception as e:
            print(f"Motion detection failed due to: {e}")
            raise e

    def motionUpdate(self, frame, mask):
        self.prev_motion_status = self.current_motion_status
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        self.frameQueue.append(frame)
        
        self.frame_counter += 1
        if self.frame_counter % 9 == 0 and len(self.frameQueue) > 1:
            self.motion_detect(self.frameQueue.popleft(), self.frameQueue.popleft(), False)
        return self.motion_mask

def main():
    videopath = 'captures/videos/video0.mp4'
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    motion_detector = MotionDetection(queue_len=10, max_workers=1, buffer=12)
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return
    mask = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    mask[:] = 255
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        motion_mask = motion_detector.motionUpdate(frame, mask)
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
            cv2.imshow("Motion Detection", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def iterate_main(main_dir:str='captures/videos', ignorance_threshold=2900):
    video_files = [f for f in os.listdir(main_dir) if f.endswith('.mp4')]
    for videopath in video_files:
        videopath = os.path.join(main_dir, videopath)
        cap = cv2.VideoCapture(videopath)
        if not cap.isOpened():
            print(f"Error: Could not open video {videopath}.")
            continue

        motion_detector = MotionDetection(queue_len=10, max_workers=1, buffer=12)
        motion_detector.set_ignorance_threshold(ignorance_threshold)

        ret, prev_frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from {videopath}.")
            cap.release()
            continue
        mask = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        mask[:] = 255
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Frames per second: {fps}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            H, W = frame.shape[:2]
            # Draw intersecting lines to divide the regions
            cv2.line(frame, (W // 3, 0), (W // 3, H), (255, 0, 0), 2)
            cv2.line(frame, (2 * W // 3, 0), (2 * W // 3, H), (255, 0, 0), 2)
            cv2.line(frame, (0, H // 2), (W, H // 2), (255, 0, 0), 2)
            motion_mask = motion_detector.motionUpdate(frame, mask)
            regions = motion_detector.regions_detected

            if motion_mask is not None:
                motion_mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                combined_frame = cv2.hconcat([frame, motion_mask_bgr])
                combined_frame = cv2.resize(combined_frame, (1280, 480))
                if regions:
                    max_region = max(regions, key=regions.get)
                    text = f"Max Motion Region: ({max_region[0]}, {max_region[1]})"
                    cv2.putText(combined_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Motion Detection", combined_frame)
            else:
                resized_frame = cv2.resize(frame, (1280, 480))
                cv2.imshow("Motion Detection", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # main()
    iterate_main(main_dir='videos_representative',ignorance_threshold=500)