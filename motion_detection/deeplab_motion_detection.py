# import cv2
# import numpy as np
# import torch
# from torchvision import models, transforms
# from PIL import Image

# # Load DeepLabV3 model (pre-trained on COCO)
# deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# # Define transformations for the input image
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# def segment_shelves(frame):
#     """
#     Segments shelves in the frame using DeepLabV3.
#     Returns a binary mask where shelves are white (255) and the rest is black (0).
#     """
#     # Preprocess the frame
#     input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     input_tensor = transform(input_image).unsqueeze(0)

#     # Perform segmentation
#     with torch.no_grad():
#         output = deeplab(input_tensor)['out'][0]
#     output_predictions = output.argmax(0).byte().cpu().numpy()

#     # Create a binary mask for shelves (assuming shelves are labeled as 'wall' or 'furniture' in COCO)
#     shelf_mask = np.zeros_like(output_predictions)
#     shelf_mask[output_predictions == 12] = 255  # COCO class 12: 'furniture' (adjust as needed)

#     return shelf_mask

# def process_video(video_path):
#     """
#     Processes a video to detect shelves and apply motion masking.
#     """
#     # Open video file or camera feed
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     # Read the first frame
#     ret, prev_frame = cap.read()
#     if not ret:
#         print("Error: Could not read video.")
#         return

#     # Convert the first frame to grayscale
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

#     # Create a mask for motion tracking
#     motion_mask = np.zeros_like(prev_frame)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  # End of video

#         # Segment shelves in the current frame
#         shelf_mask = segment_shelves(frame)

#         # Convert the current frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Compute optical flow (motion vectors) using Lucas-Kanade
#         flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#         # Compute magnitude and angle of motion vectors
#         magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

#         # Create a mask for significant motion
#         motion_threshold = 5  # Adjust as needed
#         significant_motion = cv2.inRange(magnitude, motion_threshold, 255)

#         # Apply the shelf mask to filter out irrelevant motion
#         filtered_motion = cv2.bitwise_and(significant_motion, significant_motion, mask=shelf_mask)

#         # Overlay the filtered motion on the original frame
#         output_frame = frame.copy()
#         output_frame[filtered_motion > 0] = [0, 0, 255]  # Highlight motion in red

#         # Display the output frame
#         cv2.imshow("Shelf Segmentation + Motion Masking", output_frame)

#         # Update the previous frame
#         prev_gray = gray

#         # Exit on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()

# # Usage
# video_path = "shelf_video.mp4"  # Replace with your video path or use 0 for webcam
# process_video(video_path)

from ultralytics import YOLO
import cv2

# Load your custom YOLOv8 model (replace with your model path)
model = YOLO('yolov8_model.pt')  # Example: 'grill_frame_model.pt'

# Open the video file
video_path = 'path/to/your/video.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Extract and visualize detections
    for result in results:
        # Get bounding boxes, class IDs, and confidence scores
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2] format
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

        # Loop through each detection
        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            # Filter for the "grill frame" class (replace with your class ID)
            if class_id == 0:  # Assuming class ID 0 is for "grill frame"
                x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
                label = f"Grill Frame {confidence:.2f}"  # Label with confidence

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('YOLOv8 Grill Frame Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()