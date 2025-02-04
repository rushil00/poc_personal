import cv2

cap = cv2.VideoCapture(0)  # Use 0 for webcam
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(diff, 33, 255, cv2.THRESH_TRUNC)
    prev_gray = gray

    cv2.imshow("Motion Mask", motion_mask)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # Initialize background subtractor (MOG2 handles shadows better)
# back_sub = cv2.createBackgroundSubtractorMOG2(
#     history=3,        # Adjust based on required persistence
#     varThreshold=500,    # Lower = more sensitive
#     detectShadows=False  # Disable for cleaner detection
# )

# # Kernel for morphological operations
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# # Start video capture (0 for default camera, or video path)
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# # Minimum contour area to consider as motion (adjust based on camera distance)
# MIN_CONTOUR_AREA = 100  

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Apply background subtraction
#     fg_mask = back_sub.apply(frame)
    
#     # Noise reduction and fill holes
#     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
#     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
#     fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
    
#     # Find contours in the foreground mask
#     contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     motion_coordinates = []
    
#     for contour in contours :
#         if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
#             # Get bounding box for the contour
#             x, y, w, h = cv2.boundingRect(contour)
#             motion_coordinates.append((x, y, w, h))
#             # Draw bounding box on the original frame
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Display the original frame with motion areas highlighted
#     cv2.imshow('Motion Detection', frame)
    
#     # Optionally print the coordinates of detected motion areas
#     for coord in motion_coordinates:
#         print(f'Motion detected at: {coord}')

#     # Break the loop on 'q' key press
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

