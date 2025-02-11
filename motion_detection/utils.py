import cv2
import numpy as np


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

def draw_regions(frame, num=1):
    """
    Let user draw 6 regions using mouse clicks.
    Returns list of region polygons.
    """
    regions = []
    current_polygon = []
    temp_frame = frame.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_polygon, temp_frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append((x, y))
            cv2.circle(temp_frame, (x, y), 3, (0, 255, 0), -1)
            if len(current_polygon) > 1:
                cv2.line(temp_frame, current_polygon[-2], current_polygon[-1], (0, 255, 0), 2)
            cv2.imshow("Draw Regions", temp_frame)
            
        elif event == cv2.EVENT_RBUTTONDOWN and len(current_polygon) > 2:
            cv2.line(temp_frame, current_polygon[0], current_polygon[-1], (0, 255, 0), 2)
            regions.append(np.array(current_polygon, np.int32))
            current_polygon = []
            if len(regions) < num:
                temp_frame = frame.copy()
                for region in regions:
                    cv2.polylines(temp_frame, [region], True, (0, 255, 0), 2)
            cv2.imshow("Draw Regions", temp_frame)

    cv2.namedWindow("Draw Regions")
    cv2.imshow("Draw Regions", temp_frame)
    cv2.setMouseCallback("Draw Regions", mouse_callback)
    
    print(f"Draw {num} regions. Left click to add points, right click to complete a region.")
    while len(regions) < num:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
            break
    
    cv2.destroyWindow("Draw Regions")
    return regions

def draw_quadrilateral(frame, num=1):
    """
    Let user draw 6 regions using mouse clicks.
    Returns list of region polygons.
    """
    regions = []
    current_polygon = []
    temp_frame = frame.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_polygon, temp_frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append((x, y))
            cv2.circle(temp_frame, (x, y), 3, (0, 255, 0), -1)
            
            if len(current_polygon) > 1:
                cv2.line(temp_frame, current_polygon[-2], current_polygon[-1], (0, 255, 0), 2)
            
            if len(current_polygon) == 4:
                cv2.line(temp_frame, current_polygon[0], current_polygon[-1], (0, 255, 0), 2)
                regions.append(np.array(current_polygon, np.int32))
                current_polygon = []
                if len(regions) < num:
                    temp_frame = frame.copy()
                    for region in regions:
                        cv2.polylines(temp_frame, [region], True, (0, 255, 0), 2)
                        
            cv2.imshow("Draw Regions", temp_frame)

    cv2.namedWindow("Draw Regions")
    cv2.imshow("Draw Regions", temp_frame)  # Show the frame immediately
    cv2.setMouseCallback("Draw Regions", mouse_callback)
    
    print(f"Draw {num} regions. Click to add points. Each region will complete after 4 points.")
    while len(regions) < num:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
            break
    
    cv2.destroyWindow("Draw Regions")
    return regions
