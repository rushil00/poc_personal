# import cv2
# import numpy as np
# from sklearn.cluster import KMeans, DBSCAN

# def detect_shelf_levels(image_path, n_shelves=4):
#     # Load image
#     img = cv2.imread(image_path)
#     if img is None:
#         print("Error: Image not found")
#         return
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Edge detection using Canny
#     edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
#     # Detect lines using Hough Transform
#     lines = cv2.HoughLinesP(edges, 
#                            rho=1, 
#                            theta=np.pi/180, 
#                            threshold=100,
#                            minLineLength=100, 
#                            maxLineGap=10)
    
#     # Filter horizontal lines (shelves are mostly horizontal)
#     horizontal_lines = []
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
#             if abs(angle) < 5:  # Allow small angle deviation
#                 horizontal_lines.append((y1, y2))
    
#     # Extract y-coordinates of all horizontal lines
#     y_coords = []
#     for y1, y2 in horizontal_lines:
#         y_coords.extend([y1, y2])
    
#     if not y_coords:
#         print("No shelf levels detected")
#         return img
    
#     # Cluster y-coordinates using K-Means
#     kmeans = KMeans(n_clusters=n_shelves)
#     points = np.array(y_coords).reshape(-1, 1)
#     kmeans.fit(points)
    
#     # Get cluster centers (main shelf levels)
#     shelf_levels = sorted([int(center[0]) for center in kmeans.cluster_centers_])
    
#     # Draw detected shelf levels
#     for level in shelf_levels:
#         cv2.line(img, (0, level), (img.shape[1], level), (0, 0, 255), 2)
    
#     # Visualization
#     cv2.imshow("Edges", edges)
#     cv2.imshow("Shelf Detection", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     return img

# import matplotlib.pyplot as plt

# def detect_shelf_levels_with_dbscan(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         print("Error: Image not found")
#         return
    
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#     edges = cv2.Canny(blurred, 150, 200, apertureSize=3)
    
    
#     lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

#     horizontal_lines = []
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
#             if abs(angle) < 5:
#                 horizontal_lines.append((y1, y2))
    
#     y_coords = []
#     for y1, y2 in horizontal_lines:
#         y_coords.extend([y1, y2])
    
#     if not y_coords:
#         print("No shelf levels detected")
#         return img
    
#     points = np.array(y_coords).reshape(-1, 1)
    
#     # Use DBSCAN for clustering
#     dbscan = DBSCAN(eps=75, min_samples=10)
#     clusters = dbscan.fit_predict(points)
    
#     shelf_levels = []
#     for label in set(clusters):
#         if label != -1:  # Ignore noise
#             cluster_points = points[clusters == label]
#             shelf_levels.append(int(np.mean(cluster_points)))
    
#     shelf_levels = sorted(shelf_levels)
    
#     for level in shelf_levels:
#         cv2.line(img, (0, level), (img.shape[1], level), (0, 0, 255), 2)
    
#     # Plot clustering points
#     plt.scatter(points, np.zeros_like(points), c=clusters, cmap='viridis')
#     plt.xlabel('Y-coordinates')
#     plt.title('DBSCAN Clustering of Shelf Levels')
#     plt.show()
    
#     # Visualize Hough lines
#     hough_lines_img = img.copy()
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(hough_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
#     cv2.imshow("Canny Edges", edges)
#     cv2.imshow("Hough Lines", hough_lines_img)
#     cv2.imshow("Shelf Detection with DBSCAN", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     return img

# # Usage
# input_image = "images/image.png"  # Replace with your image path
# # output_image = detect_shelf_levels(input_image, n_shelves=3)  # Adjust n_shelves as needed
# output_image = detect_shelf_levels_with_dbscan(input_image)
