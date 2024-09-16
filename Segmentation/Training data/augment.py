import numpy as np
import cv2
import os
import math

def rotate_and_extract_patches(image, overlap=0.4, angle=0):
    """Rotate the image by a specified angle and extract valid patches from the rotated image."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    if angle == 0:
        rotated = image
        rotated_corners = corners
        new_w, new_h = w, h
    elif angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_corners = np.array([
                [corners[0][1], corners[1][1]], 
                [corners[0][0], corners[0][1]], 
                [corners[2][0], corners[3][1]], 
                [corners[3][0], corners[2][1]], 
            ])
        new_w, new_h = h, w
    elif angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        rotated_corners = corners[::-1]
        new_w, new_h = w, h
    elif angle == 270:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_corners = np.array([
                [corners[1][0], corners[1][1]], 
                [corners[3][0], corners[0][1]], 
                [corners[2][0], corners[2][1]], 
                [corners[0][0], corners[0][1]], 
            ])
        new_w, new_h = h, w
    else:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos_angle = abs(M[0, 0])
        sin_angle = abs(M[0, 1])
        
        new_w = int(h * sin_angle + w * cos_angle)
        new_h = int(h * cos_angle + w * sin_angle)
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        rotated_corners = cv2.transform(np.array([corners]), M[:2, :])[0]
    
    step = int(256 * (1 - overlap))  # Overlap by the given percentage
    h_rot, w_rot = rotated.shape[:2]
    patches = []

    


    def is_point_in_polygon(px, py, polygon):
        # Convert to integer points for cv2.pointPolygonTest
        polygon_int = np.int32(polygon)
        #img_display = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        #for (vx, vy) in polygon_int:
        #    cv2.circle(img_display, (vx, vy), 10, (0, 255, 0), -1)  # Large green dots for vertices
        #center_point = (int(px), int(py))  # Ensure integer coordinates
        #if cv2.pointPolygonTest(polygon_int, (px, py), False) >= 0:
        #    cv2.circle(img_display, center_point, 15, (255, 0, 0), -1)  # Larger blue dot for point inside
        #else:
            # Point is outside the polygon
        #    cv2.circle(img_display, center_point, 15, (0, 0, 255), -1)  # Larger red dot for point outside
        #img_display_resized = cv2.resize(img_display, (0, 0), fx=0.3, fy=0.3)
        #cv2.imshow('Polygon Test', img_display_resized)
        #cv2.waitKey(0)  # Wait for key press
        #cv2.destroyAllWindows()
        return cv2.pointPolygonTest(polygon_int, (px, py), False) >= 0

    def is_patch_within_valid_area(x, y):
        """Check if a patch is within the valid area"""
        patch_corners = np.array([
            [x, y],
            [x + 255, y],
            [x, y + 255],
            [x + 255, y + 255]
        ], dtype=np.float32)

        #img_display = np.copy(rotated)
        #cv2.polylines(img_display, [np.int32(rotated_corners)], isClosed=True, color=(0, 255, 0), thickness=2)
        #for (vx, vy) in np.int32(rotated_corners):
        #    cv2.circle(img_display, (vx, vy), 10, (0, 255, 0), -1)  # Large green dots for vertices
        #cv2.rectangle(img_display, (x, y), (x + 255, y + 255), (0, 255, 0), 2)

        is_within = all(is_point_in_polygon(px, py, rotated_corners) for (px, py) in patch_corners)

        #img_display_resized = cv2.resize(img_display, (0, 0), fx=0.3, fy=0.3)
        #cv2.imshow('Patch Validity Test', img_display_resized)
        #cv2.waitKey(0)  # Wait for key press
        #cv2.destroyAllWindows()

        return is_within

    range_y = list(range(0, h_rot - 256, step)) + [h_rot - 256]
    range_x = list(range(0, w_rot - 256, step)) + [w_rot - 256]
    for y in range_y:
        for x in range_x:
            if is_patch_within_valid_area(x, y):
                patch = rotated[y:y + 256, x:x + 256]
                patches.append(patch)

    return patches

def augment_data(data_img_path, truth_img_path, output_dir, angles=np.linspace(0, 360, 5, endpoint=False), idx = 0):
    data_img = cv2.imread(data_img_path, cv2.IMREAD_COLOR)
    truth_img = cv2.imread(truth_img_path, cv2.IMREAD_GRAYSCALE)  # Assuming ground truth is binary

    if data_img is None or truth_img is None:
        raise ValueError("Could not load data or ground truth image.")

    if data_img.shape[:2] != truth_img.shape[:2]:
        raise ValueError("Data image and ground truth must have the same dimensions.")

    for angle in angles:
        data_patches = rotate_and_extract_patches(data_img, angle=angle)
        truth_patches = rotate_and_extract_patches(truth_img, angle=angle)

        for i, (data_patch, truth_patch) in enumerate(zip(data_patches, truth_patches)):
            position = f"{i:04d}"
            data_patch_filename = os.path.join(output_dir, f"batch_{idx}_{angle}_{position}_test.png")
            truth_patch_filename = os.path.join(output_dir, f"batch_{idx}_{angle}_{position}_truth.png")

            cv2.imwrite(data_patch_filename, data_patch)
            cv2.imwrite(truth_patch_filename, truth_patch)

data_img_path = "batch_3_test_3.png"
truth_img_path = "batch_3_truth_3.png"
output_dir = "./batch_3"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

augment_data(data_img_path, truth_img_path, output_dir, idx=3)

