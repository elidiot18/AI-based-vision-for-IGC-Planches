import numpy as np
import cv2
import os

# Function to rotate image and mask
def rotate_image(image, angle):
    """Rotate an image by a given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotate the image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated, M

# Function to calculate the bounding box of the largest area fully contained after rotation
def get_contained_bbox(image_shape, angle):
    """Calculate the largest axis-aligned bounding box that fits inside the image after rotation."""
    h, w = image_shape
    angle_rad = np.radians(angle)

    # Calculate the new dimensions of the rotated image
    new_w = abs(w * np.cos(angle_rad)) + abs(h * np.sin(angle_rad))
    new_h = abs(w * np.sin(angle_rad)) + abs(h * np.cos(angle_rad))

    # The largest box that fits inside the rotated image
    contained_w = int(w * min(np.cos(angle_rad), np.sin(angle_rad)))
    contained_h = int(h * min(np.cos(angle_rad), np.sin(angle_rad)))

    # Calculate the top-left corner of the bounding box within the original image
    top_left_x = (w - contained_w) // 2
    top_left_y = (h - contained_h) // 2

    return top_left_x, top_left_y, contained_w, contained_h

# Function to extract overlapping patches within a contained region
def extract_patches(image, bbox, patch_size=256, overlap=0.4):
    """Extract overlapping patches from a region of an image."""
    step = int(patch_size * (1 - overlap))  # 30% overlap
    patches = []

    top_left_x, top_left_y, contained_w, contained_h = bbox

    # Loop over the contained region to extract patches
    for y in range(top_left_y, top_left_y + contained_h - patch_size + 1, step):
        for x in range(top_left_x, top_left_x + contained_w - patch_size + 1, step):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)

    return patches

# Main augmentation function
def augment_data(data_img_path, truth_img_path, output_dir, angles=np.linspace(0, 360, 30, endpoint=False)):
    # Load data and ground truth images
    data_img = cv2.imread(data_img_path, cv2.IMREAD_COLOR)
    truth_img = cv2.imread(truth_img_path, cv2.IMREAD_GRAYSCALE)  # Assuming ground truth is binary

    if data_img.shape[:2] != truth_img.shape[:2]:
        raise ValueError("Data image and ground truth must have the same dimensions")

    h, w = data_img.shape[:2]

    # Loop over each angle
    for angle in angles:
        # Rotate images and get the rotation matrix
        rotated_data, M = rotate_image(data_img, angle)
        rotated_truth, _ = rotate_image(truth_img, angle)

        # Calculate the bounding box of the fully contained area
        bbox = get_contained_bbox((h, w), angle)

        # Extract overlapping 256x256 patches within the contained region
        data_patches = extract_patches(rotated_data, bbox)
        truth_patches = extract_patches(rotated_truth, bbox)

        # Save the patches
        for i, (data_patch, truth_patch) in enumerate(zip(data_patches, truth_patches)):
            position = f"{i:04d}"
            data_patch_filename = os.path.join(output_dir, f"batch_{angle}_{position}_{i}_test.png")
            truth_patch_filename = os.path.join(output_dir, f"batch_{angle}_{position}_{i}_truth.png")

            # Save the patches
            cv2.imwrite(data_patch_filename, data_patch)
            cv2.imwrite(truth_patch_filename, truth_patch)

# Example usage
data_img_path = "batch_4_1_test.png"
truth_img_path = "batch_4_1_truth.png"
output_dir = "./batch_4"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

augment_data(data_img_path, truth_img_path, output_dir)
