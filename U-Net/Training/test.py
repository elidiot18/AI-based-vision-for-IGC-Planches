import cv2
import numpy as np

def rotate_and_crop(image, angle):
    """Rotate the image and crop it to the largest axis-aligned rectangle fully contained in the rotated image."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotate the image with cv2.warpAffine and expand the canvas
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_angle = abs(M[0, 0])
    sin_angle = abs(M[0, 1])

    # Compute the new dimensions of the rotated image (bounding box)
    new_w = int(h * sin_angle + w * cos_angle)
    new_h = int(h * cos_angle + w * sin_angle)

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h))

    # Now calculate the largest axis-aligned rectangle that fits in the rotated image
    # This is a known geometric problem; we can calculate it using the original image dimensions.
    angle_rad = np.radians(angle)
    
    # Absolute values of cos and sin of the angle
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))
    
    # Compute the dimensions of the largest rectangle that fits within the rotated image
    contained_w = int(w * cos_a - h * sin_a)
    contained_h = int(h * cos_a - w * sin_a)

    # If the result is negative, swap width and height (for large angles like 90°, 180°, etc.)
    contained_w = min(w, contained_w) if contained_w > 0 else min(h, -contained_w)
    contained_h = min(h, contained_h) if contained_h > 0 else min(w, -contained_h)

    # Compute the top-left corner of the bounding box for cropping
    top_left_x = (new_w - contained_w) // 2
    top_left_y = (new_h - contained_h) // 2

    # Crop the rotated image to the largest contained rectangle
    cropped = rotated[top_left_y:top_left_y + contained_h, top_left_x:top_left_x + contained_w]

    return cropped
    
def main(input_path, output_path):
    # Read the input image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Unable to read image at {input_path}")
        return

    # Rotate the image by 30 degrees and crop it
    rotated_cropped = rotate_and_crop(image, 30)

    # Save the output image
    cv2.imwrite(output_path, rotated_cropped)
    print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    # Set input and output file paths
    input_path = 'batch_5_test.png'  # Replace with your input image path
    output_path = 'batch_5_30.png'  # Replace with your desired output image path

    main(input_path, output_path)

