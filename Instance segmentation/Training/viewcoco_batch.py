import cv2
import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import random

def visualize_coco_annotations(dir, name, num_images=5, margin=10):
    # Load COCO annotations
    coco = COCO(os.path.join(dir, f'{name}_patches.json'))

    # Get all image IDs
    img_ids = coco.getImgIds()

    # Randomly select the specified number of images
    selected_img_ids = random.sample(img_ids, min(num_images, len(img_ids)))

    images = []
    for img_id in selected_img_ids:
        # Load image metadata
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(dir, img_info['file_name'])

        # Load the image using OpenCV
        img = cv2.imread(image_path)

        # Get annotations for the image
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))

        # Draw annotations
        for ann in anns:
            try:
                # Draw segmentation mask
                if 'segmentation' in ann:
                    segmentation = ann['segmentation']
                    mask = None
                    if isinstance(segmentation, dict) and 'counts' in segmentation:
                        # RLE format
                        mask = coco_mask.decode(segmentation)
                    elif isinstance(segmentation, list):
                        # Polygon format (convert to mask if needed)
                        rles = coco_mask.frPyObjects(segmentation, img_info['height'], img_info['width'])
                        mask = coco_mask.decode(rles)

                    if mask is not None:
                        # Only apply the mask if it was successfully decoded
                        img[mask > 0] = [0, 255, 0]  # Highlight with green
                    else:
                        print(f"Failed to decode segmentation for annotation ID {ann['id']}")
                        continue

                # Draw bounding box
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    x, y, width, height = bbox
                    cv2.rectangle(img, (int(x), int(y)), (int(x + width), int(y + height)), color=(255, 0, 0), thickness=2)  # Draw in blue

            except Exception as e:
                print(f"Error processing annotation ID {ann['id']}: {str(e)}")
                continue

        # Resize image to a fixed width for visualization
        max_width = 500  # Adjust width here
        aspect_ratio = img.shape[1] / img.shape[0]
        resized_img = cv2.resize(img, (max_width, int(max_width / aspect_ratio)))

        images.append(resized_img)

    # Ensure there are at least some images
    if len(images) == 0:
        print("No valid images were loaded or processed.")
        return

    # Create a 2-row grid layout with dynamic number of columns
    rows = 2
    cols = (len(images) + rows - 1) // rows  # Compute number of columns based on the number of images

    # Find the maximum height and width of the images
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    # Create a blank canvas to place images with margins
    canvas_h = rows * (max_h + margin) - margin
    canvas_w = cols * (max_w + margin) - margin
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place images in the canvas
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        y = row * (max_h + margin)
        x = col * (max_w + margin)
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img

    # Display the canvas with annotations
    cv2.imshow('COCO Visualization', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualizes COCO annotations.")
    parser.add_argument("name", help="Name of the COCO Json directory to visualize.")
    parser.add_argument("--n", type=int, default=8, help="Number of images to display.")
    args = parser.parse_args()

    dir = os.path.join(args.name, args.name + '_patches')
    visualize_coco_annotations(dir, args.name, args.n)

if __name__ == "__main__":
    main()
