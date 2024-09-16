import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from datetime import datetime
import os

def visualize_coco_annotations(coco_json, image_path):
    # Load COCO annotations
    coco = COCO(coco_json)

    # Load the image using OpenCV
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # Create a window with aspect ratio maintained
    aspect_ratio = img_width / img_height
    window_height = 800
    window_width = int(window_height * aspect_ratio)

    cv2.namedWindow('COCO Visualization', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('COCO Visualization', window_width, window_height)
    print()
    # Get annotations for image
    print(datetime.now())
    img_id = coco.getImgIds()[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    print(datetime.now())

    # Draw annotations
    for ann in anns:
        # Draw segmentation mask
        if 'segmentation' in ann:
            segmentation = ann['segmentation']
            if isinstance(segmentation, dict) and 'counts' in segmentation:
                # RLE format
                img[coco_mask.decode(segmentation) > 0] = 1
            if 'bbox' in ann:
                bbox = ann['bbox']
                x, y, width, height = bbox
                cv2.rectangle(img, (int(x), int(y)), (int(x + width), int(y + height)), color=(0, 255, 0), thickness=2)
    print(datetime.now())
    # Display the image
    cv2.imshow('COCO Visualization', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualizes COCO annotations.")
    parser.add_argument("name", help="Name <name> such that <name.png> and <name.json> exist in the directory <name>.")
    args = parser.parse_args()
    visualize_coco_annotations(os.path.join(args.name, args.name + '.json'), os.path.join(args.name, args.name + '.png'))

if __name__ == "__main__":
    main()
