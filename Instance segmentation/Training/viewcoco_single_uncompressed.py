import cv2
import numpy as np
import json
import os
import argparse
from datetime import datetime
import base64
import sys

def leb128_encode(value):
    """Encodes an unsigned integer into LEB128 byte format."""
    result = []
    while value > 0x7F:  # while there are more than 7 bits
        result.append((value & 0x7F) | 0x80)  # Set MSB to 1 to indicate more bytes
        value >>= 7
    result.append(value & 0x7F)  # Final byte with MSB set to 0
    return result

def leb128_decode(encoded_bytes):
    """Decodes a LEB128 encoded list of bytes back to an integer."""
    value = 0
    shift = 0
    for byte in encoded_bytes:
        value |= (byte & 0x7F) << shift  # Take the lower 7 bits
        if byte & 0x80 == 0:  # MSB is 0, this is the last byte
            break
        shift += 7
    return value

def counts_to_string(rle_counts):
    """Compress RLE counts using LEB128 and return a base64 string representing the compressed bytes."""
    compressed_bytes = []
    for count in rle_counts:
        compressed_bytes.extend(leb128_encode(count))  # Add LEB128-encoded bytes
    # Convert the list of bytes to a base64 string
    compressed_string = base64.b64encode(bytearray(compressed_bytes)).decode('ascii')
    return compressed_string

def string_to_counts(compressed_string):
    """Decompress the RLE counts from the LEB128-compressed base64 string format."""
    compressed_bytes = base64.b64decode(compressed_string)  # Convert base64 string back to byte values
    decompressed_counts = []

    i = 0
    while i < len(compressed_bytes):
        value, shift = 0, 0
        while True:
            byte = compressed_bytes[i]
            value |= (byte & 0x7F) << shift
            i += 1
            if byte & 0x80 == 0:  # Stop if MSB is 0
                break
            shift += 7
        decompressed_counts.append(value)
    return decompressed_counts

def compressed_rle_to_mask(rle):
    """Decode compressed RLE to binary mask."""
    size = tuple(rle['size'])
    counts = string_to_counts(rle['counts'])
    mask = np.zeros(np.prod(size), dtype=np.uint8)

    current_pos = 0
    fill_value = 0  # Start with 0, which means the first run is background

    for length in counts:
        if current_pos < len(mask):
            mask[current_pos:current_pos + length] = fill_value
            current_pos += length
        fill_value = 1 - fill_value  # Toggle between 0 and 1 for each run

    return mask.reshape(size, order='F')


def visualize_coco_annotations(coco_json, image_path):
    """Visualize COCO annotations on the given image."""
    # Load COCO annotations
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    # Load the image using OpenCV
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # Create a window with aspect ratio maintained
    aspect_ratio = img_width / img_height
    window_height = 800
    window_width = int(window_height * aspect_ratio)

    f = open(os.devnull, 'w')
    sys.stdout = f
    cv2.namedWindow('COCO Visualization', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('COCO Visualization', window_width, window_height)

    print(datetime.now())
    # Get annotations for image
    img_id = coco_data['images'][0]['id']
    anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
    print(datetime.now())

    # Draw annotations
    for ann in anns:
        # Draw segmentation mask
        if 'segmentation' in ann:
            segmentation = ann['segmentation']
            if isinstance(segmentation, dict) and 'counts' in segmentation:
                mask = compressed_rle_to_mask(segmentation)
                img[mask > 0] = 1  # Assuming the mask values are in [0, 1]
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
    parser = argparse.ArgumentParser(description="Visualizes COCO annotations.")
    parser.add_argument("name", help="Name <name> such that <name.png> and <name.json> exist in the directory <name>.")
    args = parser.parse_args()
    visualize_coco_annotations(os.path.join(args.name, args.name + '.json'), os.path.join(args.name, args.name + '.png'))

if __name__ == "__main__":
    main()
