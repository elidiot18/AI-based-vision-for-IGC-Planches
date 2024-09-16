import os
import json
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from pycocotools.mask import toBbox
from pycocotools.mask import area
from scipy.ndimage import label

PATCH_SIZE = 256
OVERLAP = 0.3

def load_coco_annotations(json_path):
    """Load COCO annotations from a JSON file."""
    return COCO(json_path)

def decode_rle_to_mask(rle):
    """Decode RLE to binary mask."""
    return coco_mask.decode(rle)

def crop_to_patch(img, x, y, patch_size=PATCH_SIZE):
    """Crop the binary mask to the patch coordinates."""
    return img[y:y + patch_size, x:x + patch_size]

def compute_rle_for_patch(cropped_mask):
    """Compute RLE for the cropped patch mask."""
    rle = coco_mask.encode(np.asfortranarray(cropped_mask))
    rle['counts'] = rle['counts'].decode('ascii')  # Ensure 'counts' is a string for JSON serialization
    return rle

def find_connected_components(binary_mask):
    """Find connected components in binary mask."""
    labeled_array, num_features = label(binary_mask)
    return labeled_array, num_features

def connected_components_to_rle(binary_mask, labeled_array):
    """Convert connected components to RLE format."""
    components_rles = []
    num_labels = np.max(labeled_array)

    for label_id in range(1, num_labels + 1):
        component_mask = (labeled_array == label_id).astype(np.uint8)
        rle = coco_mask.encode(np.asfortranarray(component_mask))
        rle['counts'] = rle['counts'].decode('ascii')
        components_rles.append(rle)
    return components_rles

def rle_connected_components(rle):
    """Find connected components of an RLE mask."""
    # Decode RLE to binary mask
    binary_mask = decode_rle_to_mask(rle)

    # Find connected components
    labeled_array, num_features = find_connected_components(binary_mask)

    # Convert connected components to RLE format
    components_rles = connected_components_to_rle(binary_mask, labeled_array)

    return components_rles

def generate_patches(image, coco, output_folder, output_json_path, name, patch_size=PATCH_SIZE, overlap=OVERLAP):
    """Generate patches from the image and adjust annotations for each patch."""
    img_height, img_width = image.shape[:2]
    stride = int(patch_size * (1 - overlap))

    annotations = coco.loadAnns(coco.getAnnIds())
    num_annotations = len(annotations)  # Total number of annotations
    patch_id = 1
    ann_id = 1
    all_patch_annotations = []
    images_list = []

    for y in range(0, img_height - patch_size + 1, stride):
        for x in range(0, img_width - patch_size + 1, stride):
            patch_image = crop_to_patch(image, x, y, patch_size)
            patch_filename = os.path.join(output_folder, f"{name}_patch_{patch_id}.png")
            patch_annotations = []

            # Save patch image in the specified folder
            cv2.imwrite(patch_filename, patch_image)

            for ann in annotations:
                original_bbox = ann['bbox']
                x_min, y_min, w, h = original_bbox
                x_max, y_max = x_min + w, y_min + h

                # Check if annotation overlaps with the patch
                if not (x + patch_size < x_min or x > x_max or y + patch_size < y_min or y > y_max):
                    mask = decode_rle_to_mask(ann['segmentation'])
                    cropped_mask = crop_to_patch(mask, x, y, patch_size)

                    if np.any(cropped_mask):  # Skip if mask has no positive values
                        cropped_rle = compute_rle_for_patch(cropped_mask)
                        components = rle_connected_components(cropped_rle)
                        for k, component in enumerate(components):
                            a = float(area(component))
                            if a < 4:
                                continue
                            new_bbox = toBbox(component).tolist()
                            patch_annotations.append({
                                "id": ann_id,  # Create unique ID for the patch
                                "image_id": patch_id,
                                "name": f'{patch_id}_' + ann["name"] + (f'_{k}' if len(components) > 1 else ''),
                                "category_id": ann['category_id'],
                                "segmentation": component,
                                "bbox": new_bbox,
                                "area": a,
                                "iscrowd": ann['iscrowd']
                            })
                            ann_id += 1

            if patch_annotations:
                images_list.append({
                    "id": patch_id,
                    "width": patch_size,
                    "height": patch_size,
                    "file_name": f"{name}_patch_{patch_id}.png"
                })
                all_patch_annotations.extend(patch_annotations)

            patch_id += 1

    # Create the new COCO JSON file inside the output folder
    with open(output_json_path, 'w') as output_json:
        json.dump({
            "images": images_list,
            "annotations": all_patch_annotations,
            "categories": coco.dataset['categories']
        }, output_json, indent=4)


def process_image_and_annotations(image_path, json_path, output_folder, name):
    """Process image and annotations to generate patches and COCO JSON."""
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the image and annotations
    image = cv2.imread(image_path)
    coco = load_coco_annotations(json_path)

    # Set the output JSON path inside the output folder
    output_json_path = os.path.join(output_folder, f'{name}_patches.json')

    # Generate patches and adjust annotations
    generate_patches(image, coco, output_folder, output_json_path, name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create 256x256 overlapping patches and adjust COCO annotations.")
    parser.add_argument("name", help="Name <name> such that <name.png> and <name.json> exist in the current directory.")
    args = parser.parse_args()

    image_path = os.path.join(args.name, f"{args.name}.png")
    json_path = os.path.join(args.name, f"{args.name}.json")
    output_folder = os.path.join(args.name, f"{args.name}_patches")

    process_image_and_annotations(image_path, json_path, output_folder, args.name)
