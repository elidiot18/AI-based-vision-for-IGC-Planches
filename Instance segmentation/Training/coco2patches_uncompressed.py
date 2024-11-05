import os
import json
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.mask import area
from scipy.ndimage import label

########## costum import
import costumcoco as ccc

PATCH_SIZE = 256
OVERLAP = 0.3

################################################
############ Tools for patching
############

def crop_to_patch(img, x, y, patch_size=PATCH_SIZE):
    """Crop the image to the patch coordinates."""
    return img[y:y + patch_size, x:x + patch_size]

def mask_components(binary_mask):
    """Find connected components in binary mask."""
    labeled_array, num_features = label(binary_mask)
    return labeled_array, num_features

def connected_components_to_uncompressed_rle(binary_mask, labeled_array):
    """Convert connected components to RLE format."""
    components_rles = []
    num_labels = np.max(labeled_array)

    for label_id in range(1, num_labels + 1):
        component_mask = labeled_array == label_id
        rle = ccc.mask_to_uncompressed_rle(component_mask)
        components_rles.append(rle)
    return components_rles

def uncompressed_rle_connected_components(rle):
    """Find connected components of an RLE mask."""
    # Decode RLE to binary mask
    binary_mask = ccc.uncompressed_rle_to_mask(rle)

    # Find connected components
    labeled_array, num_features = mask_components(binary_mask)

    # Convert connected components to RLE format
    components_rles = connected_components_to_uncompressed_rle(binary_mask, labeled_array)

    return components_rles

def rotate90_uncompressed_rle(rle):
    """Rotate RLE encoding by 90 degrees clockwise assuming canvas is a square."""
    counts = rle['counts']
    edge = rle['size'][0]

    if edge != rle['size'][1]:
        raise ValueError('RLE Canvas has to be a square')

    new_counts = []

    def add_run(start, length):
        """Add a run to the new_counts list."""
        if new_counts and new_counts[-1][0] == start:
            new_counts[-1][1] += length
        else:
            new_counts.append([start, length])

    num_runs = len(counts) // 2
    for run_idx in range(num_runs):
        run_length = counts[2 * run_idx]
        run_value = counts[2 * run_idx + 1]

        # Calculate the start and end positions of the run in the rotated image
        for run_pos in range(run_length):
            y = (run_idx + run_pos) // edge
            x = (run_idx + run_pos) % edge
            new_x = edge - y - 1
            new_y = x

            start_pos = new_y * edge + new_x
            end_pos = start_pos + 1

            if run_value > 0:
                add_run(start_pos, end_pos - start_pos)

    # Flatten the list of runs into a flat array
    flattened_counts = []
    for start, length in new_counts:
        flattened_counts.append(start)
        flattened_counts.append(length)

    return {'counts': np.array(flattened_counts), 'size': rle['size']}

################################################
############ Main
############

def generate_patches(image, coco, output_folder, output_json_path, name, patch_size=PATCH_SIZE, overlap=OVERLAP):
    """Generate patches from the image and adjust annotations for each patch."""
    img_height, img_width = image.shape[:2]
    stride = int(patch_size * (1 - overlap))

    annotations = coco.loadAnns(coco.getAnnIds())
    num_annotations = len(annotations)  # Total number of annotations
    patch_id = 1
    ann_id = 1
    patch_annotations = []
    images_list = []

    for y in range(0, img_height - patch_size + 1, stride):
        for x in range(0, img_width - patch_size + 1, stride):
            patch_image = crop_to_patch(image, x, y, patch_size)

            # Save images for each rotation
            for angle in [0, 90, 180, 270]:
                if angle == 0:
                    rotated_image = patch_image
                elif angle == 90:
                    rotated_image = cv2.rotate(patch_image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    rotated_image = cv2.rotate(patch_image, cv2.ROTATE_180)
                elif angle == 270:
                    rotated_image = cv2.rotate(patch_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                patch_filename = os.path.join(output_folder, f"{name}_patch_{patch_id}_{angle}.png")
                cv2.imwrite(patch_filename, rotated_image)

                images_list.append({
                    "id": patch_id,
                    "width": patch_size,
                    "height": patch_size,
                    "file_name": os.path.basename(patch_filename)
                })

                patch_id += 1

            # Process annotations
            for ann in annotations:
                original_bbox = ann['bbox']
                x_min, y_min, w, h = original_bbox
                x_max, y_max = x_min + w, y_min + h

                if not (x + patch_size < x_min or x > x_max or y + patch_size < y_min or y > y_max):
                    mask = ccc.compressed_rle_to_mask(ann['segmentation'])
                    cropped_mask = crop_to_patch(mask, x, y, patch_size)

                    if np.any(cropped_mask):  # Skip if mask has no positive values
                        cropped_rle = ccc.mask_to_uncompressed_rle(cropped_mask)
                        components = uncompressed_rle_connected_components(cropped_rle)

                        for k, component in enumerate(components):
                            a = float(ccc.uncompressed_rle_area(component))
                            if a < 4:
                                continue

                            for angle in [0, 90, 180, 270]:
                                rotated_rle = rotate90_uncompressed_rle(component)
                                patch_annotations.append({
                                    "id": ann_id,
                                    "image_id": patch_id - 4 + angle // 90,
                                    "name": f'{patch_id - 4 + angle // 90}_{ann["name"]}_{k}_{angle}',
                                    "category_id": ann['category_id'],
                                    "segmentation": ccc.rle_compress(rotated_rle),
                                    "bbox": ccc.uncompressed_rle_to_bbox(component),
                                    "area": a,
                                    "iscrowd": ann['iscrowd']
                                })
                                ann_id += 1

    # Create the new COCO JSON file inside the output folder
    with open(output_json_path, 'w') as output_json:
        json.dump({
            "images": images_list,
            "annotations": patch_annotations,
            "categories": coco.dataset['categories']
        }, output_json, indent=4)

def process_image_and_annotations(image_path, json_path, output_folder, name):
    """Process image and annotations to generate patches and COCO JSON."""
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the image and annotations
    image = cv2.imread(image_path)
    coco = COCO(json_path)

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
