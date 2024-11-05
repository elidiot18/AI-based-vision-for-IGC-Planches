import json
import numpy as np
import cv2
import svgpathtools
from skimage.draw import polygon, disk
import os
import argparse
import random
from xml.etree import ElementTree as ET
import base64

########## costum import
import costumcoco as ccc

PATCH_SIZE = 256
OVERLAP = 0.3

################################################
############ Dealing with the raster embedded
############ inside the .svg file
############


def extract_image_from_svg(svg_path, output_path):
    """Extract the base64-encoded raster image from the SVG and save it to output_path."""
    """return: width, height"""
    with open(svg_path, 'r') as file:
        svg_content = file.read()

    root = ET.fromstring(svg_content)
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    image = root.find('.//svg:image', namespaces)

    if image is None:
        raise RuntimeError("No image found in SVG.")

    # Get the base64-encoded image data
    image_data = image.attrib.get('{http://www.w3.org/1999/xlink}href')
    if not image_data.startswith('data:image/png;base64,'):
        raise RuntimeError("Unsupported image format or no image data found.")

    # Extract base64-encoded part
    base64_data = image_data.split('base64,', 1)[1]
    image_bytes = base64.b64decode(base64_data)

    # Convert bytes to a numpy array
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the image array to an OpenCV image
    image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Error decoding the image.")

    # Save the image using OpenCV
    cv2.imwrite(output_path, image)
    return image.shape[1], image.shape[0]  # width, height

def get_raster_virtual_size(svg_path):
    """Get the virtual width and height of the raster embedded in the svg file."""
    with open(svg_path, 'r') as file:
        svg_content = file.read()
    root = ET.fromstring(svg_content)
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    image = root.find('.//svg:image', namespaces)
    if image is not None:
        width = float(image.attrib.get('width', '100px').replace('px', '').strip())
        height = float(image.attrib.get('height', '100px').replace('px', '').strip())
        return width, height
    return None

################################################
############ Main
############

def svg_to_masks(svg_path, virtual_size, intrinsic_size):
    """Convert SVG paths and circles to binary masks and extract fill colors."""
    vw, vh = virtual_size
    iw, ih = intrinsic_size

    tree = ET.parse(svg_path)
    root = tree.getroot()
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}

    masks_and_attributes = []

    def process_path(path_element):
        fill_color = path_element.attrib.get('fill', '#000000')
        path_id = path_element.attrib.get('id', str(len(masks_and_attributes) + 1))  # Default to index if ID is not found
        path = svgpathtools.parse_path(path_element.attrib.get('d', ''))
        coords = []
        for segment in path:
            if isinstance(segment, (svgpathtools.Line, svgpathtools.CubicBezier, svgpathtools.QuadraticBezier)):
                coords.extend([(p.real, p.imag) for p in segment])
        if coords:
            coords = np.array(coords)
            coords[:, 0] = (coords[:, 0] / vw) * iw
            coords[:, 1] = (coords[:, 1] / vh) * ih
            coords = np.clip(coords, 0, [iw - 1, ih - 1])
            binary_mask = np.zeros((ih, iw))
            rr, cc = polygon(coords[:, 1], coords[:, 0], shape=binary_mask.shape)
            binary_mask[rr, cc] = 1
            if np.sum(binary_mask) > 0:
                masks_and_attributes.append((binary_mask, fill_color, path_id))

    def process_circle(circle_element):
        fill_color = circle_element.attrib.get('fill', '#000000')
        circle_id = circle_element.attrib.get('id', str(len(masks_and_attributes) + 1))  # Default to index if ID is not found
        cx = float(circle_element.attrib.get('cx', '0'))
        cy = float(circle_element.attrib.get('cy', '0'))
        r = float(circle_element.attrib.get('r', '0'))
        cx = (cx / vw) * iw
        cy = (cy / vh) * ih
        r = (r / vh) * ih
        binary_mask = np.zeros((ih, iw))
        rr, cc = disk((cy, cx), r, shape=binary_mask.shape)
        binary_mask[rr, cc] = 1
        if np.sum(binary_mask) > 0:
            masks_and_attributes.append((binary_mask, fill_color, circle_id))

    path_elements = root.findall('.//svg:path', namespaces)
    circle_elements = root.findall('.//svg:circle', namespaces)

    for path_element in path_elements:
        process_path(path_element)
    for circle_element in circle_elements:
        process_circle(circle_element)
    return masks_and_attributes

def generate_coco_annotations(svg_path, output_folder):
    """Generate COCO annotations from SVG and raster image."""

    raster_path = os.path.join(output_folder, os.path.splitext(svg_path)[0] + '.png')
    image_filename = os.path.basename(raster_path)

    virtual_size = get_raster_virtual_size(svg_path)
    intrinsic_size = extract_image_from_svg(svg_path, raster_path)

    masks_and_attributes = svg_to_masks(svg_path, virtual_size, intrinsic_size)

    category_map = {}
    categories = []
    annotations = []

    for i, (binary_mask, fill_color, object_id) in enumerate(masks_and_attributes, start=1):
        if fill_color not in category_map:
            category_id = len(category_map) + 1
            category_map[fill_color] = category_id
            categories.append({"id": category_id, "name": fill_color, "supercategory": ""})
        else:
            category_id = category_map[fill_color]

        area = float(np.sum(binary_mask))
        if area > 0:
            rle = ccc.binary_mask_to_uncompressed_rle(binary_mask)  # Use uncompressed RLE conversion
            bbox = ccc.uncompressed_rle_to_bbox(rle)
            annotations.append({
                "id": i,
                "image_id": 1,
                "category_id": category_id,
                "segmentation": ccc.rle_compress(rle),
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "name": object_id  # Add name field with SVG ID
            })

    coco_data = {
        "images": [{"id": 1, "width": intrinsic_size[0], "height": intrinsic_size[1], "file_name": image_filename}],
        "categories": categories,  # Categories come before annotations
        "annotations": annotations
    }

    output_json_path = os.path.join(output_folder, os.path.splitext(svg_path)[0] + '.json')
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Takes an SVG file, extracts the embedded raster image, and generates COCO JSON annotations.")
    parser.add_argument("name", help="Name <name> such that <name>.svg exists in the current directory.")
    args = parser.parse_args()

    svg_path = f"{args.name}.svg"
    output_folder = args.name

    os.makedirs(output_folder, exist_ok=True)

    generate_coco_annotations(svg_path, output_folder)
