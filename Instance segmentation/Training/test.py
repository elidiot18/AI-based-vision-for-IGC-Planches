import cv2
import numpy as np
import xml.etree.ElementTree as ET
import base64

def extract_image_from_svg(svg_path, output_path):
    """Extract the base64-encoded raster image from the SVG and save it using OpenCV."""
    # Read and parse the SVG file
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

    print(f"Image extracted and saved to {output_path}")

# Example usage
svg_path = 'italie2.svg'
output_path = 'extracted_image.png'
extract_image_from_svg(svg_path, output_path)
