import numpy as np
import cv2
import base64


################################################
############ TOOLS
############


def uncompressed_rle_to_bbox(rle):
    height, width = tuple(rle['size'])
    counts = rle['counts']
    x_min, y_min = width, height
    x_max, y_max = 0, 0

    current_pos = 0
    fill_value = 0  # Start with 0, which means the first run is background

    for length in counts:
        if fill_value == 1:  # Only consider foreground runs
            for offset in range(length):
                x = current_pos // height
                y = current_pos % height
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                current_pos += 1
        else:
            current_pos += length  # Skip background run

        # Toggle between 0 and 1 for each run
        fill_value = 1 - fill_value

    return list(map(int, [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]))

def uncompressed_rle_area(rle):
    counts = rle['counts']
    area = 0
    fill_value = 0
    for length in counts:
        if fill_value == 1:  # Only consider foreground runs
            area += length
        fill_value = 1 - fill_value
    return area



################################################
############ DEBUG
############

def display_binary_mask(mask, window_name='Binary Mask'):
    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_color[mask > 0] = [255, 0, 0]  # Color the mask in red
    cv2.imshow(window_name, mask_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

################################################
############ RLE ENCODING
############

def mask_to_uncompressed_rle(binary_mask):
    """Convert a binary mask to uncompressed RLE format."""

    rle = {"counts": np.array([]), "size": list(binary_mask.shape)}
    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    counts = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    if flattened_mask[0] == 1:
        counts = np.concatenate(([0], counts))
    rle["counts"] = counts
    return rle

def uncompressed_rle_to_mask(rle):
    """Decode uncompressed RLE to binary mask."""
    size = tuple(rle['size'])
    counts = rle['counts']
    mask = np.zeros(np.prod(size))

    current_pos = 0
    is_foreground = False  # Alternate between background (0) and foreground (1)

    for length in counts:
        if current_pos < len(mask):
            if is_foreground:
                mask[current_pos:current_pos + length] = 1
            # Move to the next position
            current_pos += length
            # Toggle between background and foreground
            is_foreground = not is_foreground

    return mask.reshape(size, order='F')

def compressed_rle_to_mask(rle):
    new = rle.copy()
    new['counts'] = string_to_counts(rle['counts'])
    return uncompressed_rle_to_mask(new)

def rle_compress(rle):
    rle['counts'] = counts_to_string(rle['counts'])
    return rle


################################################
############ COUNT COMPRESSION IN BASE64
############


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
