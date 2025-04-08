import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from tqdm import tqdm

# Define specific input and output paths
INPUT_DIR = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\index\annotation_512"
OUTPUT_DIR = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\index\annotation_single_channel"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_hex_color(hex_color):
    """Convert hex color with or without # to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_color_mapping():
    """Create color mapping from hex to class ID."""
    color_map = {
        "0bf6d2": 0,
        "27b341": 1,
        "e657c4": 2,
        "fc7ebb": 3,
        "ffcf4a": 4,
        "fa3e77": 5,
        "fa9441": 6,
        "adadad": 7,
        "ffc17a": 9,
        "a8e854": 12,
        "d9d9d9": 13,
    }
    
    # Convert hex strings to RGB tuples
    rgb_map = {}
    for hex_color, class_id in color_map.items():
        rgb = parse_hex_color(hex_color)
        rgb_map[rgb] = class_id
        
    return rgb_map

def process_mask(mask_path, rgb_to_class, output_path):
    """Convert an RGB mask to a class ID mask and save it."""
    # Load the mask image
    mask_img = Image.open(mask_path)
    mask_array = np.array(mask_img)
    
    # Create an output array of the same height and width
    height, width = mask_array.shape[:2]
    class_mask = np.zeros((height, width), dtype=np.uint8)
    
    # For optimization, create a lookup for RGB values we've already processed
    rgb_lookup = {}
    
    # Process each pixel
    for y in range(height):
        for x in range(width):
            # Get the RGB value
            if mask_array.ndim == 3:  # RGB image
                rgb = tuple(mask_array[y, x])
            else:  # Grayscale image
                rgb = (mask_array[y, x], mask_array[y, x], mask_array[y, x])
            
            # Look up the class ID
            if rgb in rgb_lookup:
                class_id = rgb_lookup[rgb]
            elif rgb in rgb_to_class:
                class_id = rgb_to_class[rgb]
                rgb_lookup[rgb] = class_id
            else:
                # If color not found in mapping, use ignore index
                class_id = 255
                rgb_lookup[rgb] = class_id
                logging.warning(f"Unknown color {rgb} found in {mask_path}")
            
            class_mask[y, x] = class_id
    
    # Save the class mask
    class_img = Image.fromarray(class_mask)
    class_img.save(output_path)
    
    return class_mask

def main():
    setup_logging()
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Verify directories
    if not os.path.exists(INPUT_DIR):
        raise RuntimeError(f"Input directory not found: {INPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create color mapping
    rgb_to_class = create_color_mapping()
    logging.info(f"Created mapping with {len(rgb_to_class)} colors")
    
    # Print the mapping
    for rgb, class_id in rgb_to_class.items():
        logging.info(f"RGB {rgb} -> Class ID {class_id}")
    
    # Get list of mask files
    mask_files = [f for f in os.listdir(INPUT_DIR) 
                  if os.path.isfile(os.path.join(INPUT_DIR, f)) and 
                  f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    logging.info(f"Found {len(mask_files)} mask files to process")
    
    # Process each mask
    for mask_file in tqdm(mask_files, desc="Processing masks"):
        input_path = os.path.join(INPUT_DIR, mask_file)
        output_path = os.path.join(OUTPUT_DIR, mask_file)
        process_mask(input_path, rgb_to_class, output_path)
    
    logging.info("Conversion complete!")

if __name__ == "__main__":
    main()