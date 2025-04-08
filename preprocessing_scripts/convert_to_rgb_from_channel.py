import os
import numpy as np
from PIL import Image

# Define input file and output directory
INPUT_FILE = r"C:\Users\Admin\anaconda3\envs\unet\Pytorch-UNet\data\imgs\val\biodiversity_0054_OUT.png"
OUTPUT_DIR = r"C:\Users\Admin\anaconda3\envs\unet\Pytorch-UNet\data\original dataset index\predictions"

# Define the mapping from class ID to RGB values
CLASS_TO_RGB = {
    0: (11, 246, 210),   # ignore index
    1: (39, 179, 65),    # pasture class
    2: (230, 87, 196),   # woodland class
    3: (252, 126, 187),  # conifer class
    4: (255, 207, 74),   # shrub
    5: (250, 62, 119),   # hedgerow
    6: (250, 148, 65),   # seminatural grassland
    7: (173, 173, 173),  # artificial surface
    9: (255, 193, 122),  # bare field
    12: (168, 232, 84),  # arable
    13: (217, 217, 217), # artificial garden
}

def convert_single_to_rgb(image_path, output_path):
    """Convert a single-channel image to RGB using the class mapping."""
    # Load the single-channel image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Create RGB array
    height, width = img_array.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Convert each pixel value to its RGB equivalent
    for class_id, rgb_value in CLASS_TO_RGB.items():
        mask = img_array == class_id
        rgb_array[mask] = rgb_value
    
    # Convert to PIL Image and save
    rgb_img = Image.fromarray(rgb_array)
    rgb_img.save(output_path)

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get the filename from the input path
    filename = os.path.basename(INPUT_FILE)
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    try:
        convert_single_to_rgb(INPUT_FILE, output_path)
        print(f"Successfully converted {filename}")
        print(f"Saved to: {output_path}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()
