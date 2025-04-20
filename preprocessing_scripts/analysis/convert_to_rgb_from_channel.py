import os
import numpy as np
from PIL import Image
from tqdm import tqdm

INPUT_DIR = r"your/directory/with/singlechannel/predictions/here"
OUTPUT_DIR = r"your/directory/to/save/rgb/predictions/here"

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
    img = Image.open(image_path)
    img_array = np.array(img)
    
    height, width = img_array.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_id, rgb_value in CLASS_TO_RGB.items():
        mask = img_array == class_id
        rgb_array[mask] = rgb_value
    
    rgb_img = Image.fromarray(rgb_array)
    rgb_img.save(output_path)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.png')]
    print(f"Found {len(image_files)} images to process")
    
    # Process each image with progress bar if you do images in batches
    for filename in tqdm(image_files, desc="Converting images"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            convert_single_to_rgb(input_path, output_path)
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
    
    print("\nConversion complete!")
    print(f"Processed images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
