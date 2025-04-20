import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Define directories
source_dir = r"directory/with/augmented/pngs"
target_dir = r"directory/to/save/fixed/pngs/into"

ALPHA_THRESHOLD = 200  # Pixels whose alpha values below this will be replaced
TARGET_COLOR = np.array([11, 246, 210], dtype=np.uint8) 

def process_image(source_path, target_path):
    """Process a single image, replacing pixels with low alpha values"""
    try:
        img = Image.open(source_path)
             
        img_array = np.array(img)
        
        alpha_channel = img_array[:, :, 3]
        low_alpha_mask = alpha_channel < ALPHA_THRESHOLD
        
        if np.any(low_alpha_mask):
            new_img_array = img_array.copy()
            
            new_img_array[low_alpha_mask, 0:3] = TARGET_COLOR
            
            new_img_array[low_alpha_mask, 3] = 255
            
            corrected_img = Image.fromarray(new_img_array)
            corrected_img.save(target_path)
            return True, np.sum(low_alpha_mask)  # Return True and count of changed pixels
        else:
            img.save(target_path)
            return False, 0
    except Exception as e:
        raise Exception(f"Error processing {os.path.basename(source_path)}: {str(e)}")

def main():
    os.makedirs(target_dir, exist_ok=True)
    
    png_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.png'):
                # Get relative path from source directory
                rel_path = os.path.relpath(os.path.join(root, file), source_dir)
                png_files.append(rel_path)
    
    print(f"Found {len(png_files)} PNG files to process")
    
    files_modified = 0
    total_pixels_replaced = 0
    errors = []
    
    for rel_path in tqdm(png_files, desc="Processing images"):
        source_path = os.path.join(source_dir, rel_path)
        target_path = os.path.join(target_dir, rel_path)
        
        target_dir_path = os.path.dirname(target_path)
        os.makedirs(target_dir_path, exist_ok=True)
        
        try:
            modified, pixels_replaced = process_image(source_path, target_path)
            if modified:
                files_modified += 1
                total_pixels_replaced += pixels_replaced
        except Exception as e:
            errors.append(str(e))
            print(f"\n{str(e)}")
    
    print(f"\nProcessing complete!")
    print(f"Modified {files_modified} out of {len(png_files)} files")
    print(f"Total pixels replaced: {total_pixels_replaced}")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors during processing:")
        for i, error in enumerate(errors[:10], 1):
            print(f"{i}. {error}")
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more errors")

if __name__ == "__main__":
    main()
