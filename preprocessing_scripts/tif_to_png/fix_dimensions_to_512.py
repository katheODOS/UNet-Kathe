import os
from PIL import Image
import numpy as np

# Target size and background color so the models don't complain about mismatching dimensions. Applies to both images and annotations
TARGET_SIZE = (512, 512)
BACKGROUND_COLOR = (11, 246, 210)  

# Define input and output directories
input_dir = r"directory/with/pngs/to/make/512x512"
output_dir = r"directory/to/copy/new/512x512/files/into"


if not os.path.exists(output_dir):
    raise RuntimeError(f"Output directory does not exist: {output_dir}")

def resize_with_padding(image):
    """Resize image to 512x512 with specified background color, expanding from top-left"""
    if image.size == TARGET_SIZE:
        return image
    
    # Create new background with specified color
    new_image = Image.new('RGB', TARGET_SIZE, BACKGROUND_COLOR)
    
    # Paste original image at top-left (0,0)
    new_image.paste(image, (0, 0))
    return new_image

# Process all files
modified_count = 0
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            with Image.open(input_path) as img:
                if img.size != TARGET_SIZE:
                    print(f"Processing {filename} from {img.size} to {TARGET_SIZE}")
                    new_img = resize_with_padding(img)
                    new_img.save(output_path)
                    modified_count += 1
                else:
                    # If already correct size, just copy to new location
                    new_img = resize_with_padding(img)
                    new_img.save(output_path)
                    print(f"Copied {filename} (already {TARGET_SIZE})")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print(f"\nProcess completed. Processed {modified_count} files to {TARGET_SIZE} dimensions.")
print(f"All files saved to: {output_dir}")
