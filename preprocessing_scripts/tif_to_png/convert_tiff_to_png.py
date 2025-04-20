import os
import numpy as np
import tifffile
from PIL import Image

root_dir = r"directory/with/tif/files"
output_root = r"directory/to/save/pngs/into"

if not os.path.exists(output_root):
    os.makedirs(output_root)

if not os.path.exists(root_dir):
    print(f"Error: Directory not found: {root_dir}")
    exit(1)

print(f"Starting conversion from: {root_dir}")
print(f"Saving PNGs to: {output_root}")

for subdir, dirs, files in os.walk(root_dir):
    print(f"Processing directory: {subdir}")
    
    rel_path = os.path.relpath(subdir, root_dir)
    output_subdir = os.path.join(output_root, rel_path)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    
    for file in files:
        if file.lower().endswith(('.tif', '.tiff')):
            try:
                input_path = os.path.join(subdir, file)
                
                # Read TIFF using tifffile
                tiff_array = tifffile.imread(input_path)
                
                # Ensure the array is uint8 without normalizing to avoid interpolation
                if tiff_array.dtype != np.uint8:
                    tiff_array = tiff_array.astype(np.uint8)
                
                # Convert numpy array to PIL Image
                img = Image.fromarray(tiff_array)
                
                # Create output filename in new directory
                base_name = os.path.splitext(file)[0]
                output_path = os.path.join(output_subdir, base_name + '.png')
                
                # Save as PNG without any optimization
                img.save(output_path, 'PNG', optimize=False)
                print(f"Converted: {file} -> {output_path}")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                print(f"Full path: {input_path}")
                print(f"File exists: {os.path.exists(input_path)}")
                print(f"File size: {os.path.getsize(input_path) if os.path.exists(input_path) else 'N/A'}")

print("Conversion process completed!")