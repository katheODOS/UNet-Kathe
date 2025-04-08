import os
import numpy as np
import tifffile

# Directory where files already exist and where rotations will be created
work_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\index\augmented_image"

def rotate_tiff(input_path, output_path, angle):
    """Rotate a TIFF file by the specified angle"""
    # Read TIFF using tifffile
    img_array = tifffile.imread(input_path)
    
    # Rotate using numpy (counterclockwise rotation)
    if angle == 90:
        rotated = np.rot90(img_array, k=1)
    elif angle == 180:
        rotated = np.rot90(img_array, k=2)
    elif angle == 270:
        rotated = np.rot90(img_array, k=3)
    
    # Save rotated image
    tifffile.imwrite(output_path, rotated)

def process_files():
    """Process only the files that already exist in the working directory"""
    for filename in os.listdir(work_dir):
        if not filename.lower().endswith('.tif'):
            continue
            
        # Get base name without extension
        base_name = os.path.splitext(filename)[0]
        
        # Skip if it's already a rotated file
        if any(deg in base_name for deg in ['90', '180', '270']):
            continue
            
        input_path = os.path.join(work_dir, filename)
        
        # Create rotated versions in same directory
        for angle in [90, 180, 270]:
            output_name = f"{base_name}_{angle}.tif"
            output_path = os.path.join(work_dir, output_name)
            
            try:
                rotate_tiff(input_path, output_path, angle)
                print(f"Created {output_name}")
            except Exception as e:
                print(f"Error processing {filename} with {angle}° rotation: {str(e)}")

def main():
    print(f"Processing files in: {work_dir}")
    process_files()
    print("\nRotation augmentation completed!")

if __name__ == "__main__":
    main()
