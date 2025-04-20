import os
import numpy as np
import tifffile

# Directory where files already exist and where rotations will be created
work_dir = r"directory/to/create/90/degree/rotations/inside/of"

def rotate_tiff(input_path, output_path, angle):
    """Rotate a TIFF file by the specified angle"""
    img_array = tifffile.imread(input_path)
    
    if angle == 90:
        rotated = np.rot90(img_array, k=1)
    elif angle == 180:
        rotated = np.rot90(img_array, k=2)
    elif angle == 270:
        rotated = np.rot90(img_array, k=3)
    
    tifffile.imwrite(output_path, rotated)

def process_files():
    """Process only the files that already exist in the working directory"""
    for filename in os.listdir(work_dir):
        if not filename.lower().endswith('.tif'):
            continue
            
        base_name = os.path.splitext(filename)[0]
        
        if any(deg in base_name for deg in ['90', '180', '270']):
            continue
            
        input_path = os.path.join(work_dir, filename)
        
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
