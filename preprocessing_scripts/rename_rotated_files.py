import os
import shutil
import re

# Define directories
base_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\augmented"
images_dir = os.path.join(base_dir, "images")
annotations_dir = os.path.join(base_dir, "annotations")
renamed_images_dir = os.path.join(base_dir, "renamed_images")
renamed_annotations_dir = os.path.join(base_dir, "renamed_annotations")

# Create target directories if they don't exist
for directory in [renamed_images_dir, renamed_annotations_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_rotated_files(directory):
    """Get all rotated files (ending in _90, _180, _270) sorted by original number and rotation"""
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.tif'):
            # Check if file is a rotated version
            match = re.match(r'biodiversity_(\d+)_(\d+)\.tif', filename)
            if match:
                original_num = int(match.group(1))
                rotation = int(match.group(2))
                files.append((filename, original_num, rotation))
    
    # Sort by original number first, then by rotation
    return sorted(files, key=lambda x: (x[1], x[2]))

def process_directories():
    # Get rotated files from both directories
    image_files = get_rotated_files(images_dir)
    annotation_files = get_rotated_files(annotations_dir)
    
    # Start numbering from 2441
    new_number = 2441
    
    # Process files in pairs to maintain correspondence
    for (img_file, _, _), (ann_file, _, _) in zip(image_files, annotation_files):
        # Create new filenames
        new_name = f"biodiversity_{new_number}.tif"
        
        # Copy and rename image file
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(renamed_images_dir, new_name)
        shutil.copy2(src_img, dst_img)
        
        # Copy and rename annotation file
        src_ann = os.path.join(annotations_dir, ann_file)
        dst_ann = os.path.join(renamed_annotations_dir, new_name)
        shutil.copy2(src_ann, dst_ann)
        
        print(f"Renamed {img_file} -> {new_name}")
        new_number += 1

def main():
    print("Starting file renaming process...")
    process_directories()
    print("\nRenaming process completed!")

if __name__ == "__main__":
    main()
