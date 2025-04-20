import os
import json
import shutil
import re

json_source_dir = r"directory/with/json/files/to/analyze"
image_source_dir = r"directory/with/tifs/images/to/copy"
mask_source_dir = r"directory/with/tifs/annotations/to/copy"

image_target_dir = r"directory/with/tifs/images/to/copy/into"
mask_target_dir = r"directory/with/tifs/annotations/to/copy/into"
subdirs = {
    "0_single": "0_single_label",
    "2_primary": "2_primary_label",
    "3_any": "3_primary_secondary_label",
    "4_special": "4_primary_secondary_label",
    "6_any": "6_primary_secondary_label",
    "9_any": "9_any_position",
    "12_primary": "12_primary_label",
    "13_special": "13_secondary_label"
}

def verify_directories():
    """Verify all necessary directories exist"""
    missing_dirs = []
    for subdir in subdirs.values():
        mask_dir = os.path.join(base_masks_dir, subdir)
        img_dir = os.path.join(base_imgs_dir, subdir)
        if not os.path.exists(mask_dir):
            missing_dirs.append(mask_dir)
        if not os.path.exists(img_dir):
            missing_dirs.append(img_dir)
    
    if missing_dirs:
        raise RuntimeError(f"Missing required directories:\n" + "\n".join(missing_dirs))
    print("All required directories exist.")

def check_label_conditions(label_str, label_counts):
    """Check if file meets any of our conditions"""
    labels = label_str.split('-')
    
    conditions = {}
    
    # Check for single label 0
    conditions["0_single"] = len(labels) == 1 and labels[0] == '0'
    
    # Check for primary label 2
    conditions["2_primary"] = labels[0] == '2'
    
    # Check for 3 in primary or secondary position
    conditions["3_any"] = '3' in labels[:2]
    
    # Check for 4 conditions (primary or secondary without 1 primary)
    conditions["4_special"] = ('4' in labels[:2]) and (labels[0] != '1')
    
    # Check for 6 in any position
    conditions["6_any"] = '6' in labels[:2]
    
    # Check for 9 in any position without 1 primary
    conditions["9_any"] = '9' in labels and labels[0] != '1'
    
    # Check for 12 as primary
    conditions["12_primary"] = labels[0] == '12'
    
    # Check for 13 as primary or secondary without 1 primary
    conditions["13_special"] = '13' in labels[:2] and labels[0] != '1'
    
    return conditions

def copy_files(base_name, conditions):
    """Copy files to appropriate directories based on conditions"""
    files_copied = 0
    
    # Source files - both TIF and PNG
    source_files = {
        'mask': {
            'tif': os.path.join(mask_source_dir, f"{base_name}.tif"),
            'png': os.path.join(mask_source_dir, f"{base_name}.png")
        },
        'image': {
            'tif': os.path.join(image_source_dir, f"{base_name}.tif"),
            'png': os.path.join(image_source_dir, f"{base_name}.png")
        }
    }
    
    # Check if at least one format exists for each type
    if not any(os.path.exists(f) for f in source_files['mask'].values()) or \
       not any(os.path.exists(f) for f in source_files['image'].values()):
        return 0
    
    # Copy to each matching condition directory
    for condition, matches in conditions.items():
        if matches:
            # Copy mask files
            for ext in ['tif', 'png']:
                if os.path.exists(source_files['mask'][ext]):
                    mask_target = os.path.join(base_masks_dir, subdirs[condition], f"{base_name}.{ext}")
                    shutil.copy2(source_files['mask'][ext], mask_target)
                
                if os.path.exists(source_files['image'][ext]):
                    img_target = os.path.join(base_imgs_dir, subdirs[condition], f"{base_name}.{ext}")
                    shutil.copy2(source_files['image'][ext], img_target)
            
            files_copied += 1
            print(f"Copied {base_name} to {subdirs[condition]}")
            
    return files_copied

def main():
    verify_directories()
    files_processed = 0
    
    # Process each JSON file
    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue
            
        json_path = os.path.join(json_dir, filename)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            base_name = os.path.splitext(filename)[0]
            conditions = check_label_conditions(data['label'], data['label_counts'])
            
            files_copied = copy_files(base_name, conditions)
            if files_copied > 0:
                files_processed += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"\nProcessing complete! Files processed: {files_processed}")

if __name__ == "__main__":
    main()
