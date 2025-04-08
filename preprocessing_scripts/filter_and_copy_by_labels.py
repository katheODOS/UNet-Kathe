import os
import json
import shutil
import re

# Source and target directories
json_source_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\index\annotation_tif\masks_json"
image_source_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\index\image_tif"
mask_source_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\index\annotation_tif"

image_target_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\index\augmented_image"
mask_target_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\index\augmented_annotation"

# Create target directories if they don't exist
for directory in [image_target_dir, mask_target_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def check_label_conditions(label_str):
    # Split the label string into individual labels
    labels = label_str.split('-')
    
    # Condition 1: Check for 2,3,6,12 as second label
    if len(labels) > 1 and labels[1] in ['2', '3', '6', '9','12', '13']:
        return True
        
    # # Condition 2: Check for 4 as first label
    if len(labels) > 0 and labels[0] == '4':
        return True
        
    # # Condition 3: Check for 9,12,13 anywhere in labels
    # if any(label in labels for label in ['9', '12', '13']):
    #     return True
        
    return False

files_processed = 0
# Process JSON files
for filename in os.listdir(json_source_dir):
    if not (filename.startswith('biodiversity_') and filename.endswith('.json')):
        continue

    json_path = os.path.join(json_source_dir, filename)
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
            # Check if the label matches our conditions
            if check_label_conditions(data['label']):
                base_name = os.path.splitext(filename)[0]
                
                # Copy corresponding image TIF
                image_tif = f"{base_name}.tif"
                image_source = os.path.join(image_source_dir, image_tif)
                image_target = os.path.join(image_target_dir, image_tif)
                
                # Copy corresponding mask TIF
                mask_source = os.path.join(mask_source_dir, image_tif)
                mask_target = os.path.join(mask_target_dir, image_tif)
                
                if os.path.exists(image_source) and os.path.exists(mask_source):
                    shutil.copy2(image_source, image_target)
                    shutil.copy2(mask_source, mask_target)
                    files_processed += 1
                    print(f"Copied files for {base_name} (Label: {data['label']})")
                else:
                    print(f"Missing TIF files for {base_name}")
                    
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

print(f"\nProcess completed. Copied {files_processed} pairs of files.")
