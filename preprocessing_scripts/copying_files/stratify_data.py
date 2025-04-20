import os
import shutil

json_base_dir = r"directory/with/json/files/contaiing/info/on/each/label"
tif_source_dir = r"directory/with/tif/files/separated/by/secondary/label"
tif_base_dir = r"directory/to/copy/train_val/split/into"

# Process each section folder where the sections correspond to secondary labels

sec_folders = [d for d in os.listdir(json_base_dir) if d.startswith('sec ')]

# Create corresponding directories in test_train split if they don't exist; when I used this for Dataset A there were only 6 labels
for sec_folder in sec_folders:
    os.makedirs(os.path.join(tif_base_dir, sec_folder), exist_ok=True)

# Process each secondary label folder
for sec_folder in sec_folders:
    json_folder_path = os.path.join(json_base_dir, sec_folder)
    tif_target_folder = os.path.join(tif_base_dir, sec_folder)
    
    # Get list of JSON files in this secondary label folder
    json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
    
    for json_file in json_files:
        # Get the base name without extension
        base_name = os.path.splitext(json_file)[0]
        
        tif_name = base_name + '.tif'
        source_tif_path = os.path.join(tif_source_dir, tif_name)
        
        if os.path.exists(source_tif_path):
            target_tif_path = os.path.join(tif_target_folder, tif_name)
            shutil.copy2(source_tif_path, target_tif_path)
            print(f"Copied {tif_name} to {sec_folder}")
        else:
            print(f"Warning: No matching TIF found for {json_file}")

print("\nProcessing complete!")

for sec_folder in sec_folders:
    tif_count = len([f for f in os.listdir(os.path.join(tif_base_dir, sec_folder)) if f.endswith('.tif')])
    print(f"{sec_folder}: {tif_count} TIF files copied")
