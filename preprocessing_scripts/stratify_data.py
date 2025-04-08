import os
import shutil

json_base_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap\annotations\masks_json"
tif_source_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\rgb rendered mask\test_train split"
tif_base_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\rgb rendered mask\test_train split"

# Process each section folder where the sections correspond to secondary labels

sec_folders = [d for d in os.listdir(json_base_dir) if d.startswith('sec ')]

# Create corresponding directories in test_train split if they don't exist
for sec_folder in sec_folders:
    os.makedirs(os.path.join(tif_base_dir, sec_folder), exist_ok=True)

# Process each secondary label folder
for sec_folder in sec_folders:
    json_folder_path = os.path.join(json_base_dir, sec_folder)
    tif_target_folder = os.path.join(tif_base_dir, sec_folder)
    
    # Get list of JSON files in this secondary label folder
    json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
    
    # Process each JSON file
    for json_file in json_files:
        # Get the base name without extension
        base_name = os.path.splitext(json_file)[0]
        
        tif_name = base_name + '.tif'
        source_tif_path = os.path.join(tif_source_dir, tif_name)
        
        # If TIF exists, copy it to corresponding folder
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
