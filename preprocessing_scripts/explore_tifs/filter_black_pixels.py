import os
import json
import shutil
import re

# Source and target directories
source_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\index\annotation_tif\json"
target_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\index\annotation_tif\json_with_14"
tif_source_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\index\annotation_tif"
tif_target_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\index\annotation_tif\json_with_14"

# Create directories if they don't exist
for directory in [target_dir, tif_target_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Check if target directory is truly empty (ignoring hidden files and subdirectories)
visible_files = [f for f in os.listdir(target_dir) 
                 if not f.startswith('.') and 
                 os.path.isfile(os.path.join(target_dir, f))]

if not visible_files:
    print(f"Processing JSON files from: {source_dir}")
    # Process all json files
    files_processed = 0
    
    for filename in os.listdir(source_dir):
        if 'biodiversity' in filename.lower() and filename.endswith('.json'):
            file_path = os.path.join(source_dir, filename)
            
            try:
                # Read and check json file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Check if '14' exists in label_counts
                    if '14' in data.get('label_counts', {}):
                        # Copy file to target directory
                        shutil.copy2(file_path, os.path.join(target_dir, filename))
                        files_processed += 1
                        print(f"Copied JSON file: {filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"JSON processing complete! Processed {files_processed} files.")
else:
    print(f"Found {len(visible_files)} files in target directory. Please empty the directory to process files again.")
    exit(1)

# Find and copy corresponding .tif files
for json_file in os.listdir(target_dir):
    if json_file.endswith('.json'):
        # Extract number from filename using regex
        numbers = re.findall(r'\d+', json_file)
        if numbers:
            # Get the last number in the filename
            file_number = numbers[-1]
            
            # Look for matching .tif file in the correct source directory
            for tif_file in os.listdir(tif_source_dir):
                if tif_file.endswith('.tif') and file_number in tif_file:
                    tif_source = os.path.join(tif_source_dir, tif_file)
                    tif_dest = os.path.join(tif_target_dir, tif_file)
                    shutil.copy2(tif_source, tif_dest)
                    print(f"Copied matching TIF file: {tif_file}")
                    break

print("TIF file processing complete!")
