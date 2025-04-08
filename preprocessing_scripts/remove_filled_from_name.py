import os
import shutil

# Source and ta# ALl this script does is rename the files that were run through 'fill_black_pixels.py' 
source_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\rgb rendered mask\copy of masks\no zeroes"
target_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\rgb rendered mask\copy of masks\no zeroes\no filled in name"

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Process files
for filename in os.listdir(source_dir):
    if filename.endswith('.tif') and '_filled' in filename:
        # Create new filename without '_filled'
        new_filename = filename.replace('_filled', '')
        
        # Source and destination paths
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(target_dir, new_filename)
        
        # Copy file with new name
        shutil.copy2(source_path, dest_path)
        print(f"Processed: {filename} -> {new_filename}")

print("All files processed!")
