import os
from shutil import copyfile

# This script just standardizes naming conventions across the folders for data prep
def copy_and_rename_tifs(source_folder, destination_folder):
    """Copy and rename TIF files with biodiversity_xxxx naming convention."""
    
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Get list of TIF files
    tif_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.tif')]
    
    # Sort files to ensure consistent ordering
    tif_files.sort()
    
    # Process each file
    for idx, filename in enumerate(tif_files, 1):
        source_path = os.path.join(source_folder, filename)
        new_filename = f"biodiversity_{idx:04d}.tif"  # Zero-padded 4-digit number
        destination_path = os.path.join(destination_folder, new_filename)
        
        print(f"Processing {filename} -> {new_filename}")
        
        try:
            # Copy the file using shutil.copyfile
            copyfile(source_path, destination_path)
            print(f"Successfully copied and renamed: {new_filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # Define source and destination folder
    source_folder = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap\index\image_tif"
    destination_folder = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap\index\image_tif"
    
    copy_and_rename_tifs(source_folder, destination_folder)
    print("Renaming complete!")
