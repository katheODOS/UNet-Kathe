import os
import shutil
from pathlib import Path


source_tiles = Path(r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\raster\tiles")
annotations_root = Path(r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\biodiversity_dataset\annotations")
images_root = Path(r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\biodiversity_dataset png\images")

# Create image subdirectories
for subdir in ['train', 'test', 'val']:
    os.makedirs(images_root / subdir, exist_ok=True)

# Build mapping of filenames to their destination folders
file_mapping = {}
for subdir in ['train', 'test', 'val']:
    annotation_dir = annotations_root / subdir
    if annotation_dir.exists():
        for file in annotation_dir.glob('*'):
            # Store base filename (without extension) and its corresponding subfolder
            base_name = file.stem
            file_mapping[base_name] = subdir

# Process and copy files
copied_files = 0
for tile_file in source_tiles.glob('*'):
    base_name = tile_file.stem
    if base_name in file_mapping:
        destination_subdir = file_mapping[base_name]
        destination_path = images_root / destination_subdir / tile_file.name
        
        # Copy the file
        shutil.copy2(tile_file, destination_path)
        copied_files += 1
        print(f"Copied {tile_file.name} to {destination_subdir}/")

print(f"\nComplete! Copied {copied_files} files to their respective folders.")
print(f"Files were copied from: {source_tiles}")
print(f"Files were copied to: {images_root}")
