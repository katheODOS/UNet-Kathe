import os
import shutil

# Define base directory
base_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented"

# Define source and target directories
sources_and_targets = [
    # Train annotations
    {
        'tif_dir': os.path.join(base_dir, "sorted", "train", "annotations", "tif"),
        'png_source': os.path.join(base_dir, "annotations", "png"),
        'png_target': os.path.join(base_dir, "sorted", "train", "annotations", "png")
    },
    # Val annotations
    {
        'tif_dir': os.path.join(base_dir, "sorted", "val", "annotations", "tif"),
        'png_source': os.path.join(base_dir, "annotations", "png"),
        'png_target': os.path.join(base_dir, "sorted", "val", "annotations", "png")
    },
    # Train images
    {
        'tif_dir': os.path.join(base_dir, "sorted", "train", "images", "tif"),
        'png_source': os.path.join(base_dir, "images", "png"),
        'png_target': os.path.join(base_dir, "sorted", "train", "images", "png")
    },
    # Val images
    {
        'tif_dir': os.path.join(base_dir, "sorted", "val", "images", "tif"),
        'png_source': os.path.join(base_dir, "images", "png"),
        'png_target': os.path.join(base_dir, "sorted", "val", "images", "png")
    }
]

def copy_corresponding_pngs(tif_dir, png_source, png_target):
    """Copy PNG files that correspond to TIFs from source to target directory"""
    # Create target directory if it doesn't exist
    os.makedirs(png_target, exist_ok=True)
    
    files_processed = 0
    files_missing = 0
    
    # Get list of TIF files
    tif_files = [f for f in os.listdir(tif_dir) if f.endswith('.tif')]
    
    for tif_file in tif_files:
        # Get base name without extension
        base_name = os.path.splitext(tif_file)[0]
        png_name = f"{base_name}.png"
        
        # Define source and target paths for PNG
        png_source_path = os.path.join(png_source, png_name)
        png_target_path = os.path.join(png_target, png_name)
        
        # Copy if source PNG exists
        if os.path.exists(png_source_path):
            shutil.copy2(png_source_path, png_target_path)
            files_processed += 1
        else:
            print(f"Warning: Missing PNG for {base_name}")
            files_missing += 1
    
    return files_processed, files_missing

# Process all directory pairs
for dirs in sources_and_targets:
    print(f"\nProcessing {dirs['tif_dir']}")
    processed, missing = copy_corresponding_pngs(
        dirs['tif_dir'],
        dirs['png_source'],
        dirs['png_target']
    )
    print(f"Files processed: {processed}")
    if missing > 0:
        print(f"Files missing: {missing}")

print("\nAll copying operations completed!")
