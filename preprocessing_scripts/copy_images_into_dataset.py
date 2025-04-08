import os
import shutil

# Define directories
annotations_dir = r"C:\Users\Admin\anaconda3\envs\unet\Pytorch-UNet\data\overlap augmented index\masks rgb\train"
source_images_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\index\image_512"
target_images_dir = r"C:\Users\Admin\anaconda3\envs\unet\Pytorch-UNet\data\overlap augmented index\imgs\train"

# Print directory contents for debugging
print(f"\nChecking source directory: {source_images_dir}")
print("Files found:", os.listdir(source_images_dir)[:5], "...")  # Show first 5 files

# Get list of annotation filenames without extension
annotation_files = [os.path.splitext(f)[0] for f in os.listdir(annotations_dir)]
print(f"\nFirst few annotation files:", annotation_files[:5], "...")

# Counter for tracking
copied_count = 0
not_found_count = 0

# Try both .tif and .png extensions
extensions = ['.tif', '.png']

# Process each annotation filename
for base_name in annotation_files:
    found = False
    for ext in extensions:
        source_path = os.path.join(source_images_dir, f"{base_name}{ext}")
        target_path = os.path.join(target_images_dir, f"{base_name}{ext}")
        
        if os.path.exists(source_path):
            # Copy the file
            shutil.copy2(source_path, target_path)
            copied_count += 1
            print(f"Copied: {base_name}{ext}")
            found = True
            break
    
    if not found:
        not_found_count += 1
        print(f"Warning: No matching image found for {base_name} (tried: {extensions})")

# Print summary
print("\nProcessing complete!")
print(f"Total annotations: {len(annotation_files)}")
print(f"Images copied: {copied_count}")
print(f"Images not found: {not_found_count}")
