import os
import shutil

annotations_dir = r"directory/you//copy/the/specific/files/from"
source_images_dir = r"directory/with/all/the/files/you/need"
target_images_dir = r"directory/you/copy/the/exact/same/files/from/annotations_dir/into"

print(f"\nChecking source directory: {source_images_dir}")
print("Files found:", os.listdir(source_images_dir)[:5], "...")  # Show first 5 files

annotation_files = [os.path.splitext(f)[0] for f in os.listdir(annotations_dir)]
print(f"\nFirst few annotation files:", annotation_files[:5], "...")

copied_count = 0
not_found_count = 0

# Flexbility in copying files
extensions = ['.tif', '.png']

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
