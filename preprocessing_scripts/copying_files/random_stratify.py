import os
import shutil
import random
import math


base_dir = r"directory/with/files/to/copy"
source_ann_dir = os.path.join(base_dir, "sorted", "annotations", "tif")
source_img_dir = os.path.join(base_dir, "images")


train_ann_dir = os.path.join(base_dir, "sorted", "train", "annotations", "tif")
train_img_dir = os.path.join(base_dir, "sorted", "train", "images", "tif")
val_ann_dir = os.path.join(base_dir, "sorted", "val", "annotations", "tif")
val_img_dir = os.path.join(base_dir, "sorted", "val", "images", "tif")

# Labels to process and by 80% of each folder with files sorted based on primary or secondary label prevalence

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '12', '13']

# Create target directories
for directory in [train_ann_dir, train_img_dir, val_ann_dir, val_img_dir]:
    os.makedirs(directory, exist_ok=True)

def copy_corresponding_images(source_files, source_dir, target_dir):
    """Copy image files that correspond to annotation files"""
    for filename in source_files:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            print(f"Copied image: {filename}")
        else:
            print(f"Warning: Missing image for {filename}")

# Process each label folder
for label in LABELS:
    label_dir = os.path.join(source_ann_dir, label)
    if not os.path.exists(label_dir):
        print(f"Skipping label {label} - directory not found")
        continue
        
    tif_files = [f for f in os.listdir(label_dir) if f.endswith('.tif')]
    
    if not tif_files:
        print(f"No TIF files found in label {label}")
        continue
    
    # Split by 80% of each folder with files sorted based on primary or secondary label prevalence
    train_count = math.floor(len(tif_files) * 0.8)
    
    # Randomly select training files
    train_files = set(random.sample(tif_files, train_count))
    val_files = set(tif_files) - train_files
    
    
    for filename in train_files:
        source_path = os.path.join(label_dir, filename)
        target_path = os.path.join(train_ann_dir, filename)
        shutil.copy2(source_path, target_path)
        print(f"Copied to train annotations: {filename}")
    
    for filename in val_files:
        source_path = os.path.join(label_dir, filename)
        target_path = os.path.join(val_ann_dir, filename)
        shutil.copy2(source_path, target_path)
        print(f"Copied to val annotations: {filename}")
    
    print(f"\nProcessed label {label}:")
    print(f"Total files: {len(tif_files)}")
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")

# Copy corresponding images for train and val sets

print("\nCopying corresponding images for training set...")
train_ann_files = os.listdir(train_ann_dir)
copy_corresponding_images(train_ann_files, source_img_dir, train_img_dir)

print("\nCopying corresponding images for validation set...")
val_ann_files = os.listdir(val_ann_dir)
copy_corresponding_images(val_ann_files, source_img_dir, val_img_dir)

print("\nStratified split and image copying complete!")
