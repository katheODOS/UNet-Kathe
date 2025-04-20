import os
import json
import shutil


json_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\annotations\masks_json"


ann_tif_source = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\annotations"
ann_png_source = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\annotations\png"
img_tif_source = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\images"
img_png_source = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\images\png"


sorted_base = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\sorted"
ann_tif_target = os.path.join(sorted_base, "annotations", "tif")
ann_png_target = os.path.join(sorted_base, "annotations", "png")
img_tif_target = os.path.join(sorted_base, "images", "tif")
img_png_target = os.path.join(sorted_base, "images", "png")


SECONDARY_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '12', '13']

def ensure_directories():
    """Create all necessary target directories if they don't exist"""
    created_dirs = []
    for label in SECONDARY_LABELS:
        for target_base in [ann_tif_target, ann_png_target, img_tif_target, img_png_target]:
            dir_path = os.path.join(target_base, label)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                created_dirs.append(dir_path)
    
    if created_dirs:
        print("Created the following directories:")
        for dir_path in created_dirs:
            print(f"- {dir_path}")

def process_files():
    files_processed = 0
    errors = []
    
    for filename in os.listdir(json_dir):
        if 'biodiversity' not in filename or not filename.endswith('.json'):
            continue

        json_path = os.path.join(json_dir, filename)
        try:

            with open(json_path, 'r') as f:
                data = json.load(f)
            
            label_parts = data['label'].split('-')
            if len(label_parts) < 2:
                continue
            
            secondary_label = label_parts[1]
            base_name = os.path.splitext(filename)[0]
            
            # Define all source and target paths
            sources_and_targets = [
                (os.path.join(ann_tif_source, f"{base_name}.tif"),
                 os.path.join(ann_tif_target, secondary_label, f"{base_name}.tif")),
                (os.path.join(ann_png_source, f"{base_name}.png"),
                 os.path.join(ann_png_target, secondary_label, f"{base_name}.png")),
                (os.path.join(img_tif_source, f"{base_name}.tif"),
                 os.path.join(img_tif_target, secondary_label, f"{base_name}.tif")),
                (os.path.join(img_png_source, f"{base_name}.png"),
                 os.path.join(img_png_target, secondary_label, f"{base_name}.png"))
            ]

            for source, target in sources_and_targets:
                if os.path.exists(source):
                    shutil.copy2(source, target)
                else:
                    errors.append(f"Missing file: {source}")

            files_processed += 1
            print(f"Processed {base_name} (Secondary label: {secondary_label})")

        except Exception as e:
            errors.append(f"Error processing {filename}: {str(e)}")

    return files_processed, errors

def main():
    print("Starting file sorting process...")
    files_processed, errors = process_files()
    
    print(f"\nProcessing complete!")
    print(f"Files processed: {files_processed}")
    
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"- {error}")

if __name__ == "__main__":
    main()
