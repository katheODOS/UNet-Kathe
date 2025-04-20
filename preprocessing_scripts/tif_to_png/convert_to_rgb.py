import os
from PIL import Image

root_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap augmented\index\image_512"

converted_count = 0
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path) as img:
                    # Only convert if not already RGB
                    if img.mode != 'RGB':
                        print(f"Converting {file} from {img.mode} to RGB")
                        rgb_img = img.convert('RGB')
                        rgb_img.save(file_path)
                        converted_count += 1
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

print(f"\nProcess completed. Converted {converted_count} files to RGB mode.")
