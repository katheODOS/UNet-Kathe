import os
import shutil
import re

json_dir = r"directory/with/select/jsons"
source_tif_dir = r"directory/with/select/masks"
target_tif_dir = r"directory/to/copy/masks/into"

# Get numbers from JSON filenames
def get_number_from_filename(filename):
    numbers = re.findall(r'\d+', filename)
    return numbers[-1] if numbers else None

for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        number = get_number_from_filename(json_file)
        if number:
            for tif_file in os.listdir(source_tif_dir):
                if tif_file.endswith('.tif') and number in tif_file:
                    source_path = os.path.join(source_tif_dir, tif_file)
                    target_path = os.path.join(target_tif_dir, tif_file)
                    
                    # Copy the file
                    shutil.copy2(source_path, target_path)
                    print(f"Copied: {tif_file}")
                    break

print("Processing complete!")
