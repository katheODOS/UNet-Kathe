import os
import shutil
import json
import re

# # Define directories
# source_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\rgb rendered mask\test_train split"
# test_dir = os.path.join(source_dir, "test")

# # Numbers to look for
# test_numbers = {89, 112, 157, 156, 166, 173, 238, 261, 380, 388, 412, 521}

# def get_number_from_filename(filename):
#     """Extract number from filename using regex"""
#     numbers = re.findall(r'\d+', filename)
#     return int(numbers[-1]) if numbers else None

# # Process files
# for filename in os.listdir(source_dir):
#     # Skip if it's a directory
#     if os.path.isdir(os.path.join(source_dir, filename)):
#         continue
        
#     number = get_number_from_filename(filename)
#     if number in test_numbers:
#         source_path = os.path.join(source_dir, filename)
#         dest_path = os.path.join(test_dir, filename)
#         shutil.move(source_path, dest_path)
#         print(f"Moved to test set: {filename}")

# print("Processing complete!")


# Define directories
json_source_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\rgb rendered mask\masks_json"
base_target_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\rgb rendered mask\masks_json"

# Create target directories if they don't exist
for i in range(1, 6):
    sec_dir = os.path.join(base_target_dir, f"sec {i}")
    os.makedirs(sec_dir, exist_ok=True)

def get_target_directory(label_str):
    """Determine which directory a file should go to based on its second label"""
    labels = label_str.split('-')
    if len(labels) < 2:  # Skip files with only one label
        return None
    
    second_label = int(labels[1])  # Get the second label
    
    # Map second labels to directory numbers
    label_map = {
        1: "sec 1",
        2: "sec 2",
        3: "sec 3",
        4: "sec 4",
        5: "sec 5"
    }
    
    return label_map.get(second_label)

# Process JSON files
for filename in os.listdir(json_source_dir):
    if not filename.endswith('.json'):
        continue
        
    file_path = os.path.join(json_source_dir, filename)
    
    # Read JSON file
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
            label_str = data.get('label')
            
            if label_str:
                target_subdir = get_target_directory(label_str)
                if target_subdir:
                    # Copy file to appropriate directory
                    target_dir = os.path.join(base_target_dir, target_subdir)
                    shutil.copy2(file_path, os.path.join(target_dir, filename))
                    print(f"Copied {filename} to {target_subdir} (Label: {label_str})")
                    
        except json.JSONDecodeError:
            print(f"Error reading JSON file: {filename}")
            continue

print("Processing complete!")
