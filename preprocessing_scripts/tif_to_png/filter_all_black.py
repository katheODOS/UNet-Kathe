import os

# Directories
json_dirs = [
    r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\rgb rendered mask\json_with_0_v2\json",
    r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\rgb rendered mask\masks\json"
]

tif_dirs = [
    r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\rgb rendered mask\json_with_0_v2",
    r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\rgb rendered mask\masks"
]

# Files to delete (JSON) based on the json files which only have '0' as a label
json_files_to_delete = [
    "biodiversity_0024.json",
    "biodiversity_0624.json",
    "biodiversity_0625.json",
    "biodiversity_0639.json",
    "biodiversity_0640.json",
    "biodiversity_0641.json",
    "biodiversity_0642.json",
    "biodiversity_0643.json",
    "biodiversity_0644.json",
    "biodiversity_0645.json",
    "biodiversity_0646.json",
    "biodiversity_0647.json",
    "biodiversity_0648.json"
]

# Generate corresponding TIF filenames (handle both possible formats)
tif_files_to_delete = []
for json_file in json_files_to_delete:
    number = json_file.split('_')[1].split('.')[0]  # Extract number (e.g., "0024")
    # Add both possible TIF filename formats
    tif_files_to_delete.append(f"biodiversity_0{int(number):d}.tif")  # Format without leading zeros

print("Will try to delete these TIF files:", tif_files_to_delete)

# Delete JSON files
for json_dir in json_dirs:
    for json_file in json_files_to_delete:
        file_path = os.path.join(json_dir, json_file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted JSON: {file_path}")

# Delete TIF files
for tif_dir in tif_dirs:
    for tif_file in tif_files_to_delete:
        file_path = os.path.join(tif_dir, tif_file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted TIF: {file_path}")
        else:
            print(f"TIF file not found: {file_path}")

print("File deletion complete!")
