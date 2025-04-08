import os
import json
import rasterio
import numpy as np


folder_path = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\index\annotation_tif"

# Output JSON file path to write info to
output_json_path = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\index\annotation_tif\unique_values_green_index.json"

# These are the existing labels across the entire folder except for #000000 which accounts for pixels without corresponding info in the 'images' folder
color_labels = {
    "000000": 0,
    "27b341": 1,
    "e657c4": 2,
    "fc7ebb": 3,
    "ffcf4a": 4,
    "fa3e77": 5,
    "fa9441": 6,
    "adadad": 7,
    "ffc17a": 9,
    "a8e854": 12,
    "d9d9d9": 13,
    "21b341": 14
}

color_counts = {}
total_pixels = 0

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if not filename.lower().endswith(".tif"):
        continue

    file_path = os.path.join(folder_path, filename)
    print(f"Processing file: {file_path}")

    # Open the TIF file using rasterio
    with rasterio.open(file_path) as src:
        # Check that the file has at least 3 bands (RGB)
        if src.count < 3:
            print(f"  Warning: {filename} does not have at least 3 bands. Skipping.")
            continue

        # Read all bands (data shape: (bands, height, width))
        data = src.read()
        # Rearrange data to shape (height, width, bands)
        data = np.moveaxis(data, 0, -1)
        # Use only the first three channels for RGB
        rgb_data = data[..., :3]

        # Reshape the data to a 2D array where each row is a pixel ([R, G, B])
        pixels = rgb_data.reshape(-1, 3)
        # Update the total pixel counter
        total_pixels += pixels.shape[0]

        # Find unique colors and their counts
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

        # Loop over each unique color found in this file
        for color, count in zip(unique_colors, counts):
            # Make the color values into integers
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            # Convert the RGB values to a hex string in the format "#RRGGBB"
            hex_str = f"#{r:02x}{g:02x}{b:02x}"
            # Update the global dictionary count for this color
            color_counts[hex_str] = color_counts.get(hex_str, 0) + int(count)

# Calculate the percentage of each color
color_percentages = {}
for hex_str, count in color_counts.items():
    color_percentages[hex_str] = count / total_pixels if total_pixels > 0 else 0

# Create statistics keyed by numerical label from color_labels
labels_statistics = {}
for hex_val, label in color_labels.items():
    hex_with_hash = "#" + hex_val  # Convert to match the format in color_counts
    labels_statistics[label] = {
        "pixel_count": color_counts.get(hex_with_hash, 0),
        "color_percentage": color_percentages.get(hex_with_hash, 0)
    }

# Prepare the output data structure, including the new labels_statistics
output_data = {
    "total_pixel_count": total_pixels,
    "pixel_counts": color_counts,
    "color_percentages": color_percentages,
    "labels_statistics": labels_statistics
}

# Write the output to a JSON file
with open(output_json_path, "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"Processed {total_pixels} pixels across all files.")
print(f"Results have been saved to '{output_json_path}'.")
