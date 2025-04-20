import os
import json
import rasterio
import numpy as np
from itertools import combinations

# Reuse color labels from pixel_color_distribution_masks.py
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

def convert_to_python_types(obj):
    """Convert numpy types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    return obj

def sort_dict_by_value(d, reverse=True):
    """Sort dictionary by value in descending order to have a bettery idea of majority classes/distribution."""
    if not d:  
        return d
    if isinstance(next(iter(d.values())), dict):
        return dict(sorted(d.items(), key=lambda x: x[1]['count'], reverse=reverse))
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))

def process_single_tif(file_path):
    """Process a TIF file and return its color statistics."""
    with rasterio.open(file_path) as src:
        data = src.read()
        data = np.moveaxis(data, 0, -1)
        rgb_data = data[..., :3]
        pixels = rgb_data.reshape(-1, 3)
        
        # Get unique colors and counts including black pixels for analysis
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        color_stats = {}
        pixel_frequencies = {}
        total_pixels = pixels.shape[0]  # Total pixels, including black NAN 
        
        for color, count in zip(unique_colors, counts):
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            hex_str = f"{r:02x}{g:02x}{b:02x}"
            frequency = count / total_pixels
            
            # Import numerical color labels I've been using to json file formatting
            if hex_str in color_labels:
                label = color_labels[hex_str]
                count_int = int(count)
                color_stats[label] = count_int
                pixel_frequencies[label] = frequency
            elif hex_str == "000000":  # Include black pixels
                pixel_frequencies[0] = frequency
        
        # Sort both dictionaries by descending value to see most common pixels first
        color_stats = sort_dict_by_value(color_stats)
        pixel_frequencies = sort_dict_by_value(pixel_frequencies)
        
        return convert_to_python_types(color_stats), total_pixels, convert_to_python_types(pixel_frequencies)

def analyze_existing_jsons(json_folder):
    """Analyze existing JSON files to generate combination statistics."""
    combinations_by_size = {}
    label_count_frequency = {}
    total_files = 0
    
    for filename in os.listdir(json_folder):
        if not filename.startswith("biodiversity_") or not filename.endswith(".json"):
            continue
            
        file_path = os.path.join(json_folder, filename)
        with open(file_path, 'r') as f:
            image_data = json.load(f)
            
        num_labels = image_data["number_of_labels"]
        # Sort labels by their pixel counts in descending order
        label_counts = image_data["label_counts"]
        present_labels = sorted(
            label_counts.items(),
            key=lambda x: int(x[1]),
            reverse=True
        )
        # Create combination key using sorted labels
        combo_key = "-".join(label for label, _ in present_labels)
        
        total_files += 1
        
        label_count_frequency[num_labels] = label_count_frequency.get(num_labels, 0) + 1
        
        if num_labels not in combinations_by_size:
            combinations_by_size[num_labels] = {}
            
        combinations_by_size[num_labels][combo_key] = combinations_by_size[num_labels].get(combo_key, 0) + 1
    
    return total_files, label_count_frequency, combinations_by_size

def main():
    input_folder = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\index\annotation_tif"
    json_output_folder = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\index\annotation_tif\json"
    
    os.makedirs(json_output_folder, exist_ok=True)
    
    # Track statistics
    combinations_by_size = {}
    label_count_frequency = {}
    total_files = 0
    
    tif_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif')]
    
    for filename in sorted(tif_files):
        input_path = os.path.join(input_folder, filename)
        # Create JSON filename by replacing .tif extension
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(json_output_folder, json_filename)
        
        print(f"Processing {filename} -> {json_filename}")
        
        # Process the TIF file
        color_stats, total_pixels, pixel_frequencies = process_single_tif(input_path)
        present_labels = sorted(color_stats.keys())
        
        # Use the actual length of present_labels for number_of_labels
        num_labels = len(present_labels)
        total_files += 1
        
        # Count frequency of number of labels
        label_count_frequency[num_labels] = label_count_frequency.get(num_labels, 0) + 1
        
        # Count combinations for exact number of labels
        if num_labels not in combinations_by_size:
            combinations_by_size[num_labels] = {}
            
        # Create exactly one combination using all present labels since all json files have corresponding string labels in data, accunting for multiple images with the same descending pxel order
        combo_key = ''.join(map(str, present_labels))
        combinations_by_size[num_labels][combo_key] = combinations_by_size[num_labels].get(combo_key, 0) + 1
        
        # Sort labels by their pixel count in descending order
        sorted_labels = sorted(
            color_stats.items(),
            key=lambda x: x[1],  # Sort by pixel count
            reverse=True  # Descending order
        )
        
        # Generate the unique labels string with labels ordered by pixel count
        unique_label_str = "-".join(str(label) for label, _ in sorted_labels)
        
        # Save the data with the new label ordering
        image_data = {
            "original_filename": filename,
            "total_pixels": total_pixels,
            "label_counts": color_stats,
            "pixel_frequencies": pixel_frequencies,
            "number_of_labels": num_labels,
            "label": unique_label_str
        }
        
        with open(json_path, 'w') as f:
            json.dump(convert_to_python_types(image_data), f, indent=4)

    # After processing all TIFs, analyze the JSON files
    total_files, label_count_frequency, combinations_by_size = analyze_existing_jsons(json_output_folder)
    
    # Calculate label count frequency statistics
    label_frequency_stats = {}
    # Sort by frequency first
    sorted_frequencies = sorted(label_count_frequency.items(), 
                              key=lambda x: (x[1], x[0]), 
                              reverse=True)
    
    for num_labels, frequency in sorted_frequencies:
        percentage = (frequency / total_files) * 100
        label_frequency_stats[num_labels] = {
            "count": frequency,
            "percentage": percentage
        }

    # Prepare the statistics output
    combination_stats = {
        "total_images": total_files,
        "label_frequency": {
            "description": "Distribution of number of labels across images",
            "statistics": label_frequency_stats
        },
        "combinations_by_label_count": {}
    }

    # Format combinations by number of labels, sorting by frequency
    for num_labels in sorted(combinations_by_size.keys()):
        sorted_combos = dict(sorted(combinations_by_size[num_labels].items(), 
                                  key=lambda x: x[1], 
                                  reverse=True))
        combination_stats["combinations_by_label_count"][num_labels] = {
            "total_images": label_count_frequency[num_labels],
            "combinations": sorted_combos
        }

    # Write statistics to file
    with open(os.path.join(json_output_folder, "label_combination_statistics.json"), 'w') as f:
        json.dump(convert_to_python_types(combination_stats), f, indent=4)

if __name__ == "__main__":
    main()

