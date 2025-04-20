import json
import os
from collections import defaultdict

json_dir = r"directory/with/json/files"
output_path = r"directory/with/json/files/secondary_label_frequencies.json"

label_frequencies = defaultdict(int)
total_files = 0
single_label_files = []
processed_files = 0

# The goal is to generate jsons with ombinations of primary/secondary labels to better understand the distribution of labels across a dataset
for filename in os.listdir(json_dir):
    if not (filename.startswith('biodiversity_') and filename.endswith('.json')):
        continue
        
    total_files += 1
    file_path = os.path.join(json_dir, filename)
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            label_counts = data.get('label_counts', {})
            if len(label_counts) > 1:
                # Sort labels by their pixel counts in descending order
                sorted_labels = sorted(label_counts.items(), key=lambda x: int(x[1]), reverse=True)
                second_label = sorted_labels[1][0]
                label_frequencies[second_label] += 1
                processed_files += 1
            else:
                single_label_files.append(filename)
                
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

# Calculate percentages and prepare output
output_stats = {
    "total_files_found": total_files,
    "files_with_multiple_labels": processed_files,
    "files_with_single_label": len(single_label_files),
    "single_label_files": single_label_files,
    "label_statistics": {}
}

# Convert frequencies to percentages and format output
for label, frequency in sorted(label_frequencies.items(), key=lambda x: x[1], reverse=True):
    percentage = (frequency / processed_files) * 100
    output_stats["label_statistics"][label] = {
        "frequency": frequency,
        "percentage": round(percentage, 2)
    }

# Save results
with open(output_path, 'w') as f:
    json.dump(output_stats, f, indent=4)

print(f"\nAnalysis Summary:")
print(f"Total files found: {total_files}")
print(f"Files with multiple labels: {processed_files}")
print(f"Files with single label: {len(single_label_files)}")
print("\nSingle-label files:")
for file in single_label_files:
    print(f"- {file}")

print("\nSecondary Label Frequencies:")
for label, stats in output_stats["label_statistics"].items():
    print(f"Label {label}: {stats['frequency']} occurrences ({stats['percentage']}%)")
