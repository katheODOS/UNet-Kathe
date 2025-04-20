import json
import os
from collections import defaultdict

json_dir = r"directory/with/tif/jsons"
output_path = r"directory/to/save/analysis/to/primary_label_frequencies.json"

label_frequencies = defaultdict(int)
total_files = 0

# The goal is to see what the distirbution of the most common lavbel in a given mask is 
for filename in os.listdir(json_dir):
    if not (filename.startswith('biodiversity_') and filename.endswith('.json')):
        continue

    file_path = os.path.join(json_dir, filename)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            # Get the label with the highest pixel count from label_counts
            label_counts = data.get('label_counts', {})
            if label_counts:
                # Convert counts to integers and find max
                primary_label = max(label_counts.items(), key=lambda x: int(x[1]))[0]
                label_frequencies[primary_label] += 1
                total_files += 1
                
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

# Calculate percentages and prepare output
output_stats = {
    "total_files_analyzed": total_files,
    "label_statistics": {}
}

# Convert frequencies to percentages and format output
for label, frequency in sorted(label_frequencies.items(), key=lambda x: x[1], reverse=True):
    percentage = (frequency / total_files) * 100
    output_stats["label_statistics"][label] = {
        "frequency": frequency,
        "percentage": round(percentage, 2)
    }

# Save results
with open(output_path, 'w') as f:
    json.dump(output_stats, f, indent=4)

print(f"Analysis complete! Results saved to: {output_path}")
print(f"\nTotal files analyzed: {total_files}")
print("\nPrimary Label Frequencies:")
for label, stats in output_stats["label_statistics"].items():
    print(f"Label {label}: {stats['frequency']} occurrences ({stats['percentage']}%)")
