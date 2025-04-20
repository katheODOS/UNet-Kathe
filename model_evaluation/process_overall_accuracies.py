import os
import json
from pathlib import Path
import re

def extract_overall_accuracy(eval_file):
    """Extract Overall Accuracy from evaluation_report.txt"""
    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'Overall Accuracy: (\d+\.\d+)', content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Error processing {eval_file}: {str(e)}")
    return None

def process_accuracies():
    """Process all results folders and create sorted accuracy files"""
    checkpoint_dir = Path('./checkpoints')
    accuracies_dict = {}
    
    # Iterate through checkpoint subdirectories
    for folder in checkpoint_dir.iterdir():
        if folder.is_dir():
            results_dir = folder / 'results'
            eval_file = results_dir / 'evaluation_report.txt'
            
            if results_dir.exists() and eval_file.exists():
                accuracy = extract_overall_accuracy(eval_file)
                if accuracy is not None:
                    accuracies_dict[folder.name] = accuracy
    
    if not accuracies_dict:
        print("No overall accuracies found!")
        return
    
    # Save full accuracies dictionary to JSON
    with open(checkpoint_dir / 'overall_accuracies.json', 'w', encoding='utf-8') as f:
        json.dump(accuracies_dict, f, indent=4)
    
    # Create sorted accuracies text file
    sorted_accuracies = sorted(accuracies_dict.items(), key=lambda x: x[1], reverse=True)
    with open(checkpoint_dir / 'overall_accuracies_sorted.txt', 'w', encoding='utf-8') as f:
        for folder_name, accuracy in sorted_accuracies:
            f.write(f"{folder_name}: {accuracy:.4f}\n")
    
    print(f"Processed {len(accuracies_dict)} folders")
    print("Created overall_accuracies.json and overall_accuracies_sorted.txt")

if __name__ == '__main__':
    process_accuracies()
