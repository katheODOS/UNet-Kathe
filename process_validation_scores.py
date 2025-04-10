import os
import json
from pathlib import Path
import re

def extract_validation_score(output_file):
    """Extract validation Dice score from output file"""
    try:
        # Changed to use UTF-8 encoding
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = re.findall(r'wandb: validation Dice\s+(\d+\.\d+)', content)
            if matches:
                return float(matches[-1])
    except Exception as e:
        print(f"Error processing {output_file}: {str(e)}")
    return None

def process_checkpoints():
    """Process all checkpoint folders and create sorted output files"""
    checkpoint_dir = Path('./checkpoints')
    scores_dict = {}
    
    for folder in checkpoint_dir.iterdir():
        if folder.is_dir():
            output_file = folder / 'output.txt'
            if output_file.exists():
                score = extract_validation_score(output_file)
                if score is not None:
                    scores_dict[folder.name] = score
    
    if not scores_dict:
        print("No validation scores found!")
        return
    
    # Update file writing to use UTF-8
    with open(checkpoint_dir / 'validation_scores.json', 'w', encoding='utf-8') as f:
        json.dump(scores_dict, f, indent=4)
    
    # Create sorted scores text file with UTF-8 encoding
    sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    with open(checkpoint_dir / 'validation_scores_sorted.txt', 'w', encoding='utf-8') as f:
        for folder_name, score in sorted_scores:
            f.write(f"{folder_name}: {score:.4f}\n")
    
    print(f"Processed {len(scores_dict)} folders")
    print("Created validation_scores.json and validation_scores_sorted.txt")

if __name__ == '__main__':
    process_checkpoints()
