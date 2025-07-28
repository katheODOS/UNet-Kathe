import os
import json
from pathlib import Path
import re

def extract_miou_score(miou_file):
    """Extract Mean IoU score from miou.txt file"""
    try:
        with open(miou_file, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = re.findall(r'Mean IoU \(excluding border pixels\):\s+(\d+\.\d+)', content)
            if matches:
                return float(matches[0])
    except Exception as e:
        print(f"Error processing {miou_file}: {str(e)}")
    return None

def process_checkpoints():
    """Process all checkpoint folders and create sorted output files"""
    checkpoint_dir = Path(r"C:\Users\Admin\anaconda3\envs\UNet-Kathe\UNet-Kathe\checkpoints\best_runs_scale_0.5")
    scores_dict = {}
    
    for folder in checkpoint_dir.iterdir():
        if folder.is_dir():
            miou_file = folder / 'results' / 'miou.txt'
            if miou_file.exists():
                score = extract_miou_score(miou_file)
                if score is not None:
                    scores_dict[folder.name] = score
    
    if not scores_dict:
        print("No Mean IoU scores found!")
        return
    
    with open(checkpoint_dir / 'miou_scores.json', 'w', encoding='utf-8') as f:
        json.dump(scores_dict, f, indent=4)
    
    sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    with open(checkpoint_dir / 'miou_scores_sorted.txt', 'w', encoding='utf-8') as f:
        for folder_name, score in sorted_scores:
            f.write(f"{folder_name}: {score:.4f}\n")
    
    print(f"Processed {len(scores_dict)} folders")
    print("Created miou_scores.json and miou_scores_sorted.txt")

if __name__ == '__main__':
    process_checkpoints()