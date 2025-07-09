import os
import re
import json
from typing import Dict, List, Tuple

def parse_checkpoint_pairs(content: str) -> List[Tuple[int, float]]:
    """Extract checkpoint-validation score pairs by finding validation scores 
    that appear immediately before checkpoint saves."""
    # Split content into lines for easier processing
    lines = content.splitlines()
    pairs = []
    
    # Find all checkpoint save lines and their indices
    checkpoint_pattern = r"INFO: Checkpoint (\d+) saved!"
    validation_pattern = r"INFO: Validation Dice score: (\d+\.\d+)"
    
    for i, line in enumerate(lines):
        checkpoint_match = re.search(checkpoint_pattern, line)  # Changed from re.match to re.search
        if checkpoint_match:
            epoch = int(checkpoint_match.group(1))
            found_validation = False
            # Search backwards for the most recent validation score
            for prev_line in reversed(lines[:i]):
                validation_match = re.search(validation_pattern, prev_line)  # Changed from re.match to re.search
                if validation_match:
                    score = float(validation_match.group(1))
                    pairs.append((epoch, score))
                    found_validation = True
                    break
            if not found_validation:
                print(f"Warning: No validation score found for checkpoint {epoch}")
                print(f"Context around checkpoint {epoch}:")
                start_idx = max(0, i-5)
                end_idx = min(len(lines), i+5)
                print("\n".join(lines[start_idx:end_idx]))
                print("-" * 50)
    
    if not pairs:
        print("Debug: No pairs found in content")
        print("First few lines of content:")
        print("\n".join(lines[:10]))
        print("...")
        print("Last few lines of content:")
        print("\n".join(lines[-10:]))
    
    return sorted(pairs, key=lambda x: x[0])  # Sort by epoch number

def parse_validation_scores(content: str) -> List[float]:
    """Extract validation dice scores from log content."""
    pattern = r"INFO: Validation Dice score: (\d+\.\d+)"
    return [float(score) for score in re.findall(pattern, content)]

def parse_checkpoint_epochs(content: str) -> List[int]:
    """Extract checkpoint epoch numbers from log content."""
    pattern = r"INFO: Checkpoint (\d+) saved!"
    return [int(epoch) for epoch in re.findall(pattern, content)]

def process_run_folder(folder_path: str) -> Tuple[Dict[str, float], float, Tuple[int, float]]:
    """Process a single run folder and return metrics."""
    output_file = os.path.join(folder_path, 'output.txt')
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Extract checkpoint-validation pairs
        pairs = parse_checkpoint_pairs(content)
        if not pairs:
            raise ValueError("No valid checkpoint-validation score pairs found")
        
        # Create epoch-score mapping
        epoch_scores = {f"epoch_{epoch}": score for epoch, score in pairs}
        
        # Find best performance
        best_score = max(pair[1] for pair in pairs)
        best_epoch = next(epoch for epoch, score in pairs if score == best_score)
        
        # Get final performance
        final_score = pairs[-1][1]
        
        # Save individual JSON report
        json_path = os.path.join(folder_path, 'validation_scores.json')
        with open(json_path, 'w') as f:
            json.dump(epoch_scores, f, indent=4)
            
        return epoch_scores, final_score, (best_epoch, best_score)
        
    except FileNotFoundError:
        print(f"Warning: output.txt not found in {folder_path}")
        return {}, 0.0, (0, 0.0)
    except Exception as e:
        print(f"Error processing {folder_path}: {str(e)}")
        return {}, 0.0, (0, 0.0)

def main():
    checkpoints_dir = 'checkpoints'
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError("Checkpoints directory not found")
    
    summary_lines = []
    
    # Process each run folder
    for run_folder in os.listdir(checkpoints_dir):
        folder_path = os.path.join(checkpoints_dir, run_folder)
        if not os.path.isdir(folder_path):
            continue
            
        _, final_score, (best_epoch, best_score) = process_run_folder(folder_path)
        
        # Add to summary
        summary_lines.append(f"Run: {run_folder}")
        summary_lines.append(f"Final validation dice score: {final_score:.4f}")
        summary_lines.append(f"Best performance: Epoch {best_epoch}, Dice score: {best_score:.4f}")
        summary_lines.append("-" * 50)
    
    # Save summary report
    summary_path = os.path.join(checkpoints_dir, 'validation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

if __name__ == '__main__':
    main()
