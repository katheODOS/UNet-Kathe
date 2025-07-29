import os
import torch
import logging
from pathlib import Path
import argparse
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
import json

def get_last_checkpoint(model_dir):
    """Find the last checkpoint in the given directory"""
    checkpoints = list(model_dir.glob('checkpoint_epoch*.pth'))
    if not checkpoints:
        return None
    # Sort by epoch number and get the last one
    return max(checkpoints, key=lambda x: int(x.stem.split('_epoch')[-1]))

def get_model_classes(state_dict):
    """Detect the number of classes from the model state dict"""
    if 'outc.conv.bias' in state_dict:
        return len(state_dict['outc.conv.bias'])
    # Alternative detection methods
    for key in state_dict:
        if key.endswith('.conv.bias') and 'outc' in key:
            return len(state_dict[key])
    return None  # Unable to determine

def check_all_checkpoints(checkpoints_dir, weights_only=True):
    """Check the number of classes in all checkpoint directories"""
    checkpoints_dir = Path(checkpoints_dir)
    results = []
    
    # Find all directories in checkpoints
    model_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
    
    for model_dir in tqdm(model_dirs, desc="Checking checkpoints"):
        # Find the last checkpoint in this directory
        last_checkpoint = get_last_checkpoint(model_dir)
        if not last_checkpoint:
            results.append({
                'model_name': model_dir.name,
                'checkpoint': None,
                'n_classes': 'No checkpoint found',
                'status': 'Missing'
            })
            continue
        
        try:
            # Explicitly set weights_only=True to avoid warning
            state_dict = torch.load(str(last_checkpoint), map_location='cpu', weights_only=True)
            if 'mask_values' in state_dict:
                mask_values = state_dict['mask_values']
                state_dict.pop('mask_values')
            else:
                mask_values = None
            
            # Get number of classes
            n_classes = get_model_classes(state_dict)
            
            results.append({
                'model_name': model_dir.name,
                'checkpoint': last_checkpoint.name,
                'n_classes': n_classes,
                'status': 'OK',
                'mask_values': mask_values
            })
        except Exception as e:
            results.append({
                'model_name': model_dir.name,
                'checkpoint': last_checkpoint.name if last_checkpoint else None,
                'n_classes': 'Error',
                'status': str(e),
            })
    
    return results

def save_results(results, output_dir):
    """Save results to CSV, JSON and print to console"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Save as CSV
    csv_path = output_dir / 'model_classes.csv'
    df.to_csv(csv_path, index=False)
    
    # Save as JSON for easier parsing
    json_path = output_dir / 'model_classes.json'
    
    # Convert DataFrame to a list of dictionaries for JSON
    records = df.to_dict(orient='records')
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)
    
    # Print as table
    print("\nModel Classes Report:")
    print(tabulate(df, headers='keys', tablefmt='grid'))
    
    # Print summary
    class_counts = df['n_classes'].value_counts().to_dict()
    print("\nSummary:")
    for classes, count in class_counts.items():
        print(f"Models with {classes} classes: {count}")
    
    print(f"\nDetailed results saved to {csv_path} and {json_path}")

def main():
    parser = argparse.ArgumentParser(description='Check number of classes in model checkpoints')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save results')
    parser.add_argument('--single_file', type=str, 
                       help='Path to a single .pth file to check')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if args.single_file:
        try:
            state_dict = torch.load(args.single_file, map_location='cpu', weights_only=True)
            if 'mask_values' in state_dict:
                mask_values = state_dict.pop('mask_values')
                print(f"\nMask values: {mask_values}")
            
            n_classes = get_model_classes(state_dict)
            print(f"\nFile: {args.single_file}")
            print(f"Number of classes: {n_classes}")
        except Exception as e:
            print(f"Error loading file: {str(e)}")
    else:
        print(f"Checking model checkpoints in {args.checkpoints_dir}...")
        results = check_all_checkpoints(args.checkpoints_dir)
        save_results(results, args.output_dir)
    parser = argparse.ArgumentParser(description='Check number of classes in model checkpoints')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save results')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print(f"Checking model checkpoints in {args.checkpoints_dir}...")
    results = check_all_checkpoints(args.checkpoints_dir)
    save_results(results, args.output_dir)

if __name__ == "__main__":
    main()
