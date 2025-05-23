import sys
import os
import argparse
import logging
from pathlib import Path
import datetime
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from utils.data_loading import BasicDataset
    from unet import UNet
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.colors as mcolors

ORIGINAL_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 13]
CLASS_NAMES = ['Border Pixels', 'Pasture', 'Woodland', 'Conifer', 'Shrub', 
               'Hedgerow', 'Semi-natural Grassland', 'Artificial Surface', 
               'Bare Field', 'Arable', 'Artificial Garden']

def evaluate_model(net, dataloader, device, n_classes):
    net.eval()
    confusion_mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    # For storing per-image metrics
    class_accuracies = {i: [] for i in range(n_classes)}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            true_masks = batch['mask'].cpu().numpy()
            
            # Predict
            outputs = net(images)
            probs = outputs.softmax(dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            
            # Update confusion matrix
            for true, pred in zip(true_masks, preds):
                confusion_mat += confusion_matrix(
                    true.flatten(), 
                    pred.flatten(), 
                    labels=range(n_classes)
                )
                
                # Calculate per-class accuracy for this image
                for class_idx in range(n_classes):
                    mask = (true == class_idx)
                    if mask.sum() > 0:  # Only if class exists in ground truth
                        accuracy = (pred[mask] == class_idx).mean()
                        class_accuracies[class_idx].append(accuracy)
    
    return confusion_mat, class_accuracies


def plot_results(confusion_mat, class_accuracies, save_dir):
    # Convert confusion matrix to percentages (row normalization)
    conf_mat_percent = np.zeros_like(confusion_mat, dtype=float)
    for i in range(confusion_mat.shape[0]):
        if confusion_mat[i].sum() > 0:  # Avoid division by zero
            conf_mat_percent[i] = confusion_mat[i] / confusion_mat[i].sum()
    
    # Plot confusion matrix with percentages
    plt.figure(figsize=(12, 10))
    
    ax = sns.heatmap(conf_mat_percent, annot=True, fmt='.2f', cmap='YlGn',
                    vmin=0, vmax=1.0, cbar_kws={'label': ''})
    
    plt.title('Overall confusion matrix', fontsize=16)
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    
    if len(CLASS_NAMES) == confusion_mat.shape[0]:
        tick_positions = np.arange(len(CLASS_NAMES)) + 0.5
        plt.xticks(tick_positions, CLASS_NAMES, rotation=45, ha='right', fontsize=10)
        plt.yticks(tick_positions, CLASS_NAMES, rotation=0, fontsize=10)
    else:
        tick_positions = np.arange(len(ORIGINAL_CLASSES)) + 0.5
        plt.xticks(tick_positions, ORIGINAL_CLASSES, rotation=45)
        plt.yticks(tick_positions, ORIGINAL_CLASSES, rotation=0)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Get the model name from the save_dir - go up one level since we're in 'results' subfolder
    model_name = Path(save_dir).parent.name  # This will get the full folder name (e.g., 'ASARWeight_LR1e-7_e10_B1')
    
    # Plot per-class accuracies with enhanced styling
    plt.figure(figsize=(15, 8))
    accuracies = [np.mean(vals) if vals else 0 for vals in class_accuracies.values()]
    
    cmap = plt.get_cmap('YlGn')
    colors = [cmap(i) for i in np.linspace(0.3, 0.9, len(accuracies))]  # Start at 0.3 to avoid too light colors
    
    bars = plt.bar(range(len(accuracies)), accuracies, color=colors)
    
    # Enhance the plot with better styling and full model name
    plt.title(f'Per-Class Accuracies\n{model_name}', fontsize=16, pad=20)
    plt.xlabel('Class', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(0, 1)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', rotation=0)
    
    # Set x-axis labels to original class numbers
    plt.xticks(range(len(ORIGINAL_CLASSES)), ORIGINAL_CLASSES, rotation=45)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(Path(save_dir) / 'class_accuracies.png', bbox_inches='tight', dpi=300)
    plt.close()


def save_metrics(confusion_mat, class_accuracies, save_dir):
    save_dir = Path(save_dir)
    
    # Calculate metrics
    total = confusion_mat.sum(axis=1)
    correct = np.diag(confusion_mat)
    accuracies = correct / total
    
    # Save detailed report
    with open(save_dir / 'evaluation_report.txt', 'w') as f:
        f.write('=== Evaluation Report ===\n\n')
        
        f.write('Per-Class Metrics:\n')
        f.write('-' * 50 + '\n')
        for idx, class_idx in enumerate(ORIGINAL_CLASSES):
            f.write(f'\nClass {class_idx}:\n')
            f.write(f'Accuracy: {accuracies[idx]:.4f}\n')
            f.write(f'Total pixels: {total[idx]}\n')
            f.write(f'Correctly classified: {correct[idx]}\n')
            
            mean_acc = np.mean(class_accuracies[idx]) if class_accuracies[idx] else 0
            f.write(f'Mean per-image accuracy: {mean_acc:.4f}\n')
        
        f.write('\nOverall Accuracy: {:.4f}\n'.format(correct.sum() / total.sum()))
        
        # Also save confusion matrix as percentages
        f.write('\nConfusion Matrix (percentages):\n')
        for i in range(confusion_mat.shape[0]):
            row_sum = confusion_mat[i].sum()
            if row_sum > 0:
                row_percentages = [f'{(val/row_sum):.2f}' for val in confusion_mat[i]]
            else:
                row_percentages = ['0.00' for _ in range(confusion_mat.shape[1])]
            f.write(f'Class {ORIGINAL_CLASSES[i]}: {", ".join(row_percentages)}\n')


def detect_model_type(state_dict):
    """Detect if model uses bilinear upsampling based on state dict keys"""
    has_transpose = any('up.weight' in key for key in state_dict.keys())
    return not has_transpose  # If no transpose conv weights found, it's bilinear

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate U-Net model performance')
    parser.add_argument('--model', '-m', required=True, help='Path to model checkpoint')
    parser.add_argument('--input', '-i', required=True, help='Path to validation images')
    parser.add_argument('--masks', '-ma', required=True, help='Path to ground truth masks')
    parser.add_argument('--output', '-o', default='evaluation_results', help='Output directory for results')
    parser.add_argument('--classes', '-c', type=int, default=11, help='Number of classes')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Scale factor for images')
    parser.add_argument('--bilinear', action='store_true', default=None,
                       help='Use bilinear upsampling (if not specified, will auto-detect)')
    return parser.parse_args()

def get_last_checkpoint(model_dir):
    """Find the last checkpoint in the given directory"""
    checkpoints = list(model_dir.glob('checkpoint_epoch*.pth'))
    if not checkpoints:
        return None
    # Sort by epoch number and get the last one
    return max(checkpoints, key=lambda x: int(x.stem.split('_epoch')[-1]))


def get_dataset_path(model_dir_name):
    """Determine dataset path from model directory name"""
    # Extract dataset identifier from the start of the folder name
    if model_dir_name.startswith('ASA'):
        return './data/Dataset A SA'
    elif model_dir_name.startswith('BSA'):
        return './data/Dataset B SA'
    elif model_dir_name.startswith('CSA'):
        return './data/Dataset CSAR'
    elif model_dir_name.startswith('DSAR'):
        return './data/Dataset DSAR'
    elif model_dir_name.startswith('A'):
        return './data/Dataset A'
    elif model_dir_name.startswith('B'):
        return './data/Dataset B'
    elif model_dir_name.startswith('C'):
        return './data/Dataset CR'
    elif model_dir_name.startswith('DSA'):
        return './data/Dataset DSA'
    else:
        raise ValueError(f"Cannot determine dataset path for model: {model_dir_name}")

def is_evaluation_complete(results_dir):
    """Check if evaluation results are already present and complete"""
    required_files = [
        'confusion_matrix.png',
        'class_accuracies.png',
        'evaluation_report.txt'
    ]
    return (
        results_dir.exists() and 
        all((results_dir / file).exists() for file in required_files)
    )

def was_recently_modified(folder_path, hours=3):
    """Check if any files inside a folder were modified within the specified hours"""
    if not folder_path.exists():
        return False, None, None
    
    # Get current time and calculate the time threshold (3 hours ago)
    current_time = time.time()
    time_threshold = current_time - (hours * 60 * 60)
    
    # Required result files
    required_files = [
        folder_path / 'confusion_matrix.png',
        folder_path / 'class_accuracies.png',
        folder_path / 'evaluation_report.txt'
    ]
    
    # Check if any of the files were modified recently
    for file_path in required_files:
        if file_path.exists():
            mod_time = file_path.stat().st_mtime
            if mod_time > time_threshold:
                # Convert timestamps to readable format for logging
                mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                threshold_str = datetime.datetime.fromtimestamp(time_threshold).strftime('%Y-%m-%d %H:%M:%S')
                return True, mod_time_str, threshold_str
    
    return False, None, None

def process_all_checkpoints():
    """Process all checkpoint directories"""
    checkpoints_dir = Path('./checkpoints')
    
    for model_dir in checkpoints_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        # Create results directory inside the checkpoint directory
        results_dir = model_dir / 'results'
        
        # Check if any result files were recently modified (within last 3 hours)
        recently_modified, mod_time, threshold = was_recently_modified(results_dir, hours=3)
        if recently_modified:
            logging.info(f"Skipping {model_dir.name} - result files were recently modified at {mod_time} (threshold: {threshold})")
            continue
            
        # Note: We're no longer checking if evaluation is complete
        # This will overwrite existing results
            
        try:
            # Get the correct dataset path based on model directory name
            dataset_base = get_dataset_path(model_dir.name)
            input_dir = Path(dataset_base) / 'imgs' / 'val'
            masks_dir = Path(dataset_base) / 'masks' / 'val'
            
            if not input_dir.exists() or not masks_dir.exists():
                logging.error(f"Missing validation data for {model_dir.name} in {dataset_base}")
                continue
                
            # Find the last checkpoint in this directory
            last_checkpoint = get_last_checkpoint(model_dir)
            if not last_checkpoint:
                logging.warning(f"No checkpoints found in {model_dir}")
                continue
                
            # Create results directory inside the checkpoint directory
            results_dir = model_dir / 'results'
            results_dir.mkdir(exist_ok=True)
            
            logging.info(f"\nProcessing {model_dir.name}")
            logging.info(f"Using checkpoint: {last_checkpoint.name}")
            logging.info(f"Using validation data from: {dataset_base}")
            
            # Set up device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            try:
                # Load checkpoint
                state_dict = torch.load(str(last_checkpoint), map_location=device)
                if 'mask_values' in state_dict:
                    state_dict.pop('mask_values')
                
                # Auto-detect model type
                bilinear = detect_model_type(state_dict)
                
                # Initialize and load model
                net = UNet(n_channels=3, n_classes=11, bilinear=bilinear)
                net.to(device)
                net.load_state_dict(state_dict)
                
                # Create dataset and dataloader
                val_dataset = BasicDataset(input_dir, masks_dir, scale=0.5)
                val_loader = DataLoader(val_dataset, 
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True)
                
                # Evaluate
                confusion_mat, class_accuracies = evaluate_model(net, val_loader, device, 11)
                
                # Save results in the model's directory
                plot_results(confusion_mat, class_accuracies, results_dir)
                save_metrics(confusion_mat, class_accuracies, results_dir)
                
                logging.info(f"Results saved in {results_dir}")
                
            except Exception as e:
                logging.error(f"Error processing {model_dir.name}: {str(e)}")
                continue
            
        except Exception as e:
            logging.error(f"Error processing {model_dir.name}: {str(e)}")
            continue

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if len(sys.argv) > 1:
        # Original command-line behavior
        args = get_args()
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        
        # Load checkpoint first to detect model type
        state_dict = torch.load(args.model, map_location=device)
        if 'mask_values' in state_dict:
            state_dict.pop('mask_values')

        # Auto-detect model type if not specified
        if args.bilinear is None:
            args.bilinear = detect_model_type(state_dict)
        logging.info(f'Using {"bilinear" if args.bilinear else "transposed conv"} upsampling')

        # Initialize model with correct upsampling type
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        net.to(device)

        try:
            net.load_state_dict(state_dict)
        except RuntimeError as e:
            logging.error("Failed to load state dict. If error persists, manually specify --bilinear flag")
            raise
        
        # Create dataset and dataloader
        val_dataset = BasicDataset(args.input, args.masks, args.scale)
        val_loader = DataLoader(val_dataset, 
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True)
        
        # Evaluate
        logging.info('Starting evaluation...')
        confusion_mat, class_accuracies = evaluate_model(net, val_loader, device, args.classes)
        
        # Save and plot results
        logging.info('Saving results...')
        plot_results(confusion_mat, class_accuracies, args.output)
        save_metrics(confusion_mat, class_accuracies, args.output)
        
        logging.info(f'Evaluation complete! Results saved in {args.output}')
    else:
        logging.info("Starting automatic evaluation of all checkpoints")
        process_all_checkpoints()

'''
to run copy paste 'python evaluate_model.py --model checkpoints/your_model.pth --input path/to/val/images --masks path/to/val/masks --classes 11'
ex: python evaluate_model.py --model checkpoints/original_single_annotation/default_originaldata/checkpoint_epoch5.pth --input data/imgs/val --masks data/masks/val --classes 11 --output checkpoints/original_single_annotation/default_originaldata
'''
