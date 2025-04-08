import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from utils.data_loading import BasicDataset
from unet import UNet

ORIGINAL_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 13]

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
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    
    # Set original class labels
    tick_positions = np.arange(len(ORIGINAL_CLASSES)) + 0.5
    plt.xticks(tick_positions, ORIGINAL_CLASSES, rotation=45)
    plt.yticks(tick_positions, ORIGINAL_CLASSES, rotation=0)
    
    plt.savefig(Path(save_dir) / 'confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    # Plot per-class accuracies
    plt.figure(figsize=(12, 6))
    accuracies = [np.mean(vals) if vals else 0 for vals in class_accuracies.values()]
    plt.bar(range(len(accuracies)), accuracies)
    plt.title('Per-Class Accuracies')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Set x-axis labels to original class numbers
    plt.xticks(range(len(ORIGINAL_CLASSES)), ORIGINAL_CLASSES)
    
    plt.savefig(Path(save_dir) / 'class_accuracies.png', bbox_inches='tight')
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


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate U-Net model performance')
    parser.add_argument('--model', '-m', required=True, help='Path to model checkpoint')
    parser.add_argument('--input', '-i', required=True, help='Path to validation images')
    parser.add_argument('--masks', '-ma', required=True, help='Path to ground truth masks')
    parser.add_argument('--output', '-o', default='evaluation_results', help='Output directory for results')
    parser.add_argument('--classes', '-c', type=int, default=11, help='Number of classes')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Scale factor for images')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # Load model
    net = UNet(n_channels=3, n_classes=args.classes)
    state_dict = torch.load(args.model, map_location=device)
    if 'mask_values' in state_dict:
        state_dict.pop('mask_values')
    net.load_state_dict(state_dict)
    net.to(device)
    
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

'''
to run copy paste 'python evaluate_model.py --model checkpoints/your_model.pth --input path/to/val/images --masks path/to/val/masks --classes 11'
ex: python evaluate_model.py --model checkpoints/original_single_annotation/default_originaldata/checkpoint_epoch5.pth --input data/imgs/val --masks data/masks/val --classes 11 --output checkpoints/original_single_annotation/default_originaldata
'''

