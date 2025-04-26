import sys
import os
import torch
import logging
from pathlib import Path

# Add parent directory to path - this is more reliable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Try importing after fixing path
try:
    from utils.data_loading import BasicDataset
    from unet import UNet
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
# Batch predict all the imaages associated to visualize them. if images are in single channel annotation convert them back to RGB

def get_dataset_path(model_dir_name):
    """Determine dataset path from model directory name"""
    if model_dir_name.startswith('DSA'):
        return './data/Dataset DSAR'
    elif model_dir_name.startswith('ASA'):
        return './data/Dataset A SA'
    elif model_dir_name.startswith('BSA'):
        return './data/Dataset B SA'
    elif model_dir_name.startswith('CSA'):
        return './data/Dataset CSAR'
    elif model_dir_name.startswith('A'):
        return './data/Dataset A'
    elif model_dir_name.startswith('B'):
        return './data/Dataset B'
    elif model_dir_name.startswith('C'):
        return './data/Dataset C'
    else:
        raise ValueError(f"Cannot determine dataset path for model: {model_dir_name}")

def get_last_checkpoint(model_dir):
    """Find the last checkpoint in the given directory"""
    checkpoints = list(model_dir.glob('checkpoint_epoch*.pth'))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda x: int(x.stem.split('_epoch')[-1]))

def predict_img(net, img, device):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(img, scale=0.5, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).softmax(dim=1)
        mask = output.argmax(dim=1)
        return mask[0].cpu().numpy()

def process_checkpoints():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    checkpoints_dir = Path('./checkpoints')
    
    for model_dir in checkpoints_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        results_dir = model_dir / 'results'
        if not results_dir.exists():
            continue
            
        try:
            pred_dir = results_dir / 'predictions'
            pred_dir.mkdir(exist_ok=True)
            
            dataset_base = Path(get_dataset_path(model_dir.name))
            val_imgs_dir = dataset_base / 'imgs' / 'val'
            
            if not val_imgs_dir.exists():
                logging.error(f"Validation directory not found: {val_imgs_dir}")
                continue
                
            checkpoint_file = get_last_checkpoint(model_dir)
            if not checkpoint_file:
                logging.warning(f"No checkpoints found in {model_dir}")
                continue
                
            logging.info(f"\nProcessing {model_dir.name}")
            logging.info(f"Using checkpoint: {checkpoint_file.name}")
            logging.info(f"Loading validation images from: {val_imgs_dir}")
            
            state_dict = torch.load(checkpoint_file, map_location=device)
            if 'mask_values' in state_dict:
                state_dict.pop('mask_values')
                
            net = UNet(n_channels=3, n_classes=11, bilinear=True)
            net.to(device=device)
            net.load_state_dict(state_dict)
            
            for img_path in tqdm(list(val_imgs_dir.glob('*')), desc='Generating predictions'):
                img = Image.open(img_path)
                
                mask = predict_img(net, img, device)
                
                pred_filename = pred_dir / f"{img_path.stem}_pred.png"
                mask_img = Image.fromarray(mask.astype(np.uint8))
                mask_img.save(pred_filename)
            
            logging.info(f"Predictions saved to {pred_dir}")
            
        except Exception as e:
            logging.error(f"Error processing {model_dir.name}: {str(e)}")
            continue

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    process_checkpoints()
