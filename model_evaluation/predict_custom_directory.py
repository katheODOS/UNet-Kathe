import sys
import os
import torch
import logging
from pathlib import Path
import argparse

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
from tqdm import tqdm

def predict_img(net, img, device):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(img, scale=1.0, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).softmax(dim=1)
        mask = output.argmax(dim=1)
        return mask[0].cpu().numpy()

def predict_directory(input_dir, output_dir, checkpoint_path):
    """
    Generate predictions for all images in the input directory using the specified model checkpoint
    and save results to the output directory.
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load the checkpoint
    try:
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # Remove mask_values if present (not needed for inference)
        if 'mask_values' in state_dict:
            state_dict.pop('mask_values')
        
        # Determine number of classes from the output layer weights
        n_classes = state_dict['outc.conv.weight'].size(0)
        logging.info(f"Model has {n_classes} output classes")
        
        # Create model with the correct number of classes
        net = UNet(n_channels=3, n_classes=n_classes, bilinear=True)
        net.to(device=device)
        net.load_state_dict(state_dict)
        
        # Process all images in the input directory
        input_path = Path(input_dir)
        image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')) + list(input_path.glob('*.tif'))
        
        if not image_files:
            logging.warning(f"No image files found in {input_dir}")
            return
        
        logging.info(f"Found {len(image_files)} images to process")
        
        for img_path in tqdm(image_files, desc='Generating predictions'):
            try:
                img = Image.open(img_path)
                
                mask = predict_img(net, img, device)
                
                pred_filename = output_path / f"{img_path.stem}.png"
                mask_img = Image.fromarray(mask.astype(np.uint8))
                mask_img.save(pred_filename)
            except Exception as e:
                logging.error(f"Error processing {img_path.name}: {str(e)}")
                continue
        
        logging.info(f"Predictions saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error loading model or processing images: {str(e)}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description='Predict on images using a trained UNet model')
    parser.add_argument('--input', '-i', type=str, default=r"C:\Users\Admin\Desktop\test_images\UK\processing\512",
                        help='Directory containing input images')
    parser.add_argument('--output', '-o', type=str, default=r"C:\Users\Admin\Desktop\test_images\UK\predictions\png\single_channel",
                        help='Directory to save prediction outputs')
    parser.add_argument('--checkpoint', '-c', type=str, 
                        default=r"C:\Users\Admin\anaconda3\envs\UNet-Kathe\UNet-Kathe\checkpoints\BSTL1e-08W1e-06B4E60_1.0\checkpoint_epoch60.pth",
                        help='Path to the checkpoint file')
    
    args = parser.parse_args()
    
    logging.info(f"Input directory: {args.input}")
    logging.info(f"Output directory: {args.output}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    
    predict_directory(args.input, args.output, args.checkpoint)
