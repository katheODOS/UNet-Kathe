import os
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def verify_dataset_structure(base_path):
    """
    Verifies and reports on the dataset directory structure.
    """
    base_path = Path(base_path)
    logging.info(f"Verifying dataset structure at: {base_path}")
    
    # Check main directories
    imgs_path = base_path / 'imgs'
    masks_path = base_path / 'masks'
    
    if not imgs_path.exists():
        logging.error(f"Images directory not found: {imgs_path}")
        return False
    
    if not masks_path.exists():
        logging.error(f"Masks directory not found: {masks_path}")
        return False
    
    # Check subdirectories
    img_train_path = imgs_path / 'train'
    img_val_path = imgs_path / 'val'
    mask_train_path = masks_path / 'train'
    mask_val_path = masks_path / 'val'
    
    has_train_val = all(p.exists() for p in [img_train_path, img_val_path, mask_train_path, mask_val_path])
    
    if has_train_val:
        logging.info("Dataset has train/val subdirectory structure.")
        train_imgs = list(img_train_path.glob('*'))
        val_imgs = list(img_val_path.glob('*'))
        train_masks = list(mask_train_path.glob('*'))
        val_masks = list(mask_val_path.glob('*'))
        
        logging.info(f"Train images: {len(train_imgs)}")
        logging.info(f"Val images: {len(val_imgs)}")
        logging.info(f"Train masks: {len(train_masks)}")
        logging.info(f"Val masks: {len(val_masks)}")
        
        if len(train_imgs) == 0 or len(val_imgs) == 0:
            logging.error("Found empty image directories!")
            return False
            
        if len(train_masks) == 0 or len(val_masks) == 0:
            logging.error("Found empty mask directories!")
            return False
    else:
        logging.error("Missing train/val subdirectory structure!")
        return False
    
    logging.info("Dataset structure verified successfully.")
    return True

def sample_dataset_statistics(mask_dir, num_samples=10):
    """
    Samples random mask files and reports their statistics.
    """
    mask_dir = Path(mask_dir)
    mask_files = list(mask_dir.glob('*'))
    
    if len(mask_files) == 0:
        logging.error(f"No mask files found in {mask_dir}")
        return
    
    if num_samples > len(mask_files):
        num_samples = len(mask_files)
    
    # Sample random mask files
    import random
    sample_files = random.sample(mask_files, num_samples)
    
    logging.info(f"Analyzing {num_samples} sample masks from {mask_dir}")
    
    from PIL import Image
    import numpy as np
    
    all_values = set()
    
    for mask_file in sample_files:
        try:
            mask = np.array(Image.open(mask_file))
            unique_values = np.unique(mask)
            all_values.update(unique_values)
            
            logging.info(f"Mask: {mask_file.name}, shape: {mask.shape}, unique values: {unique_values}")
        except Exception as e:
            logging.error(f"Error processing mask {mask_file}: {str(e)}")
    
    logging.info(f"All unique mask values found: {sorted(list(all_values))}")
    return sorted(list(all_values))

if __name__ == "__main__":
    # Can be run as a standalone script for diagnostics
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python utils/data_debug.py <dataset_path>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    dataset_path = sys.argv[1]
    
    if verify_dataset_structure(dataset_path):
        # If structure is valid, analyze mask statistics
        sample_dataset_statistics(Path(dataset_path) / 'masks' / 'train')
