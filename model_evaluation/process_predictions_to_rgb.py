import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging

# Define the mapping from class ID to RGB values
CLASS_TO_RGB = {
    0: (11, 246, 210),   # ignore index
    1: (39, 179, 65),    # pasture class
    2: (230, 87, 196),   # woodland class
    3: (252, 126, 187),  # conifer class
    4: (255, 207, 74),   # shrub
    5: (250, 62, 119),   # hedgerow
    6: (250, 148, 65),   # seminatural grassland
    7: (173, 173, 173),  # artificial surface
    9: (255, 193, 122),  # bare field
    12: (168, 232, 84),  # arable
    13: (217, 217, 217), # artificial garden
}

def is_single_channel(image_path):
    """Check if image is single channel"""
    with Image.open(image_path) as img:
        return len(img.getbands()) == 1

def convert_single_to_rgb(image_path, output_path):
    """Convert a single-channel image to RGB using the class mapping"""
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Create RGB array
    height, width = img_array.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Convert each pixel value to its RGB equivalent
    for class_id, rgb_value in CLASS_TO_RGB.items():
        mask = img_array == class_id
        rgb_array[mask] = rgb_value
    
    # Convert to PIL Image and save
    rgb_img = Image.fromarray(rgb_array)
    rgb_img.save(output_path)

def process_checkpoint_predictions():
    """Process predictions in all checkpoint directories"""
    checkpoints_dir = Path('./checkpoints')
    
    # Find all checkpoint directories
    for model_dir in checkpoints_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        results_dir = model_dir / 'results'
        pred_dir = results_dir / 'predictions'
        
        # Check if predictions directory exists
        if not pred_dir.exists():
            continue
            
        # Check if there are any prediction images
        pred_images = list(pred_dir.glob('*.png'))
        if not pred_images:
            continue
            
        # Check if the predictions are single channel
        if not is_single_channel(pred_images[0]):
            logging.info(f"Predictions in {model_dir.name} are already RGB. Skipping...")
            continue
            
        # Create RGB predictions directory
        rgb_pred_dir = results_dir / 'rgb_predictions'
        rgb_pred_dir.mkdir(exist_ok=True)
        
        logging.info(f"\nProcessing {model_dir.name}")
        logging.info(f"Converting {len(pred_images)} predictions to RGB")
        
        # Convert each prediction
        for img_path in tqdm(pred_images, desc="Converting predictions"):
            output_path = rgb_pred_dir / img_path.name
            try:
                convert_single_to_rgb(img_path, output_path)
            except Exception as e:
                logging.error(f"Error processing {img_path.name}: {str(e)}")
                continue
        
        logging.info(f"RGB predictions saved to {rgb_pred_dir}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    process_checkpoint_predictions()
