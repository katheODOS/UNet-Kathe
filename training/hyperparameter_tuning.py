import os
import sys
import logging
from pathlib import Path
from itertools import product
import torch
import training.train as train
from training.train import train_model, UNet
import wandb
import io
from contextlib import redirect_stdout
from tqdm import tqdm
import traceback 
from io import StringIO
import atexit
import re

# Hyperparameter configurations
LEARNING_RATES = [1e-7, 1e-6]
BATCH_SIZES = [1, 2, 4]
EPOCHS = [10, 15]
WEIGHT_DECAYS = [1e-9, 1e-8, 1e-7]

# Dataset configurations with path mappings
DATASETS = {
    #'A': {'name': 'Dataset A', 'code': 'A', 'path': 'Dataset A'},
    #'B': {'name': 'Dataset B', 'code': 'B', 'path': 'Dataset B'},
    'ASA': {'name': 'Dataset A SA', 'code': 'ASA', 'path': 'Dataset A SA'},
    'BSA': {'name': 'Dataset B SA', 'code': 'BSA', 'path': 'Dataset B SA'}
}

def setup_checkpoint_dir(dataset_code, lr, wd, epochs, batch_size):
    """Create and return checkpoint directory for specific configuration"""
    dir_name = f"{dataset_code}L{lr:.0e}W{wd:.0e}B{batch_size}E{epochs}"
    checkpoint_dir = Path('./checkpoints') / dir_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir

def save_run_output(output_text, checkpoint_dir):
    """Save the run output to output.txt"""
    with open(checkpoint_dir / 'output.txt', 'w') as f:
        f.write(output_text)

class SafeOutputCapture:
    def __init__(self):
        self.buffer = StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        # Add logging capture
        self.log_handler = logging.StreamHandler(self.buffer)
        self.log_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
    def write(self, text):
        self.buffer.write(text)
        self.original_stdout.write(text)
        
    def flush(self):
        self.buffer.flush()
        self.original_stdout.flush()
        
    def get_output(self):
        return self.buffer.getvalue()
    
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        # Add logging handler when entering context
        logging.getLogger().addHandler(self.log_handler)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        # Remove logging handler when exiting context
        logging.getLogger().removeHandler(self.log_handler)
        self.buffer.close()

def cleanup_wandb():
    """Ensure wandb is properly cleaned up"""
    try:
        if wandb.run is not None:
            wandb.finish()
    except:
        pass

def extract_latest_validation_score(output_text):
    """Extract the most recent validation score from the output"""
    matches = re.findall(r'INFO: Validation Dice score: (\d+\.\d+)', output_text)
    return float(matches[-1]) if matches else 0.0

def run_training_configuration(dataset_path, checkpoint_dir, lr, batch_size, epochs, weight_decay, config_details):
    """Run training with specific configuration and capture output"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(n_channels=3, n_classes=11, bilinear=True)
    model = model.to(device=device)
    
    with SafeOutputCapture() as output:
        try:
            # Write configuration details first
            sys.stdout.write("="*80 + "\n")
            sys.stdout.write(f"Configuration Details:\n")
            sys.stdout.write("="*80 + "\n")
            sys.stdout.write(config_details + "\n")
            sys.stdout.write("="*80 + "\n\n")
            
            atexit.register(cleanup_wandb)
            
            original_img_dir = train.dir_img
            original_mask_dir = train.dir_mask
            original_checkpoint_dir = train.dir_checkpoint
            
            train.dir_img = Path(dataset_path) / 'imgs'
            train.dir_mask = Path(dataset_path) / 'masks'
            train.dir_checkpoint = checkpoint_dir
            
            # Do full training at once, we'll monitor the output for early stopping
            train_model(
                model=model,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                device=device,
                img_scale=0.5,
                save_checkpoint=True,
                weight_decay=weight_decay
            )
            
            # Check if we should have stopped early
            current_output = output.get_output()
            validation_scores = re.findall(r'INFO: Validation Dice score: (\d+\.\d+)', current_output)
            
            # Add early stopping note if applicable
            if len(validation_scores) >= 5:  # We have at least 5 scores
                early_scores = [float(score) for score in validation_scores[:5]]
                if max(early_scores) < 0.65:
                    sys.stdout.write("\nNote: Early stopping would have occurred after epoch 5 (score below 0.65)\n")
            
            # Restore directory variables
            train.dir_img = original_img_dir
            train.dir_mask = original_mask_dir
            train.dir_checkpoint = original_checkpoint_dir
            
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            print("\nFull traceback:")
            print(traceback.format_exc())
        finally:
            cleanup_wandb()
            atexit.unregister(cleanup_wandb)
            
        return output.get_output()

def is_training_completed(checkpoint_dir, epochs):
    """Check if training was already completed for this configuration"""
    final_checkpoint = checkpoint_dir / f'checkpoint_epoch{epochs}.pth'
    return final_checkpoint.exists()

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Create all possible combinations of hyperparameters
    configs = list(product(
        DATASETS.items(),
        LEARNING_RATES,
        BATCH_SIZES,
        EPOCHS,
        WEIGHT_DECAYS
    ))
    total_combinations = len(configs)
    logging.info(f"Total number of combinations to try: {total_combinations}")
    
    try:
        # Register global cleanup
        atexit.register(cleanup_wandb)
        
        for idx, ((dataset_key, dataset_info), lr, batch_size, epochs, weight_decay) in enumerate(configs, 1):
            logging.info(f"\n{'='*80}")
            logging.info(f"Running combination {idx}/{total_combinations}")
            logging.info(f"{'='*80}")
            
            # Setup directories for current configuration
            checkpoint_dir = setup_checkpoint_dir(
                dataset_info['code'], lr, weight_decay, epochs, batch_size
            )
            
            # Check if this combination was already completed
            if is_training_completed(checkpoint_dir, epochs):
                logging.info(f"Training already completed for this configuration. Skipping...")
                continue
            
            logging.info(f"""
            Configuration details:
            Dataset: {dataset_info['name']} ({dataset_key})
            Learning Rate: {lr}
            Batch Size: {batch_size}
            Epochs: {epochs}
            Weight Decay: {weight_decay}
            Checkpoint Directory: {checkpoint_dir}
            """)
            
            # Update data directories for current dataset using the path field
            dataset_base = f'./data/{dataset_info["path"]}'
            
            if not Path(dataset_base).exists():
                logging.error(f"Dataset directory {dataset_base} not found! Skipping this combination.")
                continue
                
            logging.info("Starting training for this combination...")
            
            # Format configuration details
            config_details = f"""Dataset: {dataset_info['name']} ({dataset_key})
Learning Rate: {lr}
Batch Size: {batch_size}
Epochs: {epochs}
Weight Decay: {weight_decay}
Checkpoint Directory: {checkpoint_dir}"""
            
            # Run training with config details
            try:
                output = run_training_configuration(
                    dataset_base,
                    checkpoint_dir,
                    lr,
                    batch_size,
                    epochs,
                    weight_decay,
                    config_details
                )
                
                # Save output
                save_run_output(output, checkpoint_dir)
                logging.info(f"Training completed and saved to {checkpoint_dir}")
                
            except Exception as e:
                logging.error(f"Error during training: {str(e)}")
                cleanup_wandb()  # Ensure wandb is cleaned up after error
                continue
            
            # Clean up (removed wandb.finish() since it's handled by cleanup_wandb)
            torch.cuda.empty_cache()
            
            logging.info(f"Completed combination {idx}/{total_combinations}")
    finally:
        # Final cleanup
        cleanup_wandb()
        # Deregister cleanup function
        atexit.unregister(cleanup_wandb)

if __name__ == '__main__':
    main()
