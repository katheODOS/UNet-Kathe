import os
import sys
import logging
from pathlib import Path
from itertools import product
import torch
import train
from train import train_model, UNet
import wandb
import io
from contextlib import redirect_stdout
from tqdm import tqdm
import traceback  # Add this import
from io import StringIO
import atexit

# Hyperparameter configurations
LEARNING_RATES = [1e-7, 1e-6, 1e-5, 1e-4]
BATCH_SIZES = [2, 4, 8]
EPOCHS = [5, 10, 15]
WEIGHT_DECAYS = [1e-9, 1e-8, 1e-7]

# Dataset configurations with path mappings
DATASETS = {
    'A': {'name': 'Dataset A', 'code': 'A', 'path': 'Dataset A'},
    'B': {'name': 'Dataset B', 'code': 'B', 'path': 'Dataset B'},
    'ASA': {'name': 'Dataset A SA', 'code': 'ASA', 'path': 'Dataset A SA'},
    'BSA': {'name': 'Dataset B SA', 'code': 'BSA', 'path': 'Dataset B SA'}
}

def setup_checkpoint_dir(dataset_code, lr, wd, epochs):
    """Create and return checkpoint directory for specific configuration"""
    dir_name = f"{dataset_code}L{lr:.0e}W{wd:.0e}E{epochs}"
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
        
    def write(self, text):
        # Write to buffer and original stdout without recursion
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
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.buffer.close()

def cleanup_wandb():
    """Ensure wandb is properly cleaned up"""
    try:
        if wandb.run is not None:
            wandb.finish()
    except:
        pass

def run_training_configuration(dataset_path, checkpoint_dir, lr, batch_size, epochs, weight_decay):
    """Run training with specific configuration and capture output"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with correct number of classes (11 for your case: 10 classes + background)
    model = UNet(n_channels=3, n_classes=11, bilinear=True)
    model = model.to(device=device)
    
    # Set up safer output capture
    with SafeOutputCapture() as output:
        try:
            # Register cleanup function
            atexit.register(cleanup_wandb)
            
            # Temporarily modify the global directory variables
            original_img_dir = train.dir_img
            original_mask_dir = train.dir_mask
            original_checkpoint_dir = train.dir_checkpoint
            
            train.dir_img = Path(dataset_path) / 'imgs'
            train.dir_mask = Path(dataset_path) / 'masks'
            train.dir_checkpoint = checkpoint_dir
            
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
            
            # Capture wandb summary before cleanup
            if wandb.run is not None:
                print("\nWandb Run Summary:")
                print("=" * 50)
                for key, value in wandb.run.summary.items():
                    print(f"{key}: {value}")
                print("=" * 50)
            
            # Restore directory variables
            train.dir_img = original_img_dir
            train.dir_mask = original_mask_dir
            train.dir_checkpoint = original_checkpoint_dir
            
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            print("\nFull traceback:")
            print(traceback.format_exc())
        finally:
            # Cleanup wandb run
            cleanup_wandb()
            # Deregister cleanup function
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
                dataset_info['code'], lr, weight_decay, epochs
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
            
            # Run training
            try:
                output = run_training_configuration(
                    dataset_base,
                    checkpoint_dir,
                    lr,
                    batch_size,
                    epochs,
                    weight_decay
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
