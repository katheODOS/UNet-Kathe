import os
import sys
import logging
from pathlib import Path
import torch
import train
from train import train_model, UNet
import wandb
from io import StringIO
import atexit
import traceback

CONFIGURATIONS = [
   {
        'dataset': {'name': 'Dataset DSAR', 'code': 'DSA', 'path': 'Dataset DSAR'},
        'learning_rate': 1e-7,
        'weight_decay': 1e-9,
        'batch_size': 4,
        'epochs': 20
    },
    {
       'dataset': {'name': 'Dataset DSAR', 'code': 'DSA', 'path': 'Dataset DSAR'},
        'learning_rate': 1e-7,
        'weight_decay': 1e-8,
        'batch_size': 8,
        'epochs': 15
    },
    {
        'dataset': {'name': 'Dataset DSAR', 'code': 'DSA', 'path': 'Dataset DSAR'},
        'learning_rate': 1e-7,
        'weight_decay': 1e-9,
        'batch_size': 16,
        'epochs': 20
    }



]

# Reuse the helper classes and functions from hyperparameter_tuning.py
from hyperparameter_tuning import (
    SafeOutputCapture, cleanup_wandb, setup_checkpoint_dir,
    save_run_output, is_training_completed, run_training_configuration
)

def main():
    logging.basicConfig(level=logging.INFO)
    
    total_configs = len(CONFIGURATIONS)
    logging.info(f"Total number of configurations to try: {total_configs}")
    
    try:
        atexit.register(cleanup_wandb)
        
        for idx, config in enumerate(CONFIGURATIONS, 1):
            logging.info(f"\n{'='*80}")
            logging.info(f"Running configuration {idx}/{total_configs}")
            logging.info(f"{'='*80}")
            
            # Setup directories for current configuration
            checkpoint_dir = setup_checkpoint_dir(
                config['dataset']['code'],
                config['learning_rate'],
                config['weight_decay'],
                config['epochs'],
                config['batch_size']
            )
            
            # Check if this combination was already completed
            if is_training_completed(checkpoint_dir, config['epochs']):
                logging.info(f"Training already completed for this configuration. Skipping...")
                continue
            
            # Log configuration details
            logging.info(f"""
            Configuration details:
            Dataset: {config['dataset']['name']} ({config['dataset']['code']})
            Learning Rate: {config['learning_rate']}
            Batch Size: {config['batch_size']}
            Epochs: {config['epochs']}
            Weight Decay: {config['weight_decay']}
            Checkpoint Directory: {checkpoint_dir}
            """)
            
            # Update data directories for current dataset
            dataset_base = f'./data/{config["dataset"]["path"]}'
            
            if not Path(dataset_base).exists():
                logging.error(f"Dataset directory {dataset_base} not found! Skipping this combination.")
                continue
                
            logging.info("Starting training for this configuration...")
            
            # Format configuration details
            config_details = f"""Dataset: {config['dataset']['name']} ({config['dataset']['code']})
Learning Rate: {config['learning_rate']}
Batch Size: {config['batch_size']}
Epochs: {config['epochs']}
Weight Decay: {config['weight_decay']}
Checkpoint Directory: {checkpoint_dir}"""
            
            try:
                output = run_training_configuration(
                    dataset_base,
                    checkpoint_dir,
                    config['learning_rate'],
                    config['batch_size'],
                    config['epochs'],
                    config['weight_decay'],
                    config_details
                )
                
                save_run_output(output, checkpoint_dir)
                logging.info(f"Training completed and saved to {checkpoint_dir}")
                
            except Exception as e:
                logging.error(f"Error during training: {str(e)}")
                cleanup_wandb()
                continue
            
            torch.cuda.empty_cache()
            logging.info(f"Completed configuration {idx}/{total_configs}")
            
    finally:
        cleanup_wandb()
        atexit.unregister(cleanup_wandb)

if __name__ == '__main__':
    main()
