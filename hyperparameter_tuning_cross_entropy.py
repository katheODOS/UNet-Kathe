import os
import sys
import logging
from pathlib import Path
from itertools import product
import torch
import train_cross_entropy
from train_cross_entropy import train_model, UNet
import wandb
import io
from contextlib import redirect_stdout

# Hyperparameter configurations
LEARNING_RATES = [1e-7, 1e-6, 1e-5, 1e-4]
BATCH_SIZES = [2, 4, 8]
EPOCHS = [5, 10, 15]
WEIGHT_DECAYS = [1e-9, 1e-8, 1e-7]

# Dataset configurations
DATASETS = {
    'A': {'name': 'Dataset A', 'code': 'A'},
    'B': {'name': 'Dataset B', 'code': 'B'},
    'ASA': {'name': 'Dataset A SA', 'code': 'ASA'},
    'BSA': {'name': 'Dataset B SA', 'code': 'BSA'}
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

def run_training_configuration(dataset_path, checkpoint_dir, lr, batch_size, epochs, weight_decay):
    """Run training with specific configuration and capture output"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = UNet(n_channels=3, n_classes=11)
    model = model.to(device=device, memory_format=torch.channels_last)
    
    # Temporarily modify the global checkpoint directory
    original_checkpoint_dir = train_cross_entropy.dir_checkpoint
    train_cross_entropy.dir_checkpoint = checkpoint_dir
    
    # Capture output
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        try:
            train_model(
                model=model,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                device=device,
                img_scale=0.5,
                weight_decay=weight_decay,
                save_checkpoint=True
            )
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
    
    # Restore original checkpoint directory
    train_cross_entropy.dir_checkpoint = original_checkpoint_dir
    
    return output_buffer.getvalue()

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Create all possible combinations of hyperparameters
    configs = product(
        DATASETS.items(),
        LEARNING_RATES,
        BATCH_SIZES,
        EPOCHS,
        WEIGHT_DECAYS
    )
    
    for (dataset_key, dataset_info), lr, batch_size, epochs, weight_decay in configs:
        # Setup directories for current configuration
        checkpoint_dir = setup_checkpoint_dir(
            dataset_info['code'], lr, weight_decay, epochs
        )
        
        logging.info(f"""
        Starting training with configuration:
        Dataset: {dataset_info['name']}
        Learning Rate: {lr}
        Batch Size: {batch_size}
        Epochs: {epochs}
        Weight Decay: {weight_decay}
        Checkpoint Directory: {checkpoint_dir}
        """)
        
        # Update data directories for current dataset
        dataset_base = f'./data/{dataset_key}'
        dir_img = Path(f'{dataset_base}/imgs')
        dir_mask = Path(f'{dataset_base}/masks')
        
        # Run training
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
        
        # Clean up
        wandb.finish()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
