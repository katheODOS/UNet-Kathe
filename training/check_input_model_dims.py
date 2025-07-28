"""
This script helps diagnose model initialization issues by checking
the dimensions of model layers and ensuring classes are properly set.
"""

import sys
import os
import torch
from pathlib import Path

# Add parent directory to path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unet import UNet

def print_model_dims(model):
    """Print dimensions of each layer in the model"""
    print(f"Model class count: {model.n_classes}")
    print(f"Model channel count: {model.n_channels}")
    print(f"Using bilinear upsampling: {model.bilinear}")
    
    # Print output layer details
    outc = model.outc
    conv = outc.conv
    print(f"\nOutput conv layer:")
    print(f"  Weight shape: {conv.weight.shape}")
    print(f"  Bias shape: {conv.bias.shape}")
    
    # Check the first few conv layers
    print("\nFirst few layers:")
    print(f"  inc.double_conv.0: {model.inc.double_conv[0].weight.shape}")
    print(f"  down1.maxpool_conv.1.double_conv.0: {model.down1.maxpool_conv[1].double_conv[0].weight.shape}")
    
    # Check the up layers
    print("\nUp layers:")
    for i, up in enumerate([model.up1, model.up2, model.up3, model.up4]):
        if model.bilinear:
            print(f"  up{i+1}.conv.double_conv.0: {up.conv.double_conv[0].weight.shape}")
        else:
            print(f"  up{i+1}.up: {up.up.weight.shape}")

def check_state_dict(state_dict_path):
    """Examine a saved state dict to determine its dimensions"""
    print(f"Examining state dict: {state_dict_path}")
    
    # Use weights_only=True to avoid the FutureWarning
    state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
    
    # Check the output layer dimensions
    if 'outc.conv.weight' in state_dict:
        weight = state_dict['outc.conv.weight']
        bias = state_dict['outc.conv.bias']
        print(f"Output layer dimensions:")
        print(f"  Weight shape: {weight.shape}")
        print(f"  Bias shape: {bias.shape}")
        print(f"  Number of classes inferred: {bias.shape[0]}")
    else:
        print("Could not find outc.conv.weight in state dict")
    
    return state_dict

def main():
    # Check a model with 8 classes
    model_8 = UNet(n_channels=3, n_classes=8, bilinear=False)
    print("=== Model with 8 classes ===")
    print_model_dims(model_8)
    
    # Check a model with 11 classes
    model_11 = UNet(n_channels=3, n_classes=11, bilinear=False)
    print("\n=== Model with 11 classes ===")
    print_model_dims(model_11)
    
    # If a checkpoint path is provided, examine it
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        check_state_dict(checkpoint_path)

if __name__ == "__main__":
    main()
