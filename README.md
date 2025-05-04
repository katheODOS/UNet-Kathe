# UNet-Kathe

## Overview
Hello! Thank you for visiting my Bachelor's capstone project :3 please find information that can help you navigate these files and the setup below.


> **Note:** UNet-Kathe is built upon [Alexandre Milesi's PyTorch implementation of UNet](https://github.com/milesial/Pytorch-UNet). This project extends his work with additional functionalities to test the feasibility of using biodiversity datasets in production environments.

## Setup Instructions

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or Miniconda
- CUDA-compatible GPU (recommended)

### Environment Setup
1. Create and activate a conda environment:
   ```bash
   conda create --name UNet-Kathe python=3.8.20
   conda activate UNet-Kathe
   ```

2. Clone the repository:
   ```bash
   cd C:\Users\YourUser\anaconda3\envs\UNet-Kathe 
   git clone https://github.com/yourusername/UNet-Kathe.git
   cd UNet-Kathe
   ```

3. Install PyTorch with CUDA support:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   pip install seaborn scikit-learn imgaug
   ```

## Project Structure

- **preprocessing_scripts/** - Tools for data preparation and augmentation
- **model_evaluation/** - Scripts for evaluating model performance
- **checkpoints/** - Model checkpoints and evaluation results
- **utils/** - Utility functions for data loading and processing

Each major directory contains its own README.md with detailed information about the workflow process.

## Usage

See the specific README files in each directory for detailed instructions on:
- Data preprocessing
- Model training
- Evaluation procedures
- Result analysis


## Acknowledgments

UNet-Kathe is built upon Alexandre Milesi's Pytorch implementation of the UNet (https://github.com/milesial/Pytorch-UNet). This project would not have been possible without his repository.
