#!/bin/bash

# scDiffusion Classifier Environment Setup Script
# This script creates a conda environment specifically for running classifier_train.py

set -e  # Exit on any error

echo "=== scDiffusion Classifier Environment Setup ==="
echo "This script will create a conda environment named 'scdiffusion_classifier'"
echo "This environment is separate from the main VAE/cell_train environment."
echo ""

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "Using mamba for faster package installation"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "Using conda for package installation"
else
    echo "Error: Neither conda nor mamba found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/environment_classifier.yml"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: environment_classifier.yml not found at $ENV_FILE"
    exit 1
fi

echo "Creating conda environment from $ENV_FILE..."
$CONDA_CMD env create -f "$ENV_FILE"

echo ""
echo "=== Classifier Environment Setup Complete ==="
echo "To activate the environment, run:"
echo "  conda activate scdiffusion_classifier"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate scdiffusion_classifier"
echo "2. Apply classifier-specific file changes (see below)"
echo "3. Run classifier_train.py"
echo ""
echo "=== Classifier-specific changes required ==="
echo "The following changes are needed for classifier_train.py:"
echo ""
echo "1. guided_diffusion/cell_datasets_loader.py:"
echo "   - Add weights_only=False to torch.load() calls"
echo "   - Ensure proper DDP initialization checks"
echo ""
echo "2. classifier_train.py:"
echo "   - Add DDP initialization checks before distributed operations"
echo "   - Ensure proper error handling for single-process mode"
echo ""
echo "3. guided_diffusion/train_util.py:"
echo "   - Replace all dist.* calls with dist_util.* calls"
echo "   - Ensure proper distributed training compatibility"
