#!/bin/bash

# scDiffusion Environment Setup Script
# This script creates a conda environment for running scDiffusion VAE_train and cell_train

set -e  # Exit on any error

echo "=== scDiffusion Environment Setup ==="
echo "This script will create a conda environment named 'scdiffusion'"
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
ENV_FILE="$SCRIPT_DIR/environment.yml"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: environment.yml not found at $ENV_FILE"
    exit 1
fi

echo "Creating conda environment from $ENV_FILE..."
$CONDA_CMD env create -f "$ENV_FILE"

echo ""
echo "=== Environment Setup Complete ==="
echo "To activate the environment, run:"
echo "  conda activate scdiffusion"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate scdiffusion"
echo "2. Apply file changes from ../file_changes/"
echo "3. Prepare data using ../data_preparation/data_preparation.ipynb"
echo "4. Run VAE_train.py and cell_train.py"
