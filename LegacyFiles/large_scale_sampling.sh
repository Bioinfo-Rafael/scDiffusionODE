#!/bin/bash

# Large Scale Sampling Script for scDiffusion
# Estimated total runtime: ~5 hours
# This script performs both unconditional and conditional sampling

set -e  # Exit on any error

echo "=================================="
echo "Large Scale scDiffusion Sampling"
echo "Start time: $(date)"
echo "=================================="

# Environment setup
export CONDA_ENV="scdiffusion_classifier"
export PROJECT_DIR="/home/suzuki/Projects/scDiffusion"

# Model paths
export BACKBONE_MODEL="output/checkpoint/backbone/pancreas/model010000.pt"
export CLASSIFIER_MODEL="output/checkpoint/classifier/pancreas_classifier/model000009.pt"

# Output directories
export UNCONDITIONAL_DIR="output/simulated_samples/pancreas/large_unconditional"
export CONDITIONAL_DIR="output/simulated_samples/pancreas/large_conditional"

# Create output directories
mkdir -p $UNCONDITIONAL_DIR
mkdir -p $CONDITIONAL_DIR

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate $CONDA_ENV

cd $PROJECT_DIR

echo "Starting large scale sampling..."
echo "Backbone model: $BACKBONE_MODEL"
echo "Classifier model: $CLASSIFIER_MODEL"

# ========================================
# Part 1: Large Scale Unconditional Sampling
# ========================================
echo ""
echo "========================================="
echo "PART 1: Unconditional Sampling"
echo "Target: 5,000 samples, batch size: 500"
echo "Estimated time: ~1 hour"
echo "========================================="

start_time=$(date +%s)

python cell_sample.py \
    --num_samples 5000 \
    --batch_size 500 \
    --model_path $BACKBONE_MODEL \
    --sample_dir ${UNCONDITIONAL_DIR}/unconditional_5k.npz

end_time=$(date +%s)
unconditional_duration=$((end_time - start_time))
echo "Unconditional sampling completed in ${unconditional_duration} seconds"

# ========================================
# Part 2: Large Scale Conditional Sampling
# ========================================
echo ""
echo "========================================="
echo "PART 2: Conditional Sampling"
echo "Target: 8 cell types × 625 samples = 5,000 total"
echo "Batch size: 250 per type"
echo "Estimated time: ~2 hours"
echo "========================================="

# Create a modified classifier_sample.py script for batch processing
cat > batch_conditional_sample.py << 'EOF'
import sys
import os
sys.path.append(os.getcwd())

from classifier_sample import main
import argparse

def batch_conditional_sampling():
    """Run conditional sampling for all 8 cell types"""
    
    # Parameters for each cell type
    num_samples_per_type = 625
    batch_size = 250
    
    for cell_type in range(8):
        print(f"\n--- Sampling cell type {cell_type} ---")
        print(f"Target samples: {num_samples_per_type}")
        
        # Override sys.argv to simulate command line arguments
        sys.argv = [
            'batch_conditional_sample.py',
            '--num_samples', str(num_samples_per_type),
            '--batch_size', str(batch_size),
            '--model_path', 'output/checkpoint/backbone/pancreas/model010000.pt',
            '--classifier_path', 'output/checkpoint/classifier/pancreas_classifier/model000009.pt',
            '--sample_dir', f'output/simulated_samples/pancreas/large_conditional/conditional_type',
            '--num_class', '8',
            '--classifier_scale', '2'
        ]
        
        try:
            main(cell_type=[cell_type])
            print(f"✓ Cell type {cell_type} sampling completed")
        except Exception as e:
            print(f"✗ Error in cell type {cell_type}: {e}")
            continue

if __name__ == "__main__":
    batch_conditional_sampling()
EOF

start_time=$(date +%s)

python batch_conditional_sample.py

end_time=$(date +%s)
conditional_duration=$((end_time - start_time))
echo "Conditional sampling completed in ${conditional_duration} seconds"

# ========================================
# Summary and Statistics
# ========================================
echo ""
echo "========================================="
echo "SAMPLING COMPLETED - SUMMARY"
echo "========================================="

total_duration=$((unconditional_duration + conditional_duration))
total_hours=$((total_duration / 3600))
total_minutes=$(((total_duration % 3600) / 60))

echo "Total execution time: ${total_hours}h ${total_minutes}m"
echo "Unconditional sampling: ${unconditional_duration}s"
echo "Conditional sampling: ${conditional_duration}s"

# Check output files
echo ""
echo "Generated files:"
echo "----------------"

if [ -f "${UNCONDITIONAL_DIR}/unconditional_5k.npz" ]; then
    size=$(ls -lh "${UNCONDITIONAL_DIR}/unconditional_5k.npz" | awk '{print $5}')
    echo "✓ Unconditional samples: ${UNCONDITIONAL_DIR}/unconditional_5k.npz (${size})"
else
    echo "✗ Unconditional samples: FILE NOT FOUND"
fi

echo ""
echo "Conditional samples:"
for i in {0..7}; do
    file="${CONDITIONAL_DIR}/conditional_type${i}.npz"
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "✓ Cell type $i: $(basename $file) (${size})"
    else
        echo "✗ Cell type $i: FILE NOT FOUND"
    fi
done

# Generate summary statistics
echo ""
echo "========================================="
echo "NEXT STEPS"
echo "========================================="
echo "1. Verify sample quality with downstream analysis"
echo "2. Compare conditional vs unconditional distributions"
echo "3. Run biological validation notebooks in exp_script/"
echo ""
echo "All files saved in:"
echo "  - Unconditional: $UNCONDITIONAL_DIR/"
echo "  - Conditional: $CONDITIONAL_DIR/"

echo ""
echo "Sampling pipeline completed at: $(date)"
echo "=================================="

# Clean up temporary script
rm -f batch_conditional_sample.py
