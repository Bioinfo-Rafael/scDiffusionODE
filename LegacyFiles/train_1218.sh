#!/bin/bash
# Auto Training and Sampling Script (20251127)
# Automatically passes training results to sampling

set -e  # Exit on any error

echo "=========================================="
echo "Auto Training & Sampling Script (20251127)"
echo "=========================================="

# Set paths
SCRIPT_DIR="/home/suzuki/Projects/scDiffusion"
TRAIN_SCRIPT="${SCRIPT_DIR}/cell_train_soft20251106_hvg1024.py"
SAMPLE_SCRIPT="${SCRIPT_DIR}/cell_sample_20251106_soft_hvg1024.py"

# Training parameters (matching train_20251106_soft.sh)
DATA_DIR="/home/suzuki/Projects/scDiffusion/data_preparation/pbmc68k.h5ad"
LR_ANNEAL_STEPS=10000  # Fixed: was 1000, now matches train_20251106_soft.sh
SAVE_INTERVAL=100
LOG_INTERVAL=100
BATCH_SIZE=128
DIFFUSION_STEPS=1000
LR="1e-4"
ODE_REG_LAMBDA="5"  # Fixed: was 0.0005, now matches train_20251106_soft.sh

# Sampling parameters
NUM_SAMPLES=2000
SAMPLE_BATCH_SIZE=50

# Create temporary file to capture training output
TEMP_OUTPUT=$(mktemp)

echo "Starting training..."
echo "Temporary output file: $TEMP_OUTPUT"

# Run training and capture output
cd "$SCRIPT_DIR"
# Use direct Python path instead of conda activate
export PATH="/home/suzuki/.conda/envs/scdiffusion/bin:$PATH"

python "$TRAIN_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --lr_anneal_steps $LR_ANNEAL_STEPS \
    --save_interval $SAVE_INTERVAL \
    --log_interval $LOG_INTERVAL \
    --batch_size $BATCH_SIZE \
    --diffusion_steps $DIFFUSION_STEPS \
    --lr $LR \
    --ode_reg_lambda $ODE_REG_LAMBDA \
    2>&1 | tee "$TEMP_OUTPUT"

echo ""
echo "Training completed! Extracting model information..."

# Extract shell variables from training output
TRAINED_MODEL_PATH=$(grep "TRAINED_MODEL_PATH=" "$TEMP_OUTPUT" | tail -1 | cut -d"'" -f2)
MODEL_DIR=$(grep "MODEL_DIR=" "$TEMP_OUTPUT" | tail -1 | cut -d"'" -f2)
TOTAL_STEPS=$(grep "TOTAL_STEPS=" "$TEMP_OUTPUT" | tail -1 | cut -d"=" -f2)
MODEL_NAME=$(grep "MODEL_NAME=" "$TEMP_OUTPUT" | tail -1 | cut -d"'" -f2)

echo "Extracted training results:"
echo "  Model Path: $TRAINED_MODEL_PATH"
echo "  Model Directory: $MODEL_DIR"
echo "  Total Steps: $TOTAL_STEPS"
echo "  Model Name: $MODEL_NAME"

# Check if model path was extracted successfully
if [ -z "$TRAINED_MODEL_PATH" ]; then
    echo "ERROR: Could not extract model path from training output"
    echo "Please check the training log in: $TEMP_OUTPUT"
    exit 1
fi

if [ ! -f "$TRAINED_MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $TRAINED_MODEL_PATH"
    exit 1
fi

echo ""
echo "Starting sampling with trained model..."

# Determine preprocessed data path
PREPROCESSED_DATA_DIR="data_preparation/pbmc68k_preprocessed_zscore_hvg1024_20251105.h5ad"

# Create sampling output filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAMPLE_DIR="output/simulated_samples/${MODEL_NAME}_${TOTAL_STEPS}steps_${NUM_SAMPLES}samples_${TIMESTAMP}.npz"

# Create sampling output directory if it doesn't exist
mkdir -p "$(dirname "$SAMPLE_DIR")"

# Run sampling
python "$SAMPLE_SCRIPT" \
    --model_path "$TRAINED_MODEL_PATH" \
    --sample_dir "$SAMPLE_DIR" \
    --data_dir "$PREPROCESSED_DATA_DIR" \
    --num_samples $NUM_SAMPLES \
    --batch_size $SAMPLE_BATCH_SIZE \
    2>&1 | tee -a "$TEMP_OUTPUT"

echo ""
echo "Sampling completed! Extracting sample information..."

# Extract sampling results
SAMPLE_FILE_PATH=$(grep "SAMPLE_FILE_PATH=" "$TEMP_OUTPUT" | tail -1 | cut -d"'" -f2)
NUM_SAMPLES_ACTUAL=$(grep "NUM_SAMPLES=" "$TEMP_OUTPUT" | tail -1 | cut -d"=" -f2)
NUM_GENES=$(grep "NUM_GENES=" "$TEMP_OUTPUT" | tail -1 | cut -d"=" -f2)

echo "=========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "=========================================="
echo "Training Results:"
echo "  ✓ Model saved to: $TRAINED_MODEL_PATH"
echo "  ✓ Training steps: $TOTAL_STEPS"
echo "  ✓ Model directory: $MODEL_DIR"
echo ""
echo "Sampling Results:"
echo "  ✓ Samples saved to: $SAMPLE_FILE_PATH"
echo "  ✓ Number of samples: $NUM_SAMPLES_ACTUAL"
echo "  ✓ Number of genes: $NUM_GENES"
echo ""
echo "Files for analysis:"
echo "  - Model file: $TRAINED_MODEL_PATH"
echo "  - Sample file: $SAMPLE_FILE_PATH"
echo "  - Loss details: ${MODEL_DIR}/loss_details.csv"
echo "  - Full log: $TEMP_OUTPUT"
echo "=========================================="

# Optional: Clean up temp file (uncomment if you don't want to keep the log)
# rm "$TEMP_OUTPUT"

echo ""
echo "Starting visualization analysis..."

# Run visualization script
LOSS_DETAILS_PATH="${MODEL_DIR}/loss_details.csv"
python "${SCRIPT_DIR}/visualization_analysis.py" \
    --model_path "$TRAINED_MODEL_PATH" \
    --sample_path "$SAMPLE_FILE_PATH" \
    --loss_path "$LOSS_DETAILS_PATH" \
    --model_name "$MODEL_NAME" \
    --total_steps "$TOTAL_STEPS" \
    2>&1 | tee -a "$TEMP_OUTPUT"

# Extract visualization results
VIZ_OUTPUT_DIR=$(grep "Output directory:" "$TEMP_OUTPUT" | tail -1 | awk '{print $NF}')

echo ""
echo "=========================================="
echo "COMPLETE PIPELINE FINISHED SUCCESSFULLY"
echo "=========================================="
echo "Training Results:"
echo "  ✓ Model saved to: $TRAINED_MODEL_PATH"
echo "  ✓ Training steps: $TOTAL_STEPS"
echo "  ✓ Model directory: $MODEL_DIR"
echo ""
echo "Sampling Results:"
echo "  ✓ Samples saved to: $SAMPLE_FILE_PATH"
echo "  ✓ Number of samples: $NUM_SAMPLES_ACTUAL"
echo "  ✓ Number of genes: $NUM_GENES"
echo ""
echo "Visualization Results:"
echo "  ✓ Analysis output: $VIZ_OUTPUT_DIR"
echo "  ✓ Loss curves: ${VIZ_OUTPUT_DIR}/loss_curves.png"
echo "  ✓ Gene correlation: ${VIZ_OUTPUT_DIR}/gene_correlation.png"
echo "  ✓ UMAP analysis: ${VIZ_OUTPUT_DIR}/umap_analysis.png"
echo ""
echo "All Files for Analysis:"
echo "  - Model file: $TRAINED_MODEL_PATH"
echo "  - Sample file: $SAMPLE_FILE_PATH"
echo "  - Loss details: $LOSS_DETAILS_PATH"
echo "  - Visualization directory: $VIZ_OUTPUT_DIR"
echo "  - Full log: $TEMP_OUTPUT"
echo "=========================================="

echo "Complete pipeline execution finished at $(date)"