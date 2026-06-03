#!/bin/bash
# Auto Training and Sampling Script (20251127)
# Automatically passes training results to sampling

set -e  # Exit on any error

echo "=========================================="
echo "Auto Training & Sampling Script (20251127)"
echo "=========================================="

# Parse command line arguments
NAME="Sigmoid"
SOFT_REG="True"  # default soft regularization enabled
ODE_REG_LAMBDA="10"
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            NAME="$2"
            shift 2
            ;;
        --softreg)
            SOFT_REG="$2"  # expects true/false
            shift 2
            ;;
        --ode_reg_lambda)
            ODE_REG_LAMBDA="$2"  # support lowercase variant
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Set paths
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${WORK_DIR}/cell_train_20260416.py"
SAMPLE_SCRIPT="${WORK_DIR}/cell_sample_20260416.py"
VISUALIZATION_SCRIPT="${WORK_DIR}/visualization_analysis.py"


# Data path (preprocessed data used for both training and sampling)
DATA_DIR="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad"
EDGE_TSV_PATH="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv"
SAMPLE_NAME="embryonic.npz" #"pbmc68k.npz"

# Training parameters (matching train_20251106_soft.sh)
LR_ANNEAL_STEPS=50000  # Fixed: was 1000, now matches train_20251106_soft.sh
SAVE_INTERVAL=5000
LOG_INTERVAL=1000
BATCH_SIZE=128
DIFFUSION_STEPS=1000
LR="1e-4"


# Sampling parameters
NUM_SAMPLES=3000
SAMPLE_BATCH_SIZE=50

# Create output directory with timestamp and name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${WORK_DIR}/${TIMESTAMP}_${NAME}"

# Handle existing directory with suffix
COUNTER=1
ORIGINAL_OUTPUT_DIR="${OUTPUT_DIR}"
while [ -d "${OUTPUT_DIR}" ]; do
    OUTPUT_DIR="${ORIGINAL_OUTPUT_DIR}_${COUNTER}"
    COUNTER=$((COUNTER + 1))
done

mkdir -p "${OUTPUT_DIR}"
echo "Output directory created: ${OUTPUT_DIR}"

# Create temporary file to capture training output
TEMP_OUTPUT=$(mktemp)

echo ""
echo "Starting training..."
echo "Temporary output file: $TEMP_OUTPUT"
echo "Output directory: ${OUTPUT_DIR}"

# Run training and capture output
cd "$WORK_DIR"
# Use direct Python path instead of conda activate
export PATH="/home/suzuki/.conda/envs/scdiffusion/bin:$PATH"

python "$TRAIN_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --edge_tsv_path "$EDGE_TSV_PATH" \
    --lr_anneal_steps $LR_ANNEAL_STEPS \
    --save_interval $SAVE_INTERVAL \
    --log_interval $LOG_INTERVAL \
    --batch_size $BATCH_SIZE \
    --diffusion_steps $DIFFUSION_STEPS \
    --lr $LR \
    --ode_reg_lambda $ODE_REG_LAMBDA \
    --SoftReg "$SOFT_REG" \
    --output_dir "${OUTPUT_DIR}" \
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

# Run sampling
python "$SAMPLE_SCRIPT" \
    --model_path "$TRAINED_MODEL_PATH" \
    --sample_name "$SAMPLE_NAME" \
    --data_dir "$DATA_DIR" \
    --edge_tsv_path "$EDGE_TSV_PATH" \
    --num_samples $NUM_SAMPLES \
    --batch_size $SAMPLE_BATCH_SIZE \
    --output_dir "${OUTPUT_DIR}" \
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
# MODEL_DIR points to the directory containing the model (e.g., {timestamp}_train/checkpoints/{timestamp})
# loss_details.csv is saved in that same directory
LOSS_DETAILS_PATH="${MODEL_DIR}/loss_details.csv"
python "${VISUALIZATION_SCRIPT}" \
    --model_path "$TRAINED_MODEL_PATH" \
    --sample_path "$SAMPLE_FILE_PATH" \
    --loss_path "$LOSS_DETAILS_PATH" \
    --model_name "$MODEL_NAME" \
    --total_steps "$TOTAL_STEPS" \
    --data_dir "$DATA_DIR" \
    --output_dir "${OUTPUT_DIR}" \
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