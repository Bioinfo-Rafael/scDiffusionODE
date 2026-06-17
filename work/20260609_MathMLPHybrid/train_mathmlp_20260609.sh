#!/bin/bash
# Math-MLP hybrid: train -> sample -> W-visualization pipeline (20260609)
# Parametrized by --model_type {lowrank|lincomb|matsum|lora}.
# 4 系統を同条件で比較するため、数値パラメタは固定し model_type と出力名のみ変える。

set -e

# ----------------- defaults (4 系統で共通) -----------------
MODEL_TYPE="lowrank"
RANK=16
K=8
USE_MASK_REG="True"
USE_DECAY="True"
SOFT_REG="True"
ODE_REG_LAMBDA="5"
RATIO_REG_WEIGHT=1.0
RATIO_REG_TARGET=1.0

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_type) MODEL_TYPE="$2"; shift 2 ;;
        --rank) RANK="$2"; shift 2 ;;
        --K) K="$2"; shift 2 ;;
        --use_mask_reg) USE_MASK_REG="$2"; shift 2 ;;
        --use_decay) USE_DECAY="$2"; shift 2 ;;
        --softreg) SOFT_REG="$2"; shift 2 ;;
        --ode_reg_lambda) ODE_REG_LAMBDA="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=========================================="
echo "Math-MLP Hybrid pipeline (20260609)"
echo "  model_type=${MODEL_TYPE} rank=${RANK} K=${K}"
echo "  use_mask_reg=${USE_MASK_REG} ode_reg_lambda=${ODE_REG_LAMBDA}"
echo "=========================================="

# ----------------- paths -----------------
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${WORK_DIR}/cell_train_20260609.py"
SAMPLE_SCRIPT="${WORK_DIR}/cell_sample_20260609.py"
# W/パラメタ可視化は共有 viz を使う（plot_params.py shim → ../20260609_Hybrid5x3/viz/plot_params.py）。
# 旧 visualize_W_20260609.py は legacy（plot_params が上位互換: W hist/heat + パラメタ分布 + t×step）。
VIZ_SCRIPT="${WORK_DIR}/plot_params.py"

DATA_DIR="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad"
EDGE_TSV_PATH="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv"
SAMPLE_NAME="embryonic.npz"

# ----------------- training params (4 系統で共通) -----------------
LR_ANNEAL_STEPS=30000
SAVE_INTERVAL=5000
LOG_INTERVAL=1000
BATCH_SIZE=128
DIFFUSION_STEPS=1000
LR="1e-4"
NUM_SAMPLES=1000
SAMPLE_BATCH_SIZE=50

# 出力構造: {WORK_DIR}/runs/{model}/{YYYYMMDD}/{train,sample,viz}/...（記述的 model 名）
DATE=$(date +%Y%m%d_%H%M%S)
MODEL="mathmlp__${MODEL_TYPE}"
BASE="${WORK_DIR}/runs/${MODEL}/${DATE}"
TRAIN_DIR="${BASE}/train"
SAMPLE_DIR="${BASE}/sample"
VIZ_DIR="${BASE}/viz"
mkdir -p "${TRAIN_DIR}" "${SAMPLE_DIR}" "${VIZ_DIR}"
echo "Output base: ${BASE}"

TEMP_OUTPUT=$(mktemp)
cd "$WORK_DIR"
export PATH="/home/suzuki/.conda/envs/scdiffusion/bin:$PATH"

# ----------------- 1. training -----------------
echo "Starting training..."
python "$TRAIN_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --edge_tsv_path "$EDGE_TSV_PATH" \
    --model_type "$MODEL_TYPE" \
    --rank $RANK \
    --K $K \
    --use_mask_reg "$USE_MASK_REG" \
    --use_decay "$USE_DECAY" \
    --SoftReg "$SOFT_REG" \
    --ratio_reg_weight $RATIO_REG_WEIGHT \
    --ratio_reg_target $RATIO_REG_TARGET \
    --lr_anneal_steps $LR_ANNEAL_STEPS \
    --save_interval $SAVE_INTERVAL \
    --log_interval $LOG_INTERVAL \
    --batch_size $BATCH_SIZE \
    --diffusion_steps $DIFFUSION_STEPS \
    --lr $LR \
    --ode_reg_lambda $ODE_REG_LAMBDA \
    --model_name "mathmlp_${MODEL_TYPE}_20260609" \
    --output_dir "${TRAIN_DIR}" \
    2>&1 | tee "$TEMP_OUTPUT"

TRAINED_MODEL_PATH=$(grep "TRAINED_MODEL_PATH=" "$TEMP_OUTPUT" | tail -1 | cut -d"'" -f2)
MODEL_DIR=$(grep "MODEL_DIR=" "$TEMP_OUTPUT" | tail -1 | cut -d"'" -f2)
FIELD_CONFIG=$(grep "FIELD_CONFIG=" "$TEMP_OUTPUT" | tail -1 | cut -d"'" -f2)
TOTAL_STEPS=$(grep "TOTAL_STEPS=" "$TEMP_OUTPUT" | tail -1 | cut -d"=" -f2)

echo "TRAINED_MODEL_PATH=$TRAINED_MODEL_PATH"
echo "FIELD_CONFIG=$FIELD_CONFIG"

if [ -z "$TRAINED_MODEL_PATH" ] || [ ! -f "$TRAINED_MODEL_PATH" ]; then
    echo "ERROR: trained model not found"; exit 1
fi

# ----------------- 2. sampling -----------------
echo "Starting sampling..."
python "$SAMPLE_SCRIPT" \
    --model_path "$TRAINED_MODEL_PATH" \
    --field_config "$FIELD_CONFIG" \
    --sample_name "$SAMPLE_NAME" \
    --data_dir "$DATA_DIR" \
    --edge_tsv_path "$EDGE_TSV_PATH" \
    --num_samples $NUM_SAMPLES \
    --batch_size $SAMPLE_BATCH_SIZE \
    --output_dir "${SAMPLE_DIR}" \
    2>&1 | tee -a "$TEMP_OUTPUT"

# ----------------- 3. W/パラメタ可視化（共有 plot_params）-----------------
echo "Starting W(x,t)/param visualization (shared plot_params)..."
python "$VIZ_SCRIPT" \
    --model_path "$TRAINED_MODEL_PATH" \
    --config "$FIELD_CONFIG" \
    --data_dir "$DATA_DIR" \
    --edge_tsv_path "$EDGE_TSV_PATH" \
    --output_dir "${VIZ_DIR}/params" \
    2>&1 | tee -a "$TEMP_OUTPUT" || echo "WARN: W visualization failed (non-fatal)"

echo "=========================================="
echo "PIPELINE DONE: ${MODEL_TYPE}"
echo "  model: $TRAINED_MODEL_PATH"
echo "  output: ${BASE}"
echo "=========================================="
