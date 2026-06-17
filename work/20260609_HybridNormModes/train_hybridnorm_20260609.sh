#!/bin/bash
# ODE-ML HybridNorm: train -> sample pipeline (20260609)
# Parametrized by --hybrid_norm_mode {ratio_reg|normed_learned_scale|none}.
# 3 mode を同条件で比較するため、数値パラメタは固定し mode と出力名のみ変える。

set -e

# ----------------- defaults (3 mode で共通) -----------------
HYBRID_NORM_MODE="ratio_reg"
HYBRID_SCALE_INIT=1.0
HYBRID_SCALE_EPS=1e-8
SOFT_REG="True"
ODE_REG_LAMBDA="5"
RATIO_REG_WEIGHT=1.0
RATIO_REG_TARGET=1.0

while [[ $# -gt 0 ]]; do
    case $1 in
        --hybrid_norm_mode) HYBRID_NORM_MODE="$2"; shift 2 ;;
        --hybrid_scale_init) HYBRID_SCALE_INIT="$2"; shift 2 ;;
        --hybrid_scale_eps) HYBRID_SCALE_EPS="$2"; shift 2 ;;
        --softreg) SOFT_REG="$2"; shift 2 ;;
        --ode_reg_lambda) ODE_REG_LAMBDA="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=========================================="
echo "ODE-ML HybridNorm pipeline (20260609)"
echo "  hybrid_norm_mode=${HYBRID_NORM_MODE}"
echo "  scale_init=${HYBRID_SCALE_INIT} scale_eps=${HYBRID_SCALE_EPS}"
echo "  SoftReg=${SOFT_REG} ode_reg_lambda=${ODE_REG_LAMBDA}"
echo "=========================================="

# ----------------- paths -----------------
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${WORK_DIR}/cell_train_20260609.py"
SAMPLE_SCRIPT="${WORK_DIR}/cell_sample_20260609.py"

DATA_DIR="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad"
EDGE_TSV_PATH="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv"
SAMPLE_NAME="embryonic.npz"

# ----------------- training params (3 mode で共通) -----------------
LR_ANNEAL_STEPS=30000
SAVE_INTERVAL=5000
LOG_INTERVAL=1000
BATCH_SIZE=128
DIFFUSION_STEPS=1000
LR="1e-4"
NUM_SAMPLES=1000
SAMPLE_BATCH_SIZE=50

# 出力構造: {WORK_DIR}/runs/{model}/{YYYYMMDD}/{train,sample}/...（記述的 model 名）
DATE=$(date +%Y%m%d_%H%M%S)
MODEL="hybridnorm__${HYBRID_NORM_MODE}"
BASE="${WORK_DIR}/runs/${MODEL}/${DATE}"
TRAIN_DIR="${BASE}/train"
SAMPLE_DIR="${BASE}/sample"
mkdir -p "${TRAIN_DIR}" "${SAMPLE_DIR}"
echo "Output base: ${BASE}"

TEMP_OUTPUT=$(mktemp)
cd "$WORK_DIR"
export PATH="/home/suzuki/.conda/envs/scdiffusion/bin:$PATH"

# ----------------- 1. training -----------------
echo "Starting training..."
python "$TRAIN_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --edge_tsv_path "$EDGE_TSV_PATH" \
    --hybrid_norm_mode "$HYBRID_NORM_MODE" \
    --hybrid_scale_init $HYBRID_SCALE_INIT \
    --hybrid_scale_eps $HYBRID_SCALE_EPS \
    --SoftReg "$SOFT_REG" \
    --ode_reg_lambda $ODE_REG_LAMBDA \
    --ratio_reg_weight $RATIO_REG_WEIGHT \
    --ratio_reg_target $RATIO_REG_TARGET \
    --lr_anneal_steps $LR_ANNEAL_STEPS \
    --save_interval $SAVE_INTERVAL \
    --log_interval $LOG_INTERVAL \
    --batch_size $BATCH_SIZE \
    --diffusion_steps $DIFFUSION_STEPS \
    --lr $LR \
    --model_name "hybridnorm_${HYBRID_NORM_MODE}_20260609" \
    --output_dir "${TRAIN_DIR}" \
    2>&1 | tee "$TEMP_OUTPUT"

TRAINED_MODEL_PATH=$(grep "TRAINED_MODEL_PATH=" "$TEMP_OUTPUT" | tail -1 | cut -d"'" -f2)
HYBRID_CONFIG=$(grep "HYBRID_CONFIG=" "$TEMP_OUTPUT" | tail -1 | cut -d"'" -f2)
echo "TRAINED_MODEL_PATH=$TRAINED_MODEL_PATH"
echo "HYBRID_CONFIG=$HYBRID_CONFIG"

if [ -z "$TRAINED_MODEL_PATH" ] || [ ! -f "$TRAINED_MODEL_PATH" ]; then
    echo "ERROR: trained model not found"; exit 1
fi

# ----------------- 2. sampling -----------------
echo "Starting sampling..."
python "$SAMPLE_SCRIPT" \
    --model_path "$TRAINED_MODEL_PATH" \
    --hybrid_config "$HYBRID_CONFIG" \
    --sample_name "$SAMPLE_NAME" \
    --data_dir "$DATA_DIR" \
    --edge_tsv_path "$EDGE_TSV_PATH" \
    --num_samples $NUM_SAMPLES \
    --batch_size $SAMPLE_BATCH_SIZE \
    --output_dir "${SAMPLE_DIR}" \
    2>&1 | tee -a "$TEMP_OUTPUT"

echo "=========================================="
echo "PIPELINE DONE: ${HYBRID_NORM_MODE}"
echo "  model: $TRAINED_MODEL_PATH"
echo "  output: ${BASE}"
echo "=========================================="
