#!/bin/bash
# Run train -> sample -> visualization for all requested dynamic ODE variants.

set -euo pipefail


echo "============================================================"
echo "Dynamic ODE full pipeline runner"
echo "============================================================"

# ------------------------------------------------------------
# Parse args
# ------------------------------------------------------------
NAME="dynamic_ode_compare"
QUICK_TEST="false"
DATA_DIR="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad"
EDGE_TSV_PATH="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv"
SAMPLE_NAME="embryonic.npz"

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            NAME="$2"
            shift 2
            ;;
        --quick_test)
            QUICK_TEST="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --edge_tsv_path)
            EDGE_TSV_PATH="$2"
            shift 2
            ;;
        --sample_name)
            SAMPLE_NAME="$2"
            shift 2
            ;;
        *)
            echo "[WARN] Unknown argument ignored: $1"
            shift
            ;;
    esac
done

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${WORK_DIR}/cell_train_dynamic_selectable.py"
SAMPLE_SCRIPT="${WORK_DIR}/cell_sample_dynamic_selectable.py"
VISUALIZATION_SCRIPT="${WORK_DIR}/visualization_analysis.py"

export PATH="/home/suzuki/.conda/envs/scdiffusion/bin:$PATH"

# ------------------------------------------------------------
# Base hyperparameters
# ------------------------------------------------------------
SOFT_REG="True"
ODE_REG_LAMBDA="100"
LR="1e-4"
NUM_EXPERTS=10
NUM_BASES=10
RANK=16
TIME_HIDDEN_DIM=64
ALPHA_HIDDEN_DIM=128
ALPHA_TOPK_DEFAULT=3
W_EMA=0.95
LEARN_W0="True"

if [[ "${QUICK_TEST,,}" == "true" ]]; then
    echo "[INFO] QUICK_TEST=true -> using tiny settings for smoke test"
    LR_ANNEAL_STEPS=100
    SAVE_INTERVAL=50
    LOG_INTERVAL=10
    BATCH_SIZE=8
    DIFFUSION_STEPS=100
    NUM_SAMPLES=32
    SAMPLE_BATCH_SIZE=8
else
    LR_ANNEAL_STEPS=50000
    SAVE_INTERVAL=5000
    LOG_INTERVAL=1000
    BATCH_SIZE=128
    DIFFUSION_STEPS=1000
    NUM_SAMPLES=3000
    SAMPLE_BATCH_SIZE=50
fi

# ------------------------------------------------------------
# Root output dir
# ------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ROOT_OUTPUT_DIR="${WORK_DIR}/${TIMESTAMP}_${NAME}"
COUNTER=1
ORIGINAL_ROOT_OUTPUT_DIR="${ROOT_OUTPUT_DIR}"
while [ -d "${ROOT_OUTPUT_DIR}" ]; do
    ROOT_OUTPUT_DIR="${ORIGINAL_ROOT_OUTPUT_DIR}_${COUNTER}"
    COUNTER=$((COUNTER + 1))
done
mkdir -p "${ROOT_OUTPUT_DIR}"

echo "Root output directory: ${ROOT_OUTPUT_DIR}"

echo ""
echo "Base config"
echo "  DATA_DIR         : ${DATA_DIR}"
echo "  EDGE_TSV_PATH    : ${EDGE_TSV_PATH}"
echo "  ODE_REG_LAMBDA   : ${ODE_REG_LAMBDA}"
echo "  LR_ANNEAL_STEPS  : ${LR_ANNEAL_STEPS}"
echo "  DIFFUSION_STEPS  : ${DIFFUSION_STEPS}"
echo "  NUM_EXPERTS      : ${NUM_EXPERTS}"
echo "  NUM_BASES        : ${NUM_BASES}"
echo "  RANK             : ${RANK}"
echo "  QUICK_TEST       : ${QUICK_TEST}"
echo ""

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
extract_quoted_value() {
    local key="$1"
    local file="$2"
    grep "${key}=" "$file" | tail -1 | cut -d"'" -f2
}

extract_plain_value() {
    local key="$1"
    local file="$2"
    grep "${key}=" "$file" | tail -1 | cut -d"=" -f2
}

run_pipeline() {
    local RUN_ID="$1"
    local RUN_NAME="$2"
    local ODE_MODEL_TYPE="$3"
    local ALPHA_MODE="$4"
    local ALPHA_TOPK="$5"
    local NUM_EXPERTS_LOCAL="$6"
    local NUM_BASES_LOCAL="$7"
    local RANK_LOCAL="$8"

    local RUN_OUTPUT_DIR="${ROOT_OUTPUT_DIR}/${RUN_ID}_${RUN_NAME}"
    mkdir -p "${RUN_OUTPUT_DIR}"

    local TEMP_OUTPUT
    TEMP_OUTPUT=$(mktemp)

    echo "============================================================"
    echo "[${RUN_ID}] ${RUN_NAME}"
    echo "  ode_model_type : ${ODE_MODEL_TYPE}"
    echo "  alpha_mode     : ${ALPHA_MODE}"
    echo "  alpha_topk     : ${ALPHA_TOPK}"
    echo "  num_experts    : ${NUM_EXPERTS_LOCAL}"
    echo "  num_bases      : ${NUM_BASES_LOCAL}"
    echo "  rank           : ${RANK_LOCAL}"
    echo "  output_dir     : ${RUN_OUTPUT_DIR}"
    echo "  temp_log       : ${TEMP_OUTPUT}"
    echo "============================================================"

    cd "${WORK_DIR}"

    echo "[${RUN_ID}] Training..."
    python "${TRAIN_SCRIPT}" \
        --data_dir "${DATA_DIR}" \
        --edge_tsv_path "${EDGE_TSV_PATH}" \
        --lr_anneal_steps ${LR_ANNEAL_STEPS} \
        --save_interval ${SAVE_INTERVAL} \
        --log_interval ${LOG_INTERVAL} \
        --batch_size ${BATCH_SIZE} \
        --diffusion_steps ${DIFFUSION_STEPS} \
        --lr ${LR} \
        --ode_reg_lambda ${ODE_REG_LAMBDA} \
        --SoftReg "${SOFT_REG}" \
        --output_dir "${RUN_OUTPUT_DIR}" \
        --model_name "${RUN_NAME}" \
        --ode_model_type "${ODE_MODEL_TYPE}" \
        --alpha_mode "${ALPHA_MODE}" \
        --alpha_topk ${ALPHA_TOPK} \
        --alpha_hidden_dim ${ALPHA_HIDDEN_DIM} \
        --num_experts ${NUM_EXPERTS_LOCAL} \
        --num_bases ${NUM_BASES_LOCAL} \
        --rank ${RANK_LOCAL} \
        --time_hidden_dim ${TIME_HIDDEN_DIM} \
        --w_ema ${W_EMA} \
        --learn_W0 "${LEARN_W0}" \
        2>&1 | tee "${TEMP_OUTPUT}"

    local TRAINED_MODEL_PATH
    local MODEL_DIR
    local TOTAL_STEPS
    local MODEL_NAME

    TRAINED_MODEL_PATH=$(extract_quoted_value "TRAINED_MODEL_PATH" "${TEMP_OUTPUT}")
    MODEL_DIR=$(extract_quoted_value "MODEL_DIR" "${TEMP_OUTPUT}")
    TOTAL_STEPS=$(extract_plain_value "TOTAL_STEPS" "${TEMP_OUTPUT}")
    MODEL_NAME=$(extract_quoted_value "MODEL_NAME" "${TEMP_OUTPUT}")

    if [ -z "${TRAINED_MODEL_PATH}" ] || [ ! -f "${TRAINED_MODEL_PATH}" ]; then
        echo "[${RUN_ID}] ERROR: training failed or model path not found"
        echo "Log: ${TEMP_OUTPUT}"
        exit 1
    fi

    echo "[${RUN_ID}] Sampling..."
    python "${SAMPLE_SCRIPT}" \
        --model_path "${TRAINED_MODEL_PATH}" \
        --sample_name "${SAMPLE_NAME}" \
        --data_dir "${DATA_DIR}" \
        --edge_tsv_path "${EDGE_TSV_PATH}" \
        --num_samples ${NUM_SAMPLES} \
        --batch_size ${SAMPLE_BATCH_SIZE} \
        --output_dir "${RUN_OUTPUT_DIR}" \
        --SoftReg "${SOFT_REG}" \
        --ode_model_type "${ODE_MODEL_TYPE}" \
        --alpha_mode "${ALPHA_MODE}" \
        --alpha_topk ${ALPHA_TOPK} \
        --alpha_hidden_dim ${ALPHA_HIDDEN_DIM} \
        --num_experts ${NUM_EXPERTS_LOCAL} \
        --num_bases ${NUM_BASES_LOCAL} \
        --rank ${RANK_LOCAL} \
        --time_hidden_dim ${TIME_HIDDEN_DIM} \
        --w_ema ${W_EMA} \
        --learn_W0 "${LEARN_W0}" \
        2>&1 | tee -a "${TEMP_OUTPUT}"

    local SAMPLE_FILE_PATH
    local NUM_SAMPLES_ACTUAL
    local NUM_GENES

    SAMPLE_FILE_PATH=$(extract_quoted_value "SAMPLE_FILE_PATH" "${TEMP_OUTPUT}")
    NUM_SAMPLES_ACTUAL=$(extract_plain_value "NUM_SAMPLES" "${TEMP_OUTPUT}")
    NUM_GENES=$(extract_plain_value "NUM_GENES" "${TEMP_OUTPUT}")

    if [ -z "${SAMPLE_FILE_PATH}" ] || [ ! -f "${SAMPLE_FILE_PATH}" ]; then
        echo "[${RUN_ID}] ERROR: sampling failed or sample file not found"
        echo "Log: ${TEMP_OUTPUT}"
        exit 1
    fi

    echo "[${RUN_ID}] Visualization..."
    local LOSS_DETAILS_PATH="${MODEL_DIR}/loss_details.csv"
    python "${VISUALIZATION_SCRIPT}" \
        --model_path "${TRAINED_MODEL_PATH}" \
        --sample_path "${SAMPLE_FILE_PATH}" \
        --loss_path "${LOSS_DETAILS_PATH}" \
        --model_name "${MODEL_NAME}" \
        --total_steps "${TOTAL_STEPS}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${RUN_OUTPUT_DIR}" \
        2>&1 | tee -a "${TEMP_OUTPUT}"

    echo ""
    echo "[${RUN_ID}] DONE"
    echo "  Model : ${TRAINED_MODEL_PATH}"
    echo "  Sample: ${SAMPLE_FILE_PATH}"
    echo "  Genes : ${NUM_GENES}"
    echo "  Log   : ${TEMP_OUTPUT}"
    echo ""
}

# ------------------------------------------------------------
# Run all requested variants
# ------------------------------------------------------------

# 1) f_k(x)=softplus(Wx+b), alpha = MLP softmax
run_pipeline "01" "mixture_softplus_mlp_softmax" \
    "mixture_softplus" "mlp" "0" \
    "${NUM_EXPERTS}" "${NUM_BASES}" "${RANK}"

# 2) f_k(x)=softplus(Wx+b), alpha = MLP softmax + Top-K
run_pipeline "02" "mixture_softplus_mlp_topk${ALPHA_TOPK_DEFAULT}" \
    "mixture_softplus" "mlp" "${ALPHA_TOPK_DEFAULT}" \
    "${NUM_EXPERTS}" "${NUM_BASES}" "${RANK}"

# 3) W(x)=sum alpha_k A_k, alpha = MLP softmax
run_pipeline "03" "matrix_dict_mlp_softmax" \
    "matrix_dict" "mlp" "0" \
    "${NUM_EXPERTS}" "${NUM_BASES}" "${RANK}"

# 4) W(x)=sum alpha_k A_k, alpha = linear softmax(Ax+b)
run_pipeline "04" "matrix_dict_linear_softmax" \
    "matrix_dict" "linear" "0" \
    "${NUM_EXPERTS}" "${NUM_BASES}" "${RANK}"

# 5) W(x)=sum alpha_k A_k, alpha = MLP softmax + Top-K
run_pipeline "05" "matrix_dict_mlp_topk${ALPHA_TOPK_DEFAULT}" \
    "matrix_dict" "mlp" "${ALPHA_TOPK_DEFAULT}" \
    "${NUM_EXPERTS}" "${NUM_BASES}" "${RANK}"

# 6) W(x)=W0 + sum alpha_k Delta_k, Delta_k = U_k V_k^T
run_pipeline "06" "lowrank_residual_mlp_softmax" \
    "lowrank_residual" "mlp" "0" \
    "${NUM_EXPERTS}" "${NUM_BASES}" "${RANK}"

echo "============================================================"
echo "ALL PIPELINES FINISHED"
echo "Root output: ${ROOT_OUTPUT_DIR}"
echo "============================================================"
