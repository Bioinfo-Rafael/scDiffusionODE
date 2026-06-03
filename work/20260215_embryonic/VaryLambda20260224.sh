#!/bin/bash
set -e

BASE_SCRIPT="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/train_20260123.sh"

echo "=========================================="
echo "Running 3 Lambda variations sequentially"
echo "=========================================="

############################################
# (1) Lambda=5
############################################
echo ""
echo "========== (1) Lamda5 =========="
bash "$BASE_SCRIPT" \
    --name Lamda5 \
    --softreg True \
    --ode_reg_lambda 5

############################################
# (2) Lambda=50
############################################
echo ""
echo "========== (2) Lamda50 =========="
bash "$BASE_SCRIPT" \
    --name Lamda50 \
    --softreg True \
    --ode_reg_lambda 50

############################################
# (3) Lambda=500
############################################
echo ""
echo "========== (3) Lamda500 =========="
bash "$BASE_SCRIPT" \
    --name Lamda500 \
    --softreg True \
    --ode_reg_lambda 500

echo ""
echo "=========================================="
echo "ALL 3 CONDITIONS FINISHED SUCCESSFULLY"
echo "=========================================="

