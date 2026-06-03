#!/bin/bash
set -e

BASE_SCRIPT="/home/suzuki/Projects/scDiffusion/work/20260122_1_ReconstructUMAP/train_20250122.sh"

echo "=========================================="
echo "Running 3 training conditions sequentially"
echo "=========================================="

############################################
# (1) soft
############################################
echo ""
echo "========== (1) soft =========="
bash "$BASE_SCRIPT" \
    --name soft \
    --softreg True

############################################
# (2) hard
############################################
echo ""
echo "========== (2) hard =========="
bash "$BASE_SCRIPT" \
    --name hard \
    --softreg False \
    --ode_reg_lambda 0

############################################
# (3) NoConstraintWithModel
############################################
echo ""
echo "========== (3) NoConstraintWithModel =========="
bash "$BASE_SCRIPT" \
    --name NoConstraintWithModel \
    --softreg True \
    --ode_reg_lambda 0

echo ""
echo "=========================================="
echo "ALL 3 CONDITIONS FINISHED SUCCESSFULLY"
echo "=========================================="
