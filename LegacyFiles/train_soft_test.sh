#!/bin/bash

echo "=== Cell Train Soft - Minimum Test Run ==="
echo "Testing cell_train_soft.py with minimal parameters for code validation"
echo "Start time: $(date)"

# Set minimal parameters for quick test
python cell_train_soft.py \
    --data_dir 'data_preparation/pbmc68k.h5ad' \
    --model_name 'pbmc68k_soft_test' \
    --lr_anneal_steps 100 \
    --save_interval 50 \
    --log_interval 10 \
    --batch_size 32 \
    --save_dir 'output/checkpoint/backbone' \
    --diffusion_steps 100 \
    --lr 1e-4

echo "=== Cell Train Soft Test Completed ==="
echo "End time: $(date)"
echo "Check logs at: output/logs/pbmc68k_soft_test/"
echo "Check model at: output/checkpoint/backbone/pbmc68k_soft_test/"