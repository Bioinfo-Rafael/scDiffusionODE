#!/bin/bash

echo "=== Cell Train Soft - Full Scale Training ==="
echo "Training cell_train_soft.py with production parameters"
echo "Start time: $(date)"

# Production parameters for full training
python cell_train_soft.py \
    --data_dir 'data_preparation/pbmc68k.h5ad' \
    --model_name 'pbmc68k_soft' \
    --lr_anneal_steps 10000 \
    --save_interval 2000 \
    --log_interval 100 \
    --batch_size 128 \
    --save_dir 'output/checkpoint/backbone' \
    --diffusion_steps 1000 \
    --lr 1e-4

echo "=== Cell Train Soft Training Completed ==="
echo "End time: $(date)"
echo "Check logs at: output/logs/pbmc68k_soft/"
echo "Check model at: output/checkpoint/backbone/pbmc68k_soft/"