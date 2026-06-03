#!/bin/bash
# HVG=1024 soft version (20251106) / train 1000 steps / sample 500

echo "Training diffusion backbone (Soft HVG1024 20251106)"
python /home/suzuki/Projects/scDiffusion/cell_train_soft20251106_hvg1024.py --data_dir '/home/suzuki/Projects/scDiffusion/data_preparation/pbmc68k.h5ad' \
    --lr_anneal_steps 3000 --save_interval 500 --log_interval 100 --batch_size 32 --diffusion_steps 1000 \
    --lr 5e-6 \
    --ode_reg_lambda 0.005 \
    --model_name 'pbmc68k_soft_20251106_hvg1024' \
    --save_dir 'output/diffusion_checkpoint_soft_hvg1024_20251106'
echo "Training diffusion backbone done"

echo "Generating samples from Soft HVG1024 20251106 model"  
python /home/suzuki/Projects/scDiffusion/cell_sample_20251106_soft_hvg1024.py \
    --model_path 'output/diffusion_checkpoint_soft_hvg1024_20251106/pbmc68k_soft_20251106_hvg1024/model003000.pt' \
    --sample_dir 'output/simulated_samples/pbmc68k_soft_20251106_hvg1024_500samples.npz' \
    --data_dir 'data_preparation/pbmc68k_preprocessed_zscore_hvg1024_20251105.h5ad'
echo "Sample generation done"