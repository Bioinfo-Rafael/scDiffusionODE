#!/bin/bash
# HVG=1024 soft version (20251105) / train 1000 steps / sample 500 / ODE reg lambda 0.05

echo "Training diffusion backbone (Soft HVG1024 reg005)"
python /home/suzuki/Projects/scDiffusion/cell_train_soft20251014_hvg1024.py --data_dir 'data_preparation/pbmc68k.h5ad' \
    --lr_anneal_steps 1000 --save_interval 500 --log_interval 100 --batch_size 128 --diffusion_steps 1000 \
    --ode_reg_lambda 0.005 \
    --model_name 'pbmc68k_soft_20251105_hvg1024_reg005' \
    --save_dir 'output/diffusion_checkpoint_soft_hvg1024_reg005'
echo "Training diffusion backbone done"

echo "Generating samples from Soft HVG1024 reg005 model"  
python /home/suzuki/Projects/scDiffusion/cell_sample_20251014_soft_hvg1024.py \
    --model_path 'output/diffusion_checkpoint_soft_hvg1024_reg005/pbmc68k_soft_20251105_hvg1024_reg005/model001000.pt' \
    --sample_dir 'output/simulated_samples/pbmc68k_soft_20251105_hvg1024_reg005_500samples.npz' \
    --data_dir 'data_preparation/pbmc68k_preprocessed_zscore_hvg1024.h5ad'
echo "Sample generation done"