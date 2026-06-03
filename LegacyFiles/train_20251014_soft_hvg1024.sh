#!/bin/bash
# HVG=1024 soft version (20251014) / train 1000 steps / sample 500

echo "Training diffusion backbone (Soft HVG1024)"
python /home/suzuki/Projects/scDiffusion/cell_train_soft20251014_hvg1024.py --data_dir 'data_preparation/pbmc68k.h5ad' \
    --lr_anneal_steps 1000 --save_interval 500 --log_interval 100 --batch_size 128 --diffusion_steps 1000
echo "Training diffusion backbone done"

echo "Generating samples from Soft HVG1024 model"  
python /home/suzuki/Projects/scDiffusion/cell_sample_20251014_soft_hvg1024.py
echo "Sample generation done"
