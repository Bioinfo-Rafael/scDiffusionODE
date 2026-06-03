echo "Training diffusion backbone (NoConstraint)"
python cell_train_20250929_NoConstraint.py --data_dir 'data_preparation/pbmc68k.h5ad' \
    --model_name 'pbmc68k_NoConstraint_20250929' --lr_anneal_steps 2000 --save_dir 'output/checkpoint/backbone_NoConstraint' --diffusion_steps 1000
echo "Training diffusion backbone done"

echo "Generating samples from NoConstraint model"
python cell_sample_20250929_NoConstraint.py
echo "Sample generation done"
