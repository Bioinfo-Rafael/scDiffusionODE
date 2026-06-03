echo "Training diecho "Training diffusion backbone (NoConstraint HVG1024)"
python cell_train_20251014_NoConstraint_hvg1024.py --data_dir 'data_preparation/pbmc68k.h5ad' \
    --model_name 'pbmc68k_NoConstraint_20251014_hvg1024' --lr_anneal_steps 2000 --save_dir 'output/checkpoint/backbone_NoConstraint_hvg1024' --diffusion_steps 1000
echo "Training diffusion backbone done"

echo "Waiting for model to be saved, then generating samples..."
echo "Note: Run sampling manually after training completion:"
echo "python cell_sample_20251014_NoConstraint_hvg1024.py"backbone (NoConstraint HVG1024)"
python cell_train_20251014_NoConstraint_hvg1024.py --data_dir 'data_preparation/pbmc68k.h5ad' \
    --model_name 'pbmc68k_NoConstraint_20251014_hvg1024' --lr_anneal_steps 2000 --save_dir 'output/checkpoint/backbone_NoConstraint_hvg1024' --diffusion_steps 1000
echo "Training diffusion backbone done"

# Wait for model to be saved before sampling
echo "Waiting for model files..."
sleep 5

# Check if model file exists before sampling
if [ -f "output/checkpoint/backbone_NoConstraint_hvg1024/pbmc68k_NoConstraint_20251014_hvg1024/model002000.pt" ]; then
    echo "Generating samples from NoConstraint HVG1024 model"
    python cell_sample_20251014_NoConstraint_hvg1024.py
    echo "Sample generation done"
else
    echo "Model file not found. Please check training completion and model save interval."
    echo "Expected model file: output/checkpoint/backbone_NoConstraint_hvg1024/pbmc68k_NoConstraint_20251014_hvg1024/model002000.pt"
fiaining diffusion backbone (NoConstraint HVG1024)"
python cell_train_20251014_NoConstraint_hvg1024.py --data_dir 'data_preparation/pbmc68k.h5ad' \
    --model_name 'pbmc68k_NoConstraint_20250929' --lr_anneal_steps 2000 --save_dir 'output/checkpoint/backbone_NoConstraint_hvg1024' --diffusion_steps 1000
echo "Training diffusion backbone done"

echo "Generating samples from NoConstraint HVG1024 model"
python cell_sample_20251014_NoConstraint_hvg1024.py
echo "Sample generation done"
