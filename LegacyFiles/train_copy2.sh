echo "Training diffusion backbone"
python cell_train_copy.py --data_dir 'data_preparation/pbmc68k.h5ad' \
    --model_name 'pbmc68k_GRN1' --lr_anneal_steps 10000 --save_dir 'output/checkpoint/backbone' --diffusion_steps 1000
echo "Training diffusion backbone done"

echo "Training classifier"
python classifier_train.py --data_dir 'data_preparation/pbmc68k_GRN1.h5ad' --model_path "output/checkpoint/classifier/pbmc68k_GRN1_classifier" \
    --iterations 10000 --vae_path 'output/checkpoint/AE/pbmc68k_GRN1/model_seed=0_step=9999.pt' --num_class 11
echo "Training classifier, done"