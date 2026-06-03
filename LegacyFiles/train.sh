cd VAE
echo "Training Autoencoder, this might take a long time"
python VAE_train.py --data_dir '../data_preparation/pbmc68k.h5ad' --num_genes 32738 --save_dir '../output/checkpoint/AE/pbmc68k' --max_steps 10000 --batch_size 64
echo "Training Autoencoder done"

cd ..
echo "Training diffusion backbone"
python cell_train.py --data_dir 'data_preparation/pbmc68k.h5ad' --vae_path 'output/checkpoint/AE/pbmc68k/model_seed=0_step=10000.pt' \
    --model_name 'pbmc68k' --lr_anneal_steps 10000 --save_dir 'output/checkpoint/backbone' --diffusion_steps 1000
echo "Training diffusion backbone done"

echo "Training classifier"
python classifier_train.py --data_dir 'data_preparation/pbmc68k.h5ad' --model_path "output/checkpoint/classifier/pbmc68k_classifier" \
    --iterations 10000 --vae_path 'output/checkpoint/AE/pbmc68k/model_seed=0_step=10000.pt' --num_class 11
echo "Training classifier, done"