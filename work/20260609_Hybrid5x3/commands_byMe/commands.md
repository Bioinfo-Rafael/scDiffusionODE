python run_pipeline_5x3.py --ode_branch lora --hybrid_norm_mode scale_model --scale_input_source ml_emb --SoftReg True --ode_reg_lambda 5 --name TEST --batch_size 128 --diffusion_steps 1000 --lr_anneal_steps 10000 --num_samples 1000 --sample_batch_size 50 --save_interval 5000 --log_interval 1000 --scale_model_type simple --max_cells 50000


NAME=ALL100k STEPS=100000 BS=128 SBS=50 NSAMP=10000 \
SAVEINT=5000 LOGINT=1000 MAXCELLS=50000 \
  bash run_all_pipelines.sh