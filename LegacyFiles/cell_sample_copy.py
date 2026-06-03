"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import argparse

import numpy as np
import torch as th
import torch.distributed as dist
import random
import scanpy as sc

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (   
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_gaussian_diffusion,
    diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.cell_model import Cell_Unet
from ODE.ode_analysis import GeneODE, ODE_ML_Hybrid


def save_data(all_cells, traj, data_dir):
    cell_gen = all_cells
    np.savez(data_dir, cell_gen=cell_gen)
    return

def main():
    setup_seed(1234)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir='output/checkpoint/sample_logs')

    logger.log("creating model and diffusion...")
    # Create diffusion only (same as in cell_train_copy.py)
    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Load gene list from data (same as in cell_train_copy.py)
    logger.log("loading gene list from h5ad file...")
    adata = sc.read_h5ad(args.data_dir)
    gene_list = adata.var_names.tolist()
    logger.log(f"Found {len(gene_list)} genes")
    
    # Create GeneODE instance
    logger.log("creating GeneODE instance...")
    ode = GeneODE(
        gene_list=gene_list,
        edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
        device=dist_util.dev()
    )
    
    # Get timesteps from diffusion
    timesteps = diffusion.num_timesteps
    logger.log(f"Using {timesteps} timesteps for hybrid model")
    
    # Create base Cell_Unet model
    cell_unet = Cell_Unet(
        input_dim=len(gene_list),
    )
    cell_unet.to(dist_util.dev())

    # Create hybrid model (same as in cell_train_copy.py)
    logger.log("creating ODE-ML hybrid model...")
    model = ODE_ML_Hybrid(
        ode_model=ode,
        ml_model=cell_unet,
        timesteps=timesteps
    )
    model.to(dist_util.dev())
    
    # Load the trained hybrid model state
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.eval()

    logger.log("sampling...")
    all_cells = []
    while len(all_cells) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, traj = sample_fn(
            model,
            (args.batch_size, len(gene_list)),  # Use actual gene count instead of args.input_dim
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            start_time=diffusion.betas.shape[0],
        )

        # Safe distributed handling
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            try:
                world_size = dist.get_world_size()
            except ValueError:
                world_size = 1
        if world_size > 1:
            gathered_samples = [th.zeros_like(sample) for _ in range(world_size)]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_cells.extend([s.cpu().numpy() for s in gathered_samples])
        else:
            # Single process mode
            all_cells.append(sample.cpu().numpy())
        logger.log(f"created {len(all_cells) * args.batch_size} samples")

    arr = np.concatenate(all_cells, axis=0)
    save_data(arr, traj, args.sample_dir)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=12000,
        batch_size=3000,
        use_ddim=False,
        model_path="output/checkpoint/backbone/open_problem/model800000.pt",
        sample_dir="output/simulated_samples/open_problem",
        data_dir="data_preparation/pbmc68k.h5ad"  # Add data_dir parameter
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True # 设置随机数种子


if __name__ == "__main__":
    main()
