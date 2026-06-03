#!/usr/bin/env python3
"""
PBMC68k用 Classifier Sample実行スクリプト（試行回数制限付き）
フィルタリングで受け入れられないサンプルが続く場合、試行回数制限でスキップ

【使用履歴】
- 2025-07-28 10:24:47: Type 3-10のサンプリングを実行 (フィルタリング無効)
- ログ: output/pbmc68k_classifier_sampling_no_filter_20250728_102447.log
- 注意: このファイルは"no filter"モードで使用された
"""

import sys
import os
import datetime
sys.path.append(os.getcwd())

# classifier_sample.pyを改良して試行回数制限機能を追加
import argparse
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (   
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_and_diffusion_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import scanpy as sc
import torch
from VAE.VAE_model import VAE

def load_VAE(ae_dir, num_gene):
    autoencoder = VAE(
        num_genes=num_gene,
        device='cuda',
        seed=0,
        hidden_dim=128,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(ae_dir))
    return autoencoder

def save_data(all_cells, traj, data_dir):
    cell_gen = all_cells
    np.savez(data_dir, cell_gen=cell_gen)
    return

def main_with_max_attempts(cell_type=[0], multi=False, inter=False, weight=[10,10], max_attempts=None, min_accept_rate=0.01):
    """
    試行回数制限付きのclassifier sampling
    
    Args:
        cell_type: 目標細胞タイプ
        multi: マルチ条件生成フラグ
        inter: 補間生成フラグ
        weight: 重み
        max_attempts: 最大試行回数（Noneの場合は無制限）
        min_accept_rate: 最小受入率（これ以下の場合は早期終了）
    """
    args = create_argparser(cell_type, weight).parse_args()
    
    # コマンドライン引数から追加パラメータを取得
    if hasattr(args, 'max_attempts') and args.max_attempts is not None:
        max_attempts = args.max_attempts
    if hasattr(args, 'min_accept_rate') and args.min_accept_rate is not None:
        min_accept_rate = args.min_accept_rate

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_and_diffusion_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    if multi:
        # Multi-conditional用のclassifier設定（省略）
        pass
        
    if inter:
        # Interpolation用の設定（省略）
        pass

    def cond_fn_ori(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            grad = th.autograd.grad(selected.sum(), x_in, retain_graph=True)[0] * args.classifier_scale
            return grad
        
    def model_fn(x, t, y=None, init=None, diffusion=None):
        assert y is not None
        if args.class_cond:
            return model(x, t, y if args.class_cond else None)
        else:
            return model(x, t)

    logger.log("sampling...")
    all_cell = []
    sample_num = 0
    attempt_count = 0
    total_generated = 0
    
    # 試行回数制限のログ
    if max_attempts:
        logger.log(f"Max attempts: {max_attempts}")
    if min_accept_rate:
        logger.log(f"Min accept rate: {min_accept_rate:.3f}")
    
    while sample_num < args.num_samples:
        # 試行回数制限チェック
        if max_attempts and attempt_count >= max_attempts:
            logger.log(f"Reached max attempts ({max_attempts}). Stopping sampling.")
            logger.log(f"Final stats: {sample_num}/{args.num_samples} samples accepted from {total_generated} generated")
            if total_generated > 0:
                accept_rate = sample_num / total_generated
                logger.log(f"Accept rate: {accept_rate:.3f}")
            break
            
        # 受入率チェック（十分なサンプルが生成された後）
        if total_generated >= args.batch_size * 5:  # 5バッチ後から受入率チェック
            current_accept_rate = sample_num / total_generated
            if current_accept_rate < min_accept_rate:
                logger.log(f"Accept rate too low ({current_accept_rate:.3f} < {min_accept_rate:.3f}). Stopping sampling.")
                logger.log(f"Final stats: {sample_num}/{args.num_samples} samples accepted from {total_generated} generated")
                break
        
        attempt_count += 1
        
        model_kwargs = {}
        
        if not multi and not inter:
            classes = (cell_type[0])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
        
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample, traj = sample_fn(
            model_fn,
            (args.batch_size, args.input_dim),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn_ori,
            device=dist_util.dev(),
            noise = None,
        )

        # Distributed gathering
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            try:
                world_size = dist.get_world_size()
            except ValueError:
                world_size = 1
        if world_size > 1:
            gathered_samples = [th.zeros_like(sample) for _ in range(world_size)]
            dist.all_gather(gathered_samples, sample)
        else:
            gathered_samples = [sample]

        batch_accepted = 0
        batch_generated = 0
        
        if args.filter:
            for sample in gathered_samples:
                batch_generated += sample.shape[0]
                logits = classifier(sample, torch.zeros((sample.shape[0]), device=sample.device))
                prob = F.softmax(logits, dim=-1)
                type = torch.argmax(prob, 1)
                select_index = (type == cell_type[0])
                accepted_samples = sample[select_index]
                
                if accepted_samples.shape[0] > 0:
                    all_cell.extend([accepted_samples.cpu().numpy()])
                    batch_accepted += accepted_samples.shape[0]
                    sample_num += accepted_samples.shape[0]
            
            total_generated += batch_generated
            current_accept_rate = sample_num / total_generated if total_generated > 0 else 0
            
            logger.log(f"attempt {attempt_count}: created {batch_accepted}/{batch_generated} samples " +
                      f"(total: {sample_num}/{args.num_samples}, accept rate: {current_accept_rate:.3f})")
        else:
            for sample in gathered_samples:
                all_cell.extend([sample.cpu().numpy()])
                sample_num += sample.shape[0]
                total_generated += sample.shape[0]
            logger.log(f"attempt {attempt_count}: created {args.batch_size} samples " +
                      f"(total: {sample_num}/{args.num_samples})")

    # 最終統計
    final_accept_rate = sample_num / total_generated if total_generated > 0 else 0
    logger.log(f"Sampling finished:")
    logger.log(f"  Attempts: {attempt_count}")
    logger.log(f"  Generated: {total_generated}")
    logger.log(f"  Accepted: {sample_num}")
    logger.log(f"  Accept rate: {final_accept_rate:.3f}")
    
    if len(all_cell) > 0:
        arr = np.concatenate(all_cell, axis=0)
        save_data(arr, traj, args.sample_dir+str(cell_type[0]))
        logger.log(f"Saved {arr.shape[0]} samples to {args.sample_dir+str(cell_type[0])}")
    else:
        logger.log("No samples to save")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    logger.log("sampling complete")

def create_argparser(celltype=[0], weight=[10,10]):
    defaults = dict(
        clip_denoised=True,
        num_samples=500,
        batch_size=250,
        use_ddim=False,
        class_cond=False, 

        model_path="output/checkpoint/backbone/pbmc68k/model010000.pt",
        classifier_path="output/checkpoint/classifier/pbmc68k_classifier/model009999.pt",
        num_class=11,
        classifier_scale=2,
        ae_dir='output/checkpoint/AE/pbmc68k/model_seed=0_step=9999.pt', 
        num_gene=32738,
        sample_dir=f"output/simulated_samples/pbmc68k/conditional/",
        start_guide_steps = 500,
        filter = True,
        
        # 新しいパラメータ
        max_attempts=50,        # 最大50回試行
        min_accept_rate=0.005,  # 最小受入率0.5%
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def pbmc68k_conditional_sampling_with_limits():
    """PBMC68k用 試行回数制限付き conditional sampling"""
    
    # Parameters
    num_class = 11  # Number of cell types
    
    # 完了済みファイルをチェック
    completed_types = []
    output_dir = 'output/simulated_samples/pbmc68k/conditional/'
    
    for i in range(num_class):
        output_file = f"{output_dir}type_{i}_{i}.npz"
        if os.path.exists(output_file):
            completed_types.append(i)
            print(f"✓ Type {i} already completed")
    
    remaining_types = [i for i in range(num_class) if i not in completed_types]
    
    if not remaining_types:
        print("All cell types already completed!")
        return
    
    print(f"\nRemaining types to process: {len(remaining_types)}")
    print(f"Types: {remaining_types}")
    
    log_file = f"output/pbmc68k_classifier_sampling_limited_attempts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    print(f"\n=== PBMC68k Classifier Sampling (Limited Attempts) ===")
    print(f"Max attempts per type: 50")
    print(f"Min accept rate: 0.5%")
    print(f"Log file: {log_file}")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(log_file, 'w') as f:
        f.write(f"PBMC68k Classifier Sampling (Limited Attempts) Log\n")
        f.write(f"Start time: {datetime.datetime.now()}\n")
        f.write(f"Completed types: {completed_types}\n")
        f.write(f"Remaining types: {remaining_types}\n")
        f.write("=" * 80 + "\n\n")
    
    successful_types = completed_types.copy()
    failed_types = []
    
    for cell_type_idx in remaining_types:
        print(f"\n--- Sampling cell type {cell_type_idx} ---")
        
        # Override sys.argv
        sys.argv = [
            'pbmc68k_classifier_sample_limited.py',
            '--num_samples', '500',
            '--batch_size', '250',
            '--model_path', 'output/checkpoint/backbone/pbmc68k/model010000.pt',
            '--classifier_path', 'output/checkpoint/classifier/pbmc68k_classifier/model009999.pt',
            '--sample_dir', f'output/simulated_samples/pbmc68k/conditional/type_{cell_type_idx}_',
            '--num_class', '11',
            '--classifier_scale', '2',
            '--ae_dir', 'output/checkpoint/AE/pbmc68k/model_seed=0_step=9999.pt',
            '--num_gene', '32738',
            '--filter', 'True',
            '--max_attempts', '50',
            '--min_accept_rate', '0.005'
        ]
        
        start_time = datetime.datetime.now()
        
        try:
            main_with_max_attempts(cell_type=[cell_type_idx], max_attempts=50, min_accept_rate=0.005)
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            
            # Check if file was created
            output_file = f"{output_dir}type_{cell_type_idx}_{cell_type_idx}.npz"
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / (1024*1024)
                print(f"✓ Success! File size: {file_size:.2f} MB")
                print(f"Duration: {duration}")
                successful_types.append(cell_type_idx)
                
                with open(log_file, 'a') as f:
                    f.write(f"SUCCESS: Type {cell_type_idx}\n")
                    f.write(f"  Duration: {duration}\n")
                    f.write(f"  File size: {file_size:.2f} MB\n\n")
            else:
                print(f"✗ No output file created")
                failed_types.append((cell_type_idx, "No output file"))
                    
        except Exception as e:
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            
            print(f"✗ Error: {e}")
            failed_types.append((cell_type_idx, str(e)))
            
            with open(log_file, 'a') as f:
                f.write(f"FAILED: Type {cell_type_idx}\n")
                f.write(f"  Duration: {duration}\n")
                f.write(f"  Error: {e}\n\n")
    
    # Final summary
    print("\n" + "=" * 60)
    print("SAMPLING SUMMARY")
    print("=" * 60)
    print(f"Total cell types: {num_class}")
    print(f"Successful: {len(successful_types)}")
    print(f"Failed: {len(failed_types)}")
    
    print(f"\nLog saved to: {log_file}")

if __name__ == "__main__":
    pbmc68k_conditional_sampling_with_limits()
