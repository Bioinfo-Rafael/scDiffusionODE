"""Matched source/timestep streams, inherited optimizer/EMA and measured epochs."""
import csv
import importlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from ..common import (build_diffusion, file_hash, finite, git_commit, load_real, new_dir,
                      seed_all, source_provenance, state_hash, umap_core, write_json, write_csv, validate_config)
from ..models import build_model, freeze_from_stage1, assert_frozen, optimizer_for, update_ema
from ..src.grn import load_grn
from ..src.knn import prepare_graph
from ..analysis.diagnostics import BranchRecorder
from .checkpoints import load_stage1, save_checkpoint
from .objectives import timestep_sampler, training_loss
source_batches = importlib.import_module("work.20260916_x0predict_hybrid_additive.data").source_batches


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def train_loop(model, matrix, config, output, metadata, device, graph=None):
    """Core also exercised with synthetic fixtures; full run inputs verified by train."""
    c = config
    validate_config(c)
    batches_per_epoch = matrix.shape[0]//c["batch_size"]
    if not batches_per_epoch:
        raise ValueError("dataset has fewer cells than batch size")
    steps = c["epochs"]*batches_per_epoch if c["epochs"] is not None else c["total_steps"]
    if steps > c["lr_anneal_steps"]:
        raise ValueError("requested epochs exceed original LR schedule horizon")
    model.to(device).train()
    opt = optimizer_for(model, c)
    frozen = state_hash(model.ml_model)
    ema = {k:v.detach().clone() for k,v in model.state_dict().items()}
    assert_frozen(model, opt, frozen)
    seed_all(c["seed"])
    batches = source_batches(matrix, c["batch_size"], c["source_seed"])
    rng = np.random.default_rng(c["knn"]["seed"])
    diffusion = build_diffusion(c)
    sampler = timestep_sampler(c, diffusion)
    checkpoint_dir = new_dir(output / "checkpoints")
    write_json(output / "effective_config.json", c)
    write_json(output / "metadata.json", metadata)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    sync(device)
    started = epoch_started = time.perf_counter()
    epochs, losses = [], []
    recorder = BranchRecorder()
    knn_seconds = 0.
    epoch_start_step = 0
    last_checkpoint = None
    try:
        with (output / "losses.csv").open("x", newline="") as handle:
            writer = None
            for index in range(steps):
                x, ids = next(batches)
                x = x.to(device)
                t, weights = sampler.sample(len(x), device)
                opt.zero_grad(set_to_none=True)
                record = index == 0 or (index+1) % c["log_interval"] == 0
                model.diagnostic_sink = recorder if record and c["aux_loss"] == "reconstruction" else None
                loss, values = training_loss(model, diffusion, x, t, weights, c,
                                             ids=ids, matrix=matrix, graph=graph, rng=rng)
                model.diagnostic_sink = None
                loss.backward()
                if record and c["aux_loss"] == "knn":
                    with torch.no_grad():
                        recorder(t, model.branch_outputs(x, t[:, None]))
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        finite(name+" gradient", p.grad)
                opt.step()
                update_ema(ema, model, float(c["ema_rate"]))
                for group in opt.param_groups:
                    group["lr"] = c["lr"]*(1-index/c["lr_anneal_steps"])
                step = index+1
                row = dict(step=step, epoch=index//batches_per_epoch+1, **values)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row)
                knn_seconds += values["knn_loss_seconds"]
                losses.append(values)
                if step % c["log_interval"] == 0 or step == steps:
                    handle.flush()
                    print(json.dumps(dict(condition=c["condition"], **row)), flush=True)
                if step % c["save_interval"] == 0 or step == steps:
                    assert_frozen(model, opt, frozen)
                    if state_hash({k.removeprefix("ml_model."):v for k,v in ema.items() if k.startswith("ml_model.")}) != frozen:
                        raise AssertionError("EMA changed frozen CellUNet")
                    for name, value in ema.items():
                        finite(name, value)
                    meta = dict(metadata, step=step, actual_total_steps=steps,
                                frozen_cellunet_hash_after=frozen)
                    save_checkpoint(checkpoint_dir / f"model{step:06d}.pt", model.state_dict(), dict(meta, checkpoint_kind="raw"))
                    last_checkpoint = save_checkpoint(checkpoint_dir / f"ema_{c['ema_rate']}_{step:06d}.pt", ema, dict(meta, checkpoint_kind="ema"))
                    with (checkpoint_dir / f"optimizer{step:06d}.pt").open("xb") as f:
                        torch.save(opt.state_dict(), f)
                if step % batches_per_epoch == 0 or step == steps:
                    sync(device)
                    now = time.perf_counter()
                    epochs.append(dict(epoch=len(epochs)+1, first_step=epoch_start_step+1, last_step=step,
                                       updates=step-epoch_start_step, complete_epoch=step % batches_per_epoch == 0,
                                       seconds=now-epoch_started))
                    epoch_started, epoch_start_step = now, step
        sync(device)
        elapsed = time.perf_counter()-started
        cuda = torch.device(device).type == "cuda"
        full = [e["seconds"] for e in epochs if e["complete_epoch"]]
        report = dict(total_training_seconds=elapsed, mean_epoch_seconds=float(np.mean(full)) if full else None,
                      mean_observed_epoch_seconds=float(np.mean([e["seconds"] for e in epochs])),
                      complete_epochs=len(full), actual_updates=steps, batches_per_epoch=batches_per_epoch,
                      peak_gpu_memory_allocated=torch.cuda.max_memory_allocated(torch.device(device)) if cuda else None,
                      peak_gpu_memory_reserved=torch.cuda.max_memory_reserved(torch.device(device)) if cuda else None,
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      ode_parameter_count=sum(p.numel() for p in model.ode_model.parameters()),
                      grn_edges=metadata["grn_edges"], device=str(device),
                      knn_loss_seconds=knn_seconds, knn_loss_mean_seconds=knn_seconds/steps,
                      knn_preprocessing_seconds=graph.info["preprocessing_seconds"] if graph and not graph.cache_hit else 0.,
                      knn_original_graph_build_seconds=graph.info["preprocessing_seconds"] if graph else None,
                      knn_prepare_or_load_seconds=graph.prepare_seconds if graph else 0.,
                      knn_cache_hit=graph.cache_hit if graph else None,
                      cached_graph_bytes=graph.info["cached_graph_bytes"] if graph else 0,
                      loss_means={k: float(np.mean([v[k] for v in losses])) for k in losses[0]},
                      training_time_includes="data, forward, backward, updates, finite checks, logging, checkpoint IO; excludes kNN preprocessing and evaluation")
        # Separate post-training RHS microbenchmark; it cannot alter training RNG/state.
        model.eval()
        with torch.no_grad():
            x, _ = next(batches); x = x.to(device)
            t = torch.zeros(len(x), device=device)
            model.ode_model(x, t)
            sync(device); rhs_start = time.perf_counter()
            for _ in range(5):
                finite("RHS benchmark", model.ode_model(x, t))
            sync(device)
            report["ode_rhs_mean_seconds"] = (time.perf_counter()-rhs_start)/5
        write_csv(output / "epochs.csv", epochs)
        write_json(output / "performance.json", report)
        if recorder.rows:
            recorder.save(output, "training")
        write_json(output / "completed.json", dict(status="completed", checkpoint=str(last_checkpoint),
                   checkpoint_sha256=file_hash(last_checkpoint), step=steps))
    except BaseException as exc:
        write_json(output / "failed.json", dict(error=f"{type(exc).__name__}: {exc}"))
        raise
    return last_checkpoint


def train(config, output, device="cuda"):
    validate_config(config)
    output = new_dir(output)
    stage1, origin = load_stage1(config["stage1_checkpoint"])
    if any(file_hash(config[key]) != origin[hash_key] for key,hash_key in
           (("data_dir","data_sha256"),("edge_tsv_path","edge_tsv_sha256"))):
        raise ValueError("dataset/GRN must match original Stage1 hashes; path-only relocation is allowed")
    if config["cell_unet_hidden_num"] != stage1["metadata"]["effective_config"]["cell_unet_hidden_num"]:
        raise ValueError("CellUNet architecture differs from Stage1")
    data, genes, _ = load_real(config, stage1["metadata"]["gene_names"])
    if umap_core().gene_order_hash(genes) != origin["gene_order_hash"]:
        raise ValueError("Stage1 gene order mismatch")
    graph_edges = load_grn(genes, config["edge_tsv_path"])
    write_json(output / "gene_mapping.json", graph_edges[2])
    graph = None
    if config["aux_loss"] == "knn" and config["knn"]["lambda_knn"] > 0:
        graph = prepare_graph(data.X, config["knn"], dict(data_sha256=origin["data_sha256"],gene_order_hash=origin["gene_order_hash"]))
        write_json(output / "knn_cache.json", dict(path=str(graph.path), cache_hit=graph.cache_hit, **graph.info))
    seed_all(config["seed"])
    model = build_model(config, genes, graph=graph_edges)
    frozen = freeze_from_stage1(model, stage1["state_dict"])
    metadata = dict(effective_config=config, gene_names=genes, gene_order_hash=origin["gene_order_hash"],
                    data_sha256=origin["data_sha256"], edge_tsv_sha256=origin["edge_tsv_sha256"],
                    originating_stage1=origin, frozen_cellunet_hash_before=frozen, grn_edges=len(graph_edges[0]),
                    model_mean_type="START_X", predict_xstart=True, git_commit=git_commit(), source_sha256=source_provenance())
    return train_loop(model, data.X, config, output, metadata, device, graph)
