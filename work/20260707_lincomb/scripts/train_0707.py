#!/usr/bin/env python3
"""Train one 2026-07-07 LinComb/reverse/regime-gate experiment."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
REPO_ROOT = WORK_ROOT.parent.parent
for path in (str(REPO_ROOT), str(WORK_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from local_paths import resolve_path  # noqa: E402
from guided_diffusion import dist_util, logger  # noqa: E402
from guided_diffusion.cell_datasets_loader import load_data  # noqa: E402
from guided_diffusion.resample import create_named_schedule_sampler  # noqa: E402
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults  # noqa: E402
from guided_diffusion.train_util import TrainLoop  # noqa: E402
from scripts.common import (  # noqa: E402
    artifact_dirs, bool_value, build_model_from_config, command_string,
    create_run_dir, git_state, load_experiment_config, read_json,
    update_manifest, validate_config, write_json,
)
from utils.regime_time import estimate_lambda_max_from_adata, estimate_ts_from_lambda  # noqa: E402


def _setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _parse_scalar(text):
    lowered = str(text).strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _apply_cli(config, args):
    for key in (
        "regime_gate_mode", "regime_gate_type", "t_s", "gate_tau",
        "ts_layer", "ts_n_cells", "ts_seed", "ts_cache_path",
        "hybrid_norm_mode", "gate_mode", "gate_temperature",
        "sparse_lambda", "entropy_lambda", "off_mask_lambda",
        "lr_anneal_steps", "batch_size", "diffusion_steps", "seed",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if isinstance(config.get("ts_layer"), str) and config["ts_layer"].strip().lower() in ("", "none", "null"):
        config["ts_layer"] = None
    if args.reverse_coef is not None:
        config["reverse_coef"] = bool_value(args.reverse_coef)
    if args.force_recompute_ts:
        config["force_recompute_ts"] = True
    for item in args.set_values:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item}")
        key, value = item.split("=", 1)
        config[key] = _parse_scalar(value)
    validate_config(config)
    return config


def _resolve_ts(config, adata, diffusion):
    mode = str(config.get("regime_gate_mode", "none"))
    if mode.lower() == "none":
        config["t_s"] = None
        return {
            "enabled": False,
            "regime_gate_mode": "none",
            "regime_gate_type": config.get("regime_gate_type", "sigmoid"),
            "gate_tau": float(config.get("gate_tau", 20.0)),
            "t_s": None,
        }

    request = config.get("t_s", "auto")
    if str(request).lower() != "auto":
        value = int(request)
        if not (0 <= value < diffusion.num_timesteps):
            raise ValueError(f"manual t_s must be in [0,{diffusion.num_timesteps - 1}]")
        config["t_s"] = value
        return {
            "enabled": True,
            "source": "manual",
            "regime_gate_mode": mode,
            "regime_gate_type": config.get("regime_gate_type", "sigmoid"),
            "gate_tau": float(config.get("gate_tau", 20.0)),
            "t_s": value,
            "lambda_max": None,
            "ts_layer": config.get("ts_layer"),
        }

    cache_path = str(config.get("ts_cache_path", "") or "")
    if cache_path:
        cache_path = resolve_path(cache_path)
    force = bool_value(config.get("force_recompute_ts", False))
    if cache_path and os.path.exists(cache_path) and not force:
        cached = read_json(cache_path)
        required = (
            "t_s", "lambda_max", "alpha_bar_at_ts", "score_at_ts",
            "data_dir", "ts_layer", "ts_n_cells", "seed", "diffusion_steps",
        )
        if not all(key in cached for key in required):
            raise ValueError(f"invalid T_s cache (missing required fields): {cache_path}")
        expected = {
            "data_dir": os.path.abspath(config["data_dir"]),
            "ts_layer": config.get("ts_layer"),
            "ts_n_cells": int(config.get("ts_n_cells", 200_000)),
            "seed": int(config.get("ts_seed") if config.get("ts_seed") is not None else config.get("seed", 0)),
            "diffusion_steps": int(diffusion.num_timesteps),
        }
        mismatched = {
            key: {"cached": cached.get(key), "expected": value}
            for key, value in expected.items() if cached.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"T_s cache does not match current input/schedule: {mismatched}; "
                "use --force_recompute_ts"
            )
        cached = dict(cached)
        cached["enabled"] = True
        cached["source"] = "cache"
        cached["cache_path"] = cache_path
        config["t_s"] = int(cached["t_s"])
        return cached

    ts_seed = config.get("ts_seed")
    if ts_seed is None:
        ts_seed = config.get("seed", 0)
    lambda_info = estimate_lambda_max_from_adata(
        adata,
        layer=config.get("ts_layer"),
        n_cells=int(config.get("ts_n_cells", 200_000)),
        seed=int(ts_seed),
        use_randomized_pca=True,
    )
    ts_info = estimate_ts_from_lambda(diffusion.alphas_cumprod, lambda_info["lambda_max"])
    result = {
        "enabled": True,
        "source": "auto",
        "regime_gate_mode": mode,
        "regime_gate_type": config.get("regime_gate_type", "sigmoid"),
        "gate_tau": float(config.get("gate_tau", 20.0)),
        "ts_layer": config.get("ts_layer"),
        "ts_n_cells": int(config.get("ts_n_cells", 200_000)),
        "data_dir": os.path.abspath(config["data_dir"]),
        "diffusion_steps": int(diffusion.num_timesteps),
        **lambda_info,
        **ts_info,
    }
    config["t_s"] = int(result["t_s"])
    if cache_path:
        write_json(cache_path, result)
        result["cache_path"] = cache_path
    return result


def _model_info(model):
    info = model.get_model_info() if hasattr(model, "get_model_info") else {"class": type(model).__name__}
    ode = getattr(model, "ode_model", None)
    if ode is not None and hasattr(ode, "get_model_info"):
        info = dict(info)
        info["ode_model"] = ode.get_model_info()
    return info


def _resolve_baseline_reference(config):
    reference = dict(config.get("baseline_reference") or {})
    for key in ("run_dir", "config_path", "checkpoint_path", "sample_path"):
        value = reference.get(key)
        if value and not os.path.isabs(value):
            reference[key] = str((REPO_ROOT / value).resolve())
    mismatches = []
    baseline_config_path = reference.get("config_path")
    if baseline_config_path and os.path.exists(baseline_config_path):
        baseline = read_json(baseline_config_path)
        baseline_args = baseline.get("all_args", {})
        for key in ("K", "hybrid_norm_mode", "diffusion_steps", "seed", "lr_anneal_steps", "batch_size"):
            current_value = config.get(key)
            baseline_value = baseline.get(key, baseline_args.get(key))
            if baseline_value is None:
                mismatches.append({
                    "key": key, "current": current_value,
                    "baseline": None, "reason": "not recorded in baseline config",
                })
            elif current_value != baseline_value:
                mismatches.append({
                    "key": key, "current": current_value, "baseline": baseline_value,
                })
    else:
        mismatches.append({"key": "config_path", "reason": "missing baseline config"})
    reference["mismatches"] = mismatches
    reference["compatibility_status"] = "matched" if not mismatches else "reference_only"
    reference["train_new_model"] = False
    return reference


def main():
    args = build_parser().parse_args()
    config = _apply_cli(load_experiment_config(args.config), args)
    config["data_dir"] = resolve_path(config["data_dir"])
    config["edge_tsv_path"] = resolve_path(config["edge_tsv_path"])
    config["baseline_reference"] = _resolve_baseline_reference(config)
    experiment = config["experiment"]

    if args.dry_run:
        proposed = args.run_dir or str(WORK_ROOT / "runs" / experiment / "<timestamp>")
        print(json.dumps({"command": command_string(), "run_dir": proposed, "config": config}, indent=2))
        return

    run_dir = Path(args.run_dir).resolve() if args.run_dir else create_run_dir(experiment, args.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_dirs(run_dir)
    for key in ("train", "samples", "viz", "logs", "results"):
        paths[key].mkdir(parents=True, exist_ok=True)
    (run_dir / "commands").mkdir(exist_ok=True)
    (run_dir / "commands" / "train.txt").write_text(command_string() + "\n", encoding="utf-8")

    state = git_state()
    write_json(run_dir / "git_state.json", state)
    update_manifest(
        run_dir,
        schema_version="20260707",
        experiment=experiment,
        run_directory=str(run_dir),
        baseline_reference=config.get("baseline_reference"),
        reverse_coef=bool_value(config.get("reverse_coef", False)),
        regime_gate_mode=config.get("regime_gate_mode", "none"),
        regime_gate_type=config.get("regime_gate_type", "sigmoid"),
        gate_tau=float(config.get("gate_tau", 20.0)),
        stages={"train": {"status": "running", "started_at": datetime.now().isoformat()}},
    )

    seed = int(config.get("seed", 1234))
    _setup_seed(seed)
    dist_util.setup_dist()
    logger.configure(dir=str(paths["logs"] / "train"))

    diffusion_args = model_and_diffusion_defaults()
    for key in diffusion_args:
        if key in config:
            diffusion_args[key] = config[key]
    _, diffusion = create_model_and_diffusion(**diffusion_args)

    logger.log(f"loading AnnData: {config['data_dir']}")
    adata = sc.read_h5ad(config["data_dir"])
    gene_list = list(adata.var["gene_name"].unique())
    if len(gene_list) != adata.n_vars:
        raise ValueError("gene_name must map one-to-one to model input variables")
    ts_request = config.get("t_s")
    ts_info = _resolve_ts(config, adata, diffusion)
    config["t_s_request"] = ts_request
    write_json(paths["train"] / "ts_estimate.json", ts_info)
    write_json(run_dir / "ts_estimate.json", ts_info)

    model = build_model_from_config(config, gene_list, diffusion.num_timesteps, dist_util.dev())
    model.to(dist_util.dev())
    model_info = _model_info(model)
    model_info["ts_estimate"] = ts_info
    config.update({
        "n_genes": len(gene_list),
        "train_output_dir": str(paths["train"]),
        "command": command_string(),
        "resolved_t_s": config.get("t_s"),
        "lambda_max": ts_info.get("lambda_max"),
        "alpha_bar_at_ts": ts_info.get("alpha_bar_at_ts"),
        "score_at_ts": ts_info.get("score_at_ts"),
        "ts_estimate": ts_info,
    })
    write_json(run_dir / "exp_config.json", config)
    write_json(paths["train"] / "exp_config.json", config)
    write_json(run_dir / "model_info.json", model_info)
    update_manifest(
        run_dir, config_path=str(run_dir / "exp_config.json"), model_info=model_info,
        t_s=config.get("t_s"), lambda_max=ts_info.get("lambda_max"),
        alpha_bar_at_ts=ts_info.get("alpha_bar_at_ts"),
        score_at_ts=ts_info.get("score_at_ts"), ts_estimate=ts_info,
        ts_estimate_path=str(run_dir / "ts_estimate.json"),
    )

    data = load_data(
        data_dir=config["data_dir"], batch_size=int(config["batch_size"]),
        train_vae=True, preprocess=False, layer=config.get("ts_layer"),
    )
    schedule_sampler = create_named_schedule_sampler(config["schedule_sampler"], diffusion)
    hook_enabled = (
        float(config.get("off_mask_lambda", 0)) > 0
        or float(config.get("sparse_lambda", 0)) > 0
        or float(config.get("entropy_lambda", 0)) > 0
        or (
            config.get("denoiser_mode") == "hybrid"
            and config.get("hybrid_norm_mode") == "ratio_reg"
            and float(config.get("ratio_reg_weight", 0)) > 0
        )
    )
    checkpoint_dir = paths["train"] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    TrainLoop(
        model=model, diffusion=diffusion, data=data,
        batch_size=int(config["batch_size"]), microbatch=int(config["microbatch"]),
        lr=float(config["lr"]), ema_rate=str(config["ema_rate"]),
        log_interval=int(config["log_interval"]), save_interval=int(config["save_interval"]),
        resume_checkpoint=str(config.get("resume_checkpoint", "")),
        use_fp16=bool_value(config.get("use_fp16", False)),
        fp16_scale_growth=float(config.get("fp16_scale_growth", 1e-3)),
        schedule_sampler=schedule_sampler, weight_decay=float(config["weight_decay"]),
        lr_anneal_steps=int(config["lr_anneal_steps"]),
        model_name=str(config["model_name"]), save_dir=str(checkpoint_dir),
        ode_reg_lambda=1.0 if hook_enabled else 0.0,
        ode_reg_norm="l1", save_loss_details=bool_value(config.get("save_loss_details", True)),
    ).run_loop()

    checkpoints = glob.glob(str(checkpoint_dir / "**" / "ema_*.pt"), recursive=True)
    if not checkpoints:
        checkpoints = glob.glob(str(checkpoint_dir / "**" / "model*.pt"), recursive=True)
    latest = max(checkpoints, key=os.path.getmtime) if checkpoints else ""
    manifest = read_json(run_dir / "manifest.json")
    stages = manifest.get("stages", {})
    stages["train"] = {
        "status": "completed", "finished_at": datetime.now().isoformat(),
        "checkpoint_path": latest,
    }
    update_manifest(run_dir, stages=stages, checkpoint_path=latest)
    print(f"RUN_DIR='{run_dir}'")
    print(f"TRAINED_MODEL_PATH='{latest}'")
    print(f"EXP_CONFIG='{run_dir / 'exp_config.json'}'")
    print(f"TS_ESTIMATE='{run_dir / 'ts_estimate.json'}'")


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", dest="set_values", action="append", default=[])
    parser.add_argument("--reverse_coef", default=None)
    parser.add_argument("--regime_gate_mode", default=None)
    parser.add_argument("--regime_gate_type", default=None)
    parser.add_argument("--t_s", default=None)
    parser.add_argument("--gate_tau", type=float, default=None)
    parser.add_argument("--ts_layer", default=None)
    parser.add_argument("--ts_n_cells", type=int, default=None)
    parser.add_argument("--ts_seed", type=int, default=None)
    parser.add_argument("--ts_cache_path", default=None)
    parser.add_argument("--force_recompute_ts", action="store_true")
    parser.add_argument("--hybrid_norm_mode", default=None)
    parser.add_argument("--gate_mode", default=None)
    parser.add_argument("--gate_temperature", type=float, default=None)
    parser.add_argument("--sparse_lambda", type=float, default=None)
    parser.add_argument("--entropy_lambda", type=float, default=None)
    parser.add_argument("--off_mask_lambda", type=float, default=None)
    parser.add_argument("--lr_anneal_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


if __name__ == "__main__":
    main()
