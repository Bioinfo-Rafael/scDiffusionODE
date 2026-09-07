#!/usr/bin/env python3
"""Train one isolated 20260816 direct-Hill condition with safe segmented resume."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
import signal
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
LEGACY_0707_ROOT = REPO_ROOT / "work" / "20260707_lincomb"
for search_path in (REPO_ROOT, SUITE_ROOT, HERE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from common import (  # noqa: E402
    ConfigurationError,
    apply_set_overrides,
    artifact_dirs,
    atomic_write_text,
    bool_value,
    build_model_from_config,
    checkpoint_metadata,
    command_string,
    config_fingerprint,
    config_provenance,
    create_run_dir,
    git_state,
    latest_checkpoint_bundle,
    load_experiment_config,
    next_checkpoint_segment,
    now_iso,
    prepare_run_dirs,
    read_json,
    record_command,
    resolve_config_paths,
    update_manifest,
    update_stage,
    validate_config,
    validate_resume_bundle,
    validate_run_dir,
    write_json,
)


def _setup_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True


def _select_device(torch: Any, dist_util: Any, requested: str) -> Any:
    choice = str(requested or "auto").lower()
    if choice == "auto":
        dist_util.setup_dist()
        return dist_util.dev()
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        dist_util.setup_dist()
        return dist_util.dev()
    if choice == "mps":
        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not available:
            raise RuntimeError("device=mps requested but MPS is not available")
        device = torch.device("mps")
    elif choice == "cpu":
        device = torch.device("cpu")
    else:
        raise ConfigurationError(f"unsupported device: {requested!r}")

    # The legacy TrainLoop consults dist_util.dev() internally.  Explicit CPU
    # and MPS runs are single-process, so point that accessor at the requested
    # device without initializing a CUDA distributed process group.
    dist_util.dev = lambda: device
    return device


def _legacy_regime_time_module() -> Any:
    module_path = LEGACY_0707_ROOT / "utils" / "regime_time.py"
    spec = importlib.util.spec_from_file_location("regime_time_20260707", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load 20260707 T_s utility: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ts_expected_cache_fields(config: Mapping[str, Any], diffusion: Any) -> dict[str, Any]:
    ts_seed = config.get("ts_seed")
    if ts_seed is None:
        ts_seed = config.get("seed", 0)
    data_path = Path(str(config["data_dir"]))
    return {
        "data_dir": str(data_path),
        "data_size_bytes": data_path.stat().st_size,
        "data_mtime_ns": data_path.stat().st_mtime_ns,
        "ts_layer": config.get("ts_layer"),
        "ts_n_cells": int(config.get("ts_n_cells", 200_000)),
        "seed": int(ts_seed),
        "diffusion_steps": int(diffusion.num_timesteps),
        "noise_schedule": str(config.get("noise_schedule", "linear")),
        "rescale_timesteps": bool_value(config.get("rescale_timesteps", False)),
        "timestep_respacing": config.get("timestep_respacing", ""),
    }


def resolve_ts(
    config: dict[str, Any],
    adata: Any,
    diffusion: Any,
    *,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Resolve manual/automatic T_s and strictly validate a suite-local cache."""

    mode = str(config.get("regime_gate_mode", "none"))
    if mode.lower() == "none":
        config["t_s"] = None
        return {
            "enabled": False,
            "source": "disabled",
            "regime_gate_mode": "none",
            "regime_gate_type": config.get("regime_gate_type", "sigmoid"),
            "gate_tau": float(config.get("gate_tau", 80.0)),
            "t_s": None,
        }

    request = config.get("t_s", "auto")
    if str(request).strip().lower() != "auto":
        value = int(request)
        if not 0 <= value < int(diffusion.num_timesteps):
            raise ConfigurationError(
                f"manual t_s must be in [0, {diffusion.num_timesteps - 1}], got {value}"
            )
        config["t_s"] = value
        return {
            "enabled": True,
            "source": "manual",
            "regime_gate_mode": mode,
            "regime_gate_type": config.get("regime_gate_type", "sigmoid"),
            "gate_tau": float(config.get("gate_tau", 80.0)),
            "t_s": value,
            "lambda_max": None,
            "ts_layer": config.get("ts_layer"),
        }

    cache_text = str(config.get("ts_cache_path", "") or "")
    if not cache_text:
        raise ConfigurationError("automatic T_s requires a suite-local ts_cache_path")
    cache_path = Path(cache_text)
    expected = _ts_expected_cache_fields(config, diffusion)
    force = force_recompute or bool_value(config.get("force_recompute_ts", False))
    if cache_path.exists() and not force:
        cached = read_json(cache_path)
        required = {
            "t_s",
            "lambda_max",
            "alpha_bar_at_ts",
            "score_at_ts",
            *expected.keys(),
        }
        missing = sorted(required.difference(cached))
        if missing:
            raise ConfigurationError(
                f"invalid T_s cache {cache_path}; missing {missing}. "
                "Use --force-recompute-ts to replace it."
            )
        mismatches = {
            key: {"cached": cached.get(key), "expected": expected_value}
            for key, expected_value in expected.items()
            if cached.get(key) != expected_value
        }
        if mismatches:
            raise ConfigurationError(
                f"T_s cache does not match the current data/schedule: {mismatches}. "
                "Use --force-recompute-ts to replace it."
            )
        t_s = int(cached["t_s"])
        if not 0 <= t_s < int(diffusion.num_timesteps):
            raise ConfigurationError(f"cached t_s is outside the diffusion schedule: {t_s}")
        config["t_s"] = t_s
        result = dict(cached)
        result.update({"enabled": True, "source": "cache", "cache_path": str(cache_path)})
        return result

    utility = _legacy_regime_time_module()
    lambda_info = utility.estimate_lambda_max_from_adata(
        adata,
        layer=config.get("ts_layer"),
        n_cells=int(config.get("ts_n_cells", 200_000)),
        seed=int(expected["seed"]),
        use_randomized_pca=True,
    )
    ts_info = utility.estimate_ts_from_lambda(diffusion.alphas_cumprod, lambda_info["lambda_max"])
    result = {
        "enabled": True,
        "source": "auto",
        "cache_path": str(cache_path),
        "regime_gate_mode": mode,
        "regime_gate_type": config.get("regime_gate_type", "sigmoid"),
        "gate_tau": float(config.get("gate_tau", 80.0)),
        **expected,
        **lambda_info,
        **ts_info,
    }
    config["t_s"] = int(result["t_s"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, result)
    return result


def _gene_list(adata: Any) -> list[str]:
    if "gene_name" not in adata.var.columns:
        raise ConfigurationError("AnnData .var must contain the baseline 'gene_name' column")
    genes = [str(value) for value in adata.var["gene_name"].tolist()]
    if len(genes) != int(adata.n_vars):
        raise ConfigurationError("gene_name must map one-to-one to model input variables")
    if len(set(genes)) != len(genes):
        raise ConfigurationError("gene_name contains duplicates; edge-mask alignment would be ambiguous")
    return genes


def _model_info(model: Any) -> dict[str, Any]:
    if hasattr(model, "get_model_info"):
        info = dict(model.get_model_info())
    else:
        info = {"class": type(model).__name__}
    ode = getattr(model, "ode_model", None)
    if ode is not None and hasattr(ode, "get_model_info"):
        info.setdefault("ode_model", ode.get_model_info())
    info["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    info["trainable_parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return info


def _next_attempt(manifest: Mapping[str, Any]) -> int:
    histories = manifest.get("train_attempts", [])
    return len(histories) + 1 if isinstance(histories, list) else 1


def _append_train_attempt(run_dir: Path, payload: Mapping[str, Any]) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    histories = list(manifest.get("train_attempts", []))
    histories.append(dict(payload))
    update_manifest(run_dir, train_attempts=histories)


def _record_checkpoint(run_dir: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    metadata = checkpoint_metadata(bundle)
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    history = list(manifest.get("checkpoint_history", []))
    identity = (metadata["raw_checkpoint_path"], int(metadata["step"]))
    if not any((item.get("raw_checkpoint_path"), int(item.get("step", -1))) == identity for item in history):
        history.append(metadata)
    update_manifest(
        run_dir,
        checkpoint_history=history,
        checkpoint_path=metadata["ema_checkpoint_path"],
        raw_checkpoint_path=metadata["raw_checkpoint_path"],
        ema_checkpoint_path=metadata["ema_checkpoint_path"],
        ema_checkpoint_paths=metadata["ema_checkpoint_paths"],
        optimizer_checkpoint_path=metadata["optimizer_checkpoint_path"],
        checkpoint_step=int(metadata["step"]),
        checkpoint_metadata=metadata,
    )
    return metadata


def _requested_config_for_run(
    config_path: Path,
    set_values: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_experiment_config(config_path)
    config = apply_set_overrides(config, set_values)
    config = resolve_config_paths(config)
    validate_config(config)
    provenance = config_provenance(config_path)
    return config, provenance


def _config_differences(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    keys = sorted(set(expected).union(actual))
    return {
        key: {"stored": expected.get(key), "requested": actual.get(key)}
        for key in keys
        if expected.get(key) != actual.get(key)
    }


def _resolve_resume(
    run_dir: Path,
    config: Mapping[str, Any],
    resume: str,
) -> dict[str, Any] | None:
    latest = latest_checkpoint_bundle(run_dir, config["ema_rate"], require_complete=True)
    if not resume:
        if latest is not None:
            raise RuntimeError(
                f"run already contains checkpoint step {latest['step']}; use --resume auto "
                "or select a new batch/run directory"
            )
        return None
    if resume == "auto":
        return (
            validate_resume_bundle(run_dir, latest["raw_checkpoint_path"], config["ema_rate"])
            if latest is not None
            else None
        )
    return validate_resume_bundle(run_dir, resume, config["ema_rate"])


def _install_interrupt_handlers() -> None:
    def interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)


def run_training(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    requested_config, provenance = _requested_config_for_run(config_path, args.set_values)
    experiment = str(requested_config["experiment"])

    proposed_run = (
        validate_run_dir(args.run_dir, experiment)
        if args.run_dir
        else (SUITE_ROOT / "runs" / experiment / (args.batch_id or "<timestamp>"))
    )
    if args.dry_run:
        resume_preview = None
        if args.run_dir and Path(args.run_dir).exists() and args.resume == "auto":
            resume_preview = latest_checkpoint_bundle(
                Path(args.run_dir), requested_config["ema_rate"], require_complete=True
            )
        print(json.dumps({
            "command": command_string(),
            "experiment": experiment,
            "run_dir": str(proposed_run),
            "resume": args.resume,
            "resume_bundle": resume_preview,
            "config": requested_config,
            "config_provenance": provenance,
        }, indent=2, ensure_ascii=False, default=str))
        return 0

    if args.run_dir:
        run_dir = validate_run_dir(args.run_dir, experiment)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(experiment, args.batch_id)
    paths = prepare_run_dirs(run_dir)
    record_command(run_dir, "train")
    if not (run_dir / "git_state_before.json").exists():
        write_json(run_dir / "git_state_before.json", git_state(), overwrite=False)

    requested_path = run_dir / "requested_config.json"
    if requested_path.exists():
        stored_requested = read_json(requested_path)
        differences = _config_differences(stored_requested, requested_config)
        if differences:
            raise RuntimeError(
                "resume config differs from this run's immutable requested_config.json: "
                + json.dumps(differences, ensure_ascii=False, default=str)
            )
    else:
        write_json(requested_path, requested_config, overwrite=False)

    manifest_path = run_dir / "manifest.json"
    existing_manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if existing_manifest.get("stages", {}).get("train", {}).get("status") == "completed":
        if args.resume == "auto":
            print(f"TRAIN_ALREADY_COMPLETED='{run_dir}'")
            return 0
        raise RuntimeError("training is already completed; choose a new batch id")

    attempt = _next_attempt(existing_manifest)
    update_manifest(
        run_dir,
        schema_version="20260816",
        experiment=experiment,
        model_family=requested_config["model_family"],
        ode_type=requested_config["ode_type"],
        batch_id=run_dir.name,
        run_directory=str(run_dir),
        requested_config_path=str(requested_path),
        requested_config_sha256=config_fingerprint(requested_config),
        config_provenance=provenance,
    )

    resume_bundle = _resolve_resume(run_dir, requested_config, args.resume)
    segment_number, segment_dir = next_checkpoint_segment(run_dir)
    started_at = now_iso()
    attempt_summary = {
        "attempt": attempt,
        "segment": segment_number,
        "segment_directory": str(segment_dir),
        "started_at": started_at,
        "status": "running",
        "resume_bundle": resume_bundle,
    }
    _append_train_attempt(run_dir, attempt_summary)
    update_stage(
        run_dir,
        "train",
        "running",
        started_at=started_at,
        attempt=attempt,
        segment=segment_number,
        segment_directory=str(segment_dir),
        resume_bundle=resume_bundle,
    )

    config = copy.deepcopy(requested_config)
    config["resume_checkpoint"] = (
        resume_bundle["raw_checkpoint_path"] if resume_bundle is not None else ""
    )
    config["lr_anneal_steps"] = int(config.get("total_steps", config["lr_anneal_steps"]))
    config["total_steps"] = int(config["lr_anneal_steps"])
    segment_status_path = segment_dir / "segment_manifest.json"

    try:
        import scanpy as sc
        import torch

        from guided_diffusion import dist_util, logger
        from guided_diffusion.cell_datasets_loader import load_data
        from guided_diffusion.resample import create_named_schedule_sampler
        from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
        from guided_diffusion.train_util import TrainLoop

        _install_interrupt_handlers()
        _setup_seed(torch, int(config.get("seed", 1234)))
        device = _select_device(torch, dist_util, str(config.get("device", "auto")))
        logger.configure(dir=str(paths["logs"] / "train" / f"segment_{segment_number:03d}"))

        diffusion_args = model_and_diffusion_defaults()
        for key in diffusion_args:
            if key in config:
                diffusion_args[key] = config[key]
        _, diffusion = create_model_and_diffusion(**diffusion_args)

        logger.log(f"loading AnnData: {config['data_dir']}")
        adata = sc.read_h5ad(config["data_dir"])
        genes = _gene_list(adata)
        t_s_request = config.get("t_s")
        ts_info = resolve_ts(
            config,
            adata,
            diffusion,
            force_recompute=args.force_recompute_ts,
        )
        model = build_model_from_config(config, genes, diffusion.num_timesteps, device)
        model.to(device)
        model_info = _model_info(model)

        actual_config = copy.deepcopy(config)
        actual_config.update({
            "n_genes": len(genes),
            "device_resolved": str(device),
            "run_directory": str(run_dir),
            "checkpoint_segment": segment_number,
            "checkpoint_segment_directory": str(segment_dir),
            "t_s_request": t_s_request,
            "resolved_t_s": config.get("t_s"),
            "lambda_max": ts_info.get("lambda_max"),
            "alpha_bar_at_ts": ts_info.get("alpha_bar_at_ts"),
            "score_at_ts": ts_info.get("score_at_ts"),
            "ts_estimate": ts_info,
            "config_provenance": provenance,
            "requested_config_sha256": config_fingerprint(requested_config),
        })
        root_config_path = run_dir / "exp_config.json"
        if root_config_path.exists():
            stored_actual = read_json(root_config_path)
            # A resolved automatic T_s is immutable for every resume segment.
            if stored_actual.get("requested_config_sha256") != actual_config["requested_config_sha256"]:
                raise RuntimeError("stored exp_config.json does not match requested config fingerprint")
            if stored_actual.get("resolved_t_s") != actual_config.get("resolved_t_s"):
                raise RuntimeError("resolved T_s changed across resume attempts")
        else:
            write_json(root_config_path, actual_config, overwrite=False)
            write_json(run_dir / "model_info.json", model_info, overwrite=False)
            write_json(run_dir / "ts_estimate.json", ts_info, overwrite=False)
        write_json(segment_dir / "exp_config.json", actual_config, overwrite=False)
        write_json(segment_dir / "model_info.json", model_info, overwrite=False)
        write_json(segment_dir / "ts_estimate.json", ts_info, overwrite=False)
        write_json(segment_status_path, attempt_summary, overwrite=False)
        update_manifest(
            run_dir,
            config_path=str(root_config_path),
            model_info_path=str(run_dir / "model_info.json"),
            ts_estimate_path=str(run_dir / "ts_estimate.json"),
            t_s=config.get("t_s"),
            ts_estimate=ts_info,
        )

        data = load_data(
            data_dir=config["data_dir"],
            batch_size=int(config["batch_size"]),
            train_vae=True,
            preprocess=False,
            layer=config.get("ts_layer"),
        )
        schedule_sampler = create_named_schedule_sampler(config["schedule_sampler"], diffusion)
        model_name = str(config.get("model_name", "direct_hill_20260816"))
        if Path(model_name).name != model_name or not model_name:
            raise ConfigurationError(f"model_name must be one safe path component: {model_name!r}")

        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=int(config["batch_size"]),
            microbatch=int(config["microbatch"]),
            lr=float(config["lr"]),
            ema_rate=str(config["ema_rate"]),
            log_interval=int(config["log_interval"]),
            save_interval=int(config["save_interval"]),
            resume_checkpoint=str(config["resume_checkpoint"]),
            use_fp16=bool_value(config.get("use_fp16", False)),
            fp16_scale_growth=float(config.get("fp16_scale_growth", 1e-3)),
            schedule_sampler=schedule_sampler,
            weight_decay=float(config["weight_decay"]),
            lr_anneal_steps=int(config["lr_anneal_steps"]),
            model_name=model_name,
            save_dir=str(segment_dir),
            ode_reg_lambda=1.0,
            ode_reg_norm=str(config.get("ode_reg_norm", "l1")),
            save_loss_details=bool_value(config.get("save_loss_details", True)),
        ).run_loop()

        bundle = latest_checkpoint_bundle(run_dir, config["ema_rate"], require_complete=True)
        if bundle is None:
            raise RuntimeError("TrainLoop returned without a complete raw/EMA/optimizer bundle")
        bundle_path = Path(bundle["raw_checkpoint_path"])
        try:
            bundle_path.relative_to(segment_dir.resolve())
        except ValueError as exc:
            raise RuntimeError("current training segment produced no complete checkpoint bundle") from exc
        if int(bundle["step"]) < int(config["total_steps"]):
            raise RuntimeError(
                f"training stopped at checkpoint step {bundle['step']}, below target {config['total_steps']}"
            )
        metadata = _record_checkpoint(run_dir, bundle)
        finished_at = now_iso()
        completed_attempt = {
            **attempt_summary,
            "status": "completed",
            "finished_at": finished_at,
            "checkpoint": metadata,
        }
        write_json(segment_status_path, completed_attempt)
        manifest = read_json(run_dir / "manifest.json")
        histories = list(manifest.get("train_attempts", []))
        histories[-1] = completed_attempt
        update_manifest(run_dir, train_attempts=histories)
        update_stage(
            run_dir,
            "train",
            "completed",
            started_at=started_at,
            finished_at=finished_at,
            attempt=attempt,
            segment=segment_number,
            checkpoint_step=int(metadata["step"]),
            checkpoint_path=metadata["ema_checkpoint_path"],
            raw_checkpoint_path=metadata["raw_checkpoint_path"],
            ema_checkpoint_path=metadata["ema_checkpoint_path"],
            optimizer_checkpoint_path=metadata["optimizer_checkpoint_path"],
        )
        print(f"RUN_DIR='{run_dir}'")
        print(f"RAW_CHECKPOINT_PATH='{metadata['raw_checkpoint_path']}'")
        print(f"EMA_CHECKPOINT_PATH='{metadata['ema_checkpoint_path']}'")
        print(f"OPTIMIZER_CHECKPOINT_PATH='{metadata['optimizer_checkpoint_path']}'")
        print(f"CHECKPOINT_STEP={metadata['step']}")
        return 0
    except BaseException as exc:
        failed_at = now_iso()
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at": failed_at,
        }
        latest = latest_checkpoint_bundle(run_dir, config["ema_rate"], require_complete=True)
        if latest is not None:
            try:
                _record_checkpoint(run_dir, latest)
            except Exception:
                failure["checkpoint_record_error"] = traceback.format_exc()
        failed_attempt = {
            **attempt_summary,
            "status": "failed",
            "finished_at": failed_at,
            "failure": failure,
            "last_complete_checkpoint": latest,
        }
        if segment_status_path.exists():
            write_json(segment_status_path, failed_attempt)
        else:
            write_json(segment_status_path, failed_attempt, overwrite=False)
        manifest = read_json(run_dir / "manifest.json")
        histories = list(manifest.get("train_attempts", []))
        if histories:
            histories[-1] = failed_attempt
        else:
            histories.append(failed_attempt)
        update_manifest(run_dir, train_attempts=histories, last_failure=failure)
        update_stage(
            run_dir,
            "train",
            "failed",
            started_at=started_at,
            finished_at=failed_at,
            attempt=attempt,
            segment=segment_number,
            failure=failure,
            last_complete_checkpoint=latest,
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", "--run_dir", default="")
    parser.add_argument("--batch-id", default="")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default="",
        help="'auto' or a raw modelNNNNNN.pt path from this same run",
    )
    parser.add_argument("--set", dest="set_values", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--force-recompute-ts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_training(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
