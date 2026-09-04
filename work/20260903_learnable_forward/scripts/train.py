#!/usr/bin/env python3
"""Train one isolated dense learnable-forward experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for search_path in (REPO_ROOT, SUITE_ROOT, HERE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from common import (  # noqa: E402
    ConfigurationError,
    load_experiment_config,
    model_metadata,
    new_batch_id,
    next_segment_directory,
    resolve_resume_checkpoint,
    run_directory,
    validate_run_directory,
    write_json,
)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ConfigurationError(f"cannot interpret {value!r} as bool")


def _setup_seed(torch, numpy, seed: int, rank: int) -> int:
    """Seed each rank independently; DDP later synchronizes model parameters."""

    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    numpy.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
    return effective_seed


def _select_device(torch, dist_util, requested: str):
    choice = str(requested or "auto").lower()
    if choice in {"auto", "cuda"}:
        if choice == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is unavailable")
        dist_util.setup_dist()
        return dist_util.dev()
    if choice == "cpu":
        device = torch.device("cpu")
    elif choice == "mps":
        raise ConfigurationError(
            "device=mps is not supported by this exact dense experiment; "
            "the physical-time sampler and correctness path require float64 "
            "operations that MPS does not implement"
        )
    else:
        raise ConfigurationError("device must be one of auto, cpu, cuda, mps")
    # Existing TrainLoop moves microbatches through dist_util.dev(). Explicit
    # CPU runs are single-process and therefore use this work-local patch.
    dist_util.dev = lambda: device
    return device


def _configure_import_caches() -> None:
    """Keep third-party import caches writable and inside this work suite.

    Some Scanpy/Numba builds cannot cache functions beside a read-only or
    relocated site-package installation.  Setting an explicit cache before
    importing Scanpy avoids that environment-dependent import failure.  User
    overrides are respected, and all defaults remain below the ignored
    ``runs/`` directory.
    """

    cache_root = SUITE_ROOT / "runs" / ".runtime_cache"
    numba_cache = cache_root / "numba"
    matplotlib_cache = cache_root / "matplotlib"
    numba_cache.mkdir(parents=True, exist_ok=True)
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))


def _gene_list(adata, gene_column: str) -> list[str]:
    if gene_column not in adata.var.columns:
        raise ConfigurationError(
            f"AnnData .var must contain gene column {gene_column!r}"
        )
    genes = [str(value) for value in adata.var[gene_column].tolist()]
    if len(genes) != int(adata.n_vars):
        raise ConfigurationError("gene column must map one-to-one to variables")
    if len(set(genes)) != len(genes):
        raise ConfigurationError(
            "gene column contains duplicates; GRN orientation would be ambiguous"
        )
    return genes


def _select_run_path(config: Mapping[str, Any], args) -> Path:
    if args.run_dir:
        run_path = validate_run_directory(args.run_dir)
        if run_path.parent.name != str(config["experiment"]):
            raise ConfigurationError(
                "run-dir experiment component does not match config: "
                f"{run_path.parent.name!r} != {config['experiment']!r}"
            )
        return run_path
    return validate_run_directory(
        run_directory(config["experiment"], args.batch_id or new_batch_id())
    )


def _prepare_run(
    run_path: Path,
    config: Mapping[str, Any],
    provenance: Mapping[str, str],
    *,
    rank: int,
    dist_util,
) -> None:
    def prepare() -> None:
        run_path.mkdir(parents=True, exist_ok=True)
        requested_path = run_path / "requested_config.json"
        payload = {"config": dict(config), "provenance": dict(provenance)}
        if requested_path.exists():
            with requested_path.open(encoding="utf-8") as handle:
                stored = json.load(handle)
            if stored != payload:
                raise RuntimeError(
                    "run directory already contains a different requested_config.json"
                )
        else:
            write_json(requested_path, payload)

    _collective_rank_zero(
        prepare,
        rank=rank,
        dist_util=dist_util,
        description="prepare run directory",
    )


def _collective_rank_zero(action, *, rank: int, dist_util, description: str) -> None:
    """Run a filesystem action on rank zero without stranding DDP peers."""

    failure = None
    if rank == 0:
        try:
            action()
        except BaseException as exc:  # propagate even interrupts to waiting peers
            failure = f"{type(exc).__name__}: {exc}"

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        payload = [failure]
        dist.broadcast_object_list(payload, src=0)
        failure = payload[0]
    if failure is not None:
        raise RuntimeError(f"rank-zero action failed ({description}): {failure}")
    dist_util.barrier()


def _write_status(path: Path, payload: Mapping[str, Any], rank: int) -> None:
    if rank == 0:
        write_json(path, dict(payload))


def run_training(args: argparse.Namespace) -> int:
    config, provenance = load_experiment_config(
        args.config,
        base_path=args.base_config,
        set_values=args.set_values,
    )
    # Keep dry-run honest: reject unsupported training policies before any run
    # directory or heavyweight Scanpy import is touched.
    from models.factory import validate_training_policy

    validate_training_policy(config)
    if str(config.get("forward_dtype", "float32")).lower() != "float32":
        raise ConfigurationError(
            "the unchanged legacy TrainLoop requires float32 learnable "
            "parameters: its MixedPrecisionTrainer norm calculation rejects "
            "float64 parameters. Use forward_dtype=float32 for train.py; "
            "float64 remains available in correctness tests and the direct "
            "dense benchmark/evaluation code."
        )
    if str(config.get("device", "auto")).lower() == "mps":
        raise ConfigurationError(
            "device=mps is not supported by this exact dense experiment"
        )
    run_path = _select_run_path(config, args)
    resume_request = args.resume or str(config.get("resume_checkpoint", ""))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": config,
                    "provenance": provenance,
                    "run_directory": str(run_path),
                    "resume": resume_request,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    _configure_import_caches()
    import numpy as np
    import scanpy as sc
    import torch

    from guided_diffusion import dist_util, logger
    from guided_diffusion.cell_datasets_loader import load_data
    from guided_diffusion.train_util import TrainLoop
    from models.factory import build_experiment_components

    device = _select_device(torch, dist_util, str(config.get("device", "auto")))
    rank = dist_util.get_rank()
    effective_seed = _setup_seed(torch, np, int(config.get("seed", 1234)), rank)
    _prepare_run(run_path, config, provenance, rank=rank, dist_util=dist_util)
    resume_checkpoint = resolve_resume_checkpoint(run_path, resume_request)

    segment_path = next_segment_directory(run_path)
    _collective_rank_zero(
        lambda: segment_path.mkdir(parents=True, exist_ok=False),
        rank=rank,
        dist_util=dist_util,
        description="create training segment",
    )
    logger.configure(dir=str(segment_path / "logs"))
    status_path = segment_path / "status.json"
    started_at = datetime.now().astimezone().isoformat()
    _write_status(
        status_path,
        {
            "status": "running",
            "started_at": started_at,
            "rank_zero_seed": int(config.get("seed", 1234)),
            "local_seed": effective_seed,
            "resume_checkpoint": resume_checkpoint,
        },
        rank,
    )

    try:
        data_path = Path(str(config["data_dir"]))
        if not data_path.is_file():
            raise FileNotFoundError(f"AnnData input does not exist: {data_path}")
        logger.log(f"loading AnnData metadata: {data_path}")
        adata = sc.read_h5ad(data_path, backed="r")
        try:
            genes = _gene_list(
                adata, str(config.get("gene_column", "gene_name"))
            )
        finally:
            file_manager = getattr(adata, "file", None)
            if file_manager is not None:
                file_manager.close()

        components = build_experiment_components(config, genes, device)
        metadata = model_metadata(components.model, config)
        metadata.update(
            {
                "gene_count": len(genes),
                "device": str(device),
                "time_map": components.time_map.metadata(),
            }
        )
        def write_metadata() -> None:
            write_json(run_path / "model_info.json", metadata)
            write_json(segment_path / "resolved_config.json", dict(config))

        _collective_rank_zero(
            write_metadata,
            rank=rank,
            dist_util=dist_util,
            description="write model metadata",
        )

        data_layer = config.get("data_layer", config.get("layer"))
        data = load_data(
            data_dir=str(data_path),
            batch_size=int(config["batch_size"]),
            deterministic=_bool(config.get("deterministic_data", False)),
            train_vae=True,
            preprocess=False,
            layer=data_layer,
        )
        total_steps = int(
            config.get("lr_anneal_steps", config.get("total_steps", 0))
        )
        logger.log(
            "starting dense learnable-forward training: "
            f"experiment={config['experiment']}, "
            f"forward_model={config['forward_model']}, "
            f"loss_mode={config.get('loss_mode', 'paper_elbo')}, "
            f"genes={len(genes)}, device={device}"
        )
        TrainLoop(
            model=components.model,
            diffusion=components.diffusion,
            data=data,
            batch_size=int(config["batch_size"]),
            microbatch=int(config.get("microbatch", -1)),
            lr=float(config.get("lr", 1e-4)),
            ema_rate=str(config.get("ema_rate", "0.9999")),
            log_interval=int(config.get("log_interval", 10)),
            save_interval=int(config.get("save_interval", 10000)),
            resume_checkpoint=resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=float(config.get("fp16_scale_growth", 1e-3)),
            schedule_sampler=components.schedule_sampler,
            weight_decay=float(config.get("weight_decay", 0.0)),
            lr_anneal_steps=total_steps,
            model_name="model",
            save_dir=str(segment_path),
            # Forward-specific regularization is already returned inside the
            # wrapper loss. The legacy ODE hook must remain inert.
            ode_reg_lambda=0.0,
            ode_reg_norm="l1",
            # The core CSV schema is ODE-specific and cannot represent the
            # new decomposed ELBO. Generic logger keys remain enabled.
            save_loss_details=False,
        ).run_loop()
        finished_at = datetime.now().astimezone().isoformat()
        _write_status(
            status_path,
            {
                "status": "completed",
                "started_at": started_at,
                "finished_at": finished_at,
                "resume_checkpoint": resume_checkpoint,
            },
            rank,
        )
        if rank == 0:
            print(f"RUN_DIR='{run_path}'")
            print(f"SEGMENT_DIR='{segment_path}'")
        return 0
    except BaseException as exc:
        _write_status(
            status_path,
            {
                "status": "failed",
                "started_at": started_at,
                "failed_at": datetime.now().astimezone().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "resume_checkpoint": resume_checkpoint,
            },
            rank,
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--batch-id", default="")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default="",
        help="resume from the latest complete in-run checkpoint or an explicit raw path",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_training(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
