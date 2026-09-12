#!/usr/bin/env python3
"""Restore one source EMA in a fresh process; optionally run Stage 1."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import ROOT, WORK

from src.artifacts import result_path, write_json
from src.model_runs import restore, select_run
from src.settings import PAPER, Settings
from src.source_imports import SUITES, activate_suite, umap_core
from src.trajectory import sample_to_disk, snapshot_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--suite", choices=list(SUITES), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--edge-path", type=Path)
    parser.add_argument("--no-save-states", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--restore-only",
        action="store_true",
        help="Strict real-checkpoint restore only, no sampling",
    )
    mode.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate run/config/checkpoint selection without loading model",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def sampling_metadata(
    args,
    selection,
    original_config: dict,
    genes: list[str],
    diffusion,
    device,
    seed: int,
    stamp: str,
) -> dict:
    import numpy as np
    import torch

    common, _ = activate_suite(selection.suite)
    settings = Settings()
    core = umap_core()
    return {
        "status": "sampling",
        "source_suite": args.suite,
        "source_suite_directory": SUITES[args.suite],
        "experiment": selection.experiment,
        "source_run_dir": str(selection.run_dir),
        "source_exp_config": str(selection.run_dir / "exp_config.json"),
        "source_config": original_config,
        "effective_config": selection.config,
        "source_exp_config_sha256": common.file_sha256(selection.run_dir / "exp_config.json"),
        "checkpoint_path": str(selection.checkpoint),
        "checkpoint_step": selection.step,
        "ema_rate": selection.ema_rate,
        "checkpoint_sha256": common.file_sha256(selection.checkpoint),
        "data_path": selection.config["data_dir"],
        "edge_path": selection.config["edge_tsv_path"],
        "edge_sha256": common.file_sha256(selection.config["edge_tsv_path"]),
        "gene_count": len(genes),
        "gene_names": genes,
        "gene_order_hash": core.gene_order_hash(genes),
        "settings": asdict(settings),
        "selected_reverse_steps": [
            row["reverse_step"] for row in snapshot_table(settings, diffusion.timestep_map)
        ],
        "actual_diffusion_timesteps": [
            row["diffusion_t"] for row in snapshot_table(settings, diffusion.timestep_map)
        ],
        "sampling_batch_size": args.batch_size,
        "seed": seed,
        "device": str(device),
        "snapshot_table": snapshot_table(settings, diffusion.timestep_map),
        "paper": PAPER,
        "source_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "created_at": stamp,
        "rng_policy": "seed once after process imports, before model restoration; batching affects RNG consumption; reuse batch size to reproduce",
        "command": sys.argv,
    }


def main() -> None:
    args = parse_args()
    run = args.run_dir.expanduser().resolve()
    checkpoint = str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else ""
    selection = select_run(args.suite, run, checkpoint)
    original_config = dict(selection.config)
    for key, path in (("data_dir", args.data_path), ("edge_tsv_path", args.edge_path)):
        if path is not None:
            selection.config[key] = str(path.expanduser().resolve())
        if not Path(selection.config[key]).is_file():
            raise FileNotFoundError(
                f"{key} not found: {selection.config[key]}; supply explicit relocated path"
            )
    if args.inspect_only:
        print(
            json.dumps(
                {
                    "status": "selection_validated",
                    "suite": args.suite,
                    "experiment": selection.experiment,
                    "checkpoint": str(selection.checkpoint),
                    "checkpoint_step": selection.step,
                }
            )
        )
        return
    import numpy as np
    import torch

    seed = args.seed if args.seed is not None else int(selection.config.get("seed", 1234))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, diffusion, genes, device = restore(selection, args.device)
    if args.restore_only:
        print(
            json.dumps(
                {
                    "status": "restored",
                    "suite": args.suite,
                    "experiment": selection.experiment,
                    "checkpoint": str(selection.checkpoint),
                    "checkpoint_step": selection.step,
                    "genes": len(genes),
                    "diffusion_steps": diffusion.num_timesteps,
                    "device": str(device),
                }
            )
        )
        return
    settings = Settings()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    result = result_path(
        args.result_dir
        or WORK / "results" / args.suite / selection.experiment / f"{run.name}_{stamp}"
    )
    result.mkdir(parents=True, exist_ok=False)
    metadata = sampling_metadata(
        args, selection, original_config, genes, diffusion, device, seed, stamp
    )
    write_json(result / "metadata.json", metadata)
    try:
        sampling = sample_to_disk(
            model,
            diffusion,
            result / "trajectory",
            settings,
            gene_count=len(genes),
            batch_size=args.batch_size,
            device=device,
            clip_denoised=bool(selection.config.get("clip_denoised", False)),
            save_states=not args.no_save_states,
        )
        metadata.update(
            status="trajectory_completed",
            final_pred_sample_relation=sampling["final_relation"],
            final_pred_sample_max_abs_error=sampling["final_pred_sample_max_abs_error"],
        )
        write_json(result / "metadata.json", metadata)
        del model, diffusion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for script in ("analyze_breakpoint.py", "plot_independent_umaps.py"):
            subprocess.run(
                [sys.executable, str(WORK / "scripts" / script), "--result-dir", str(result)],
                cwd=ROOT,
                check=True,
            )
        metadata["status"] = "stage1_completed"
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(result / "metadata.json", metadata)
    print(f"RESULT_DIR={result}", flush=True)


if __name__ == "__main__":
    main()
