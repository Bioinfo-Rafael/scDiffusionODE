#!/usr/bin/env python3
"""Strictly restore one completed run and generate suite-local samples."""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for search_path in (REPO_ROOT, SUITE_ROOT, HERE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from common import (  # noqa: E402
    artifact_dirs,
    command_string,
    latest_checkpoint_bundle,
    now_iso,
    read_json,
    record_command,
    update_manifest,
    update_stage,
    validate_run_dir,
    write_json,
)


def _select_device(torch: Any, name: str) -> Any:
    choice = str(name).lower()
    if choice == "auto":
        if torch.cuda.is_available():
            choice = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            choice = "mps"
        else:
            choice = "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is unavailable")
    if choice == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("device=mps requested but MPS is unavailable")
    return torch.device(choice)


def _assert_finite(torch: Any, name: str, value: Any) -> None:
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains NaN or Inf")
    elif not bool(np.isfinite(np.asarray(value)).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _genes(config: Mapping[str, Any]) -> list[str]:
    import scanpy as sc

    adata = sc.read_h5ad(str(config["data_dir"]), backed="r")
    try:
        if "gene_name" not in adata.var.columns:
            raise KeyError("AnnData .var must contain gene_name")
        genes = [str(value) for value in adata.var["gene_name"].tolist()]
        if len(genes) != int(adata.n_vars) or len(set(genes)) != len(genes):
            raise ValueError("gene_name must be a unique one-to-one variable mapping")
        return genes
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def _choose_checkpoint(run_dir: Path, config: Mapping[str, Any], requested: str) -> Path:
    if requested:
        path = Path(requested).expanduser().resolve()
    else:
        manifest = read_json(run_dir / "manifest.json")
        value = manifest.get("ema_checkpoint_path") or manifest.get("checkpoint_path")
        if value:
            path = Path(str(value)).expanduser().resolve()
        else:
            bundle = latest_checkpoint_bundle(run_dir, config["ema_rate"], require_complete=True)
            if bundle is None:
                raise FileNotFoundError("no complete checkpoint bundle found")
            path = Path(bundle["ema_checkpoint_path"])
    checkpoint_root = (run_dir / "train" / "checkpoints").resolve()
    try:
        path.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ValueError(f"checkpoint must belong to this run: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_sampling(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from guided_diffusion import dist_util, logger
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
    from ODE.ode_20260609_mathmlp import clean_state_dict
    from models import build_model_from_config

    run_dir = validate_run_dir(args.run_dir)
    config = read_json(run_dir / "exp_config.json")
    checkpoint = _choose_checkpoint(run_dir, config, args.model_path)
    paths = artifact_dirs(run_dir)
    num_samples = int(args.num_samples or config.get("num_samples", 3000))
    batch_size = int(args.batch_size or config.get("sample_batch_size", 50))
    if num_samples <= 0 or batch_size <= 0:
        raise ValueError("num-samples and batch-size must be positive")
    preview = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "num_samples": num_samples,
        "batch_size": batch_size,
        "output_directory": str(paths["samples"]),
        "command": command_string(),
    }
    if args.dry_run:
        return {"dry_run": True, **preview}

    record_command(run_dir, "sample")
    started_at = now_iso()
    update_stage(
        run_dir,
        "sample",
        "running",
        started_at=started_at,
        checkpoint=preview["checkpoint"],
        num_samples=num_samples,
        batch_size=batch_size,
        output_directory=preview["output_directory"],
        command=preview["command"],
    )
    try:
        seed = int(args.seed if args.seed is not None else config.get("seed", 1234))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        device = _select_device(torch, args.device)
        logger.configure(dir=str(paths["logs"] / "sample"))

        diffusion_args = model_and_diffusion_defaults()
        for key in diffusion_args:
            if key in config:
                diffusion_args[key] = config[key]
        _, diffusion = create_model_and_diffusion(**diffusion_args)
        genes = _genes(config)
        model = build_model_from_config(config, genes, diffusion.num_timesteps, device)
        state = clean_state_dict(dist_util.load_state_dict(str(checkpoint), map_location="cpu"))
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint did not contain a state_dict mapping")
        for name, tensor in state.items():
            _assert_finite(torch, f"checkpoint tensor {name}", tensor)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()

        use_ddim = bool(config.get("use_ddim", False))
        sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
        chunks: list[np.ndarray] = []
        generated = 0
        with torch.no_grad():
            while generated < num_samples:
                count = min(batch_size, num_samples - generated)
                kwargs: dict[str, Any] = {
                    "clip_denoised": bool(config.get("clip_denoised", False)),
                }
                if not use_ddim:
                    kwargs["start_time"] = diffusion.num_timesteps
                sample, _ = sample_fn(model, (count, len(genes)), **kwargs)
                _assert_finite(torch, "generated batch", sample)
                chunks.append(sample.detach().cpu().numpy().astype(np.float32, copy=False))
                generated += count
                logger.log(f"Generated {generated}/{num_samples}")
        array = np.concatenate(chunks, axis=0)
        _assert_finite(torch, "generated samples", array)

        paths["samples"].mkdir(parents=True, exist_ok=True)
        stem = f"samples_{checkpoint.stem}"
        destination = paths["samples"] / f"{stem}.npz"
        suffix = 1
        while destination.exists():
            destination = paths["samples"] / f"{stem}_{suffix:03d}.npz"
            suffix += 1
        np.savez_compressed(destination, cell_gen=array)
        sample_manifest = {
            **preview,
            "status": "completed",
            "finished_at": now_iso(),
            "sample_path": str(destination),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "seed": seed,
            "device": str(device),
            "use_ddim": use_ddim,
            "clip_denoised": bool(config.get("clip_denoised", False)),
            "diffusion_steps": int(diffusion.num_timesteps),
            "t_s": config.get("t_s"),
            "gate_tau": config.get("gate_tau"),
        }
        manifest_path = destination.with_suffix(".manifest.json")
        write_json(manifest_path, sample_manifest, overwrite=False)
        update_manifest(
            run_dir,
            sample_path=str(destination),
            sample_manifest_path=str(manifest_path),
        )
        update_stage(
            run_dir,
            "sample",
            "completed",
            started_at=started_at,
            finished_at=sample_manifest["finished_at"],
            checkpoint=str(checkpoint),
            sample_path=str(destination),
            sample_manifest_path=str(manifest_path),
            num_samples=num_samples,
        )
        return sample_manifest
    except BaseException as exc:
        update_stage(
            run_dir,
            "sample",
            "failed",
            started_at=started_at,
            failed_at=now_iso(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-dir", "--run_dir", required=True)
    parser.add_argument("--model-path", "--model_path", default="")
    parser.add_argument("--num-samples", "--num_samples", type=int, default=0)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run_sampling(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if not result.get("dry_run"):
        print(f"SAMPLE_FILE_PATH='{result['sample_path']}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
