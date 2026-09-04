#!/usr/bin/env python3
"""Generate cells with the custom dense reverse SDE and Appendix-I decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for search_path in (REPO_ROOT, SUITE_ROOT, HERE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from analysis.common import load_final_ema  # noqa: E402
from sampling.artifacts import save_sample_archive  # noqa: E402
from sampling.reverse_sde import sample_reverse_sde  # noqa: E402


def _slug_rate(value: str) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _close_backed_adata(adata) -> None:
    file_manager = getattr(adata, "file", None)
    if file_manager is not None:
        file_manager.close()


def sample_run(args: argparse.Namespace) -> dict:
    loaded = load_final_ema(
        args.run_dir,
        device=args.device,
        ema_rate=args.ema_rate,
    )
    config = loaded["config"]
    sample_count = int(
        config.get("sample_count", 2000)
        if args.sample_count is None
        else args.sample_count
    )
    batch_size = int(
        config.get("sample_batch_size", 128)
        if args.batch_size is None
        else args.batch_size
    )
    seed = int(
        config.get("sampling_seed", config.get("seed", 1234))
        if args.seed is None
        else args.seed
    )
    reverse_stride = int(
        config.get("reverse_stride", 1)
        if args.reverse_stride is None
        else args.reverse_stride
    )
    decoder_mode = str(
        config.get("decoder_sampling_mode", "sample")
        if args.decoder_sampling_mode is None
        else args.decoder_sampling_mode
    ).lower()
    if sample_count <= 0 or batch_size <= 0:
        raise ValueError("sample count and batch size must be positive")
    run = loaded["run_dir"]
    filename = (
        f"samples_ema_{_slug_rate(loaded['ema_rate'])}_"
        f"step{loaded['checkpoint_step']:06d}_seed{seed}.npz"
    )
    destination = run / "samples" / filename
    sidecar = destination.with_suffix(".json")
    requested = {
        "checkpoint_path": str(loaded["checkpoint"]),
        "checkpoint_step": int(loaded["checkpoint_step"]),
        "ema_rate": str(loaded["ema_rate"]),
        "forward_model": str(config["forward_model"]),
        "random_seed": seed,
        "sampler_type": "custom_reverse_sde_euler_maruyama",
        "reverse_stride": reverse_stride,
        "decoder_sampling_mode": decoder_mode,
        "sample_count": sample_count,
        "sample_batch_size": batch_size,
        "gene_order_sha256": loaded["gene_order_sha256"],
    }
    if destination.exists() or sidecar.exists():
        if not destination.is_file() or not sidecar.is_file():
            raise RuntimeError("partial sample artifact exists; refusing to overwrite")
        with sidecar.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if all(existing.get(key) == value for key, value in requested.items()):
            _close_backed_adata(loaded["adata"])
            return {
                "status": "skipped_completed",
                "sample_path": str(destination),
                "metadata_path": str(sidecar),
            }
        raise RuntimeError(
            "sample filename already exists with different provenance; "
            "completed artifacts are never overwritten"
        )
    if args.dry_run:
        _close_backed_adata(loaded["adata"])
        return {
            "status": "dry_run",
            "sample_path": str(destination),
            **requested,
        }

    model = loaded["components"].model
    time_map = loaded["components"].time_map
    generated = []
    reverse_steps = None
    for start in range(0, sample_count, batch_size):
        count = min(batch_size, sample_count - start)
        result = sample_reverse_sde(
            model,
            time_map,
            batch_size=count,
            seed=seed + start,
            reverse_stride=reverse_stride,
            decoder_sampling_mode=decoder_mode,
        )
        if reverse_steps is None:
            reverse_steps = result.reverse_steps
        elif reverse_steps != result.reverse_steps:
            raise RuntimeError("inconsistent reverse step count across batches")
        generated.append(result.samples.detach().cpu().float().numpy())
    cells = np.concatenate(generated, axis=0)
    metadata = {
        **requested,
        "reverse_steps": int(reverse_steps),
        "status": "completed",
        "batch_seed_policy": "batch seed = random_seed + first output row index",
    }
    archive, metadata_path = save_sample_archive(
        destination,
        cells,
        loaded["genes"],
        metadata,
    )
    _close_backed_adata(loaded["adata"])
    return {
        "status": "completed",
        "sample_path": str(archive),
        "metadata_path": str(metadata_path),
        "sample_count": sample_count,
        "reverse_steps": int(reverse_steps),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ema-rate", default="")
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reverse-stride", type=int, default=None)
    parser.add_argument(
        "--decoder-sampling-mode", choices=("sample", "mean"), default=None
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = sample_run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
