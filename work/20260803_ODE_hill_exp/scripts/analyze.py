#!/usr/bin/env python3
"""Restore one run and create all applicable analysis/visualization artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for path in (str(REPO_ROOT), str(SUITE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from viz.analysis_helpers import analyze_run  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def _mark_stage(run_dir: Path, status: str, **details) -> None:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = {}
    stages = dict(manifest.get("stages", {}))
    stages["analysis"] = {"status": status, **details}
    manifest["stages"] = stages
    _write_json(manifest_path, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one 20260803 ODE experiment run. Non-finite inputs or outputs "
            "fail preflight and are never silently replaced."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--run-dir", "--run_dir", required=True)
    parser.add_argument(
        "--checkpoint", "--model-path", "--model_path", default="",
        help="checkpoint override; default restores manifest checkpoint_path (EMA)",
    )
    parser.add_argument(
        "--sample-path", "--sample_path", default="",
        help="samples.npz override; default restores manifest sample_path",
    )
    parser.add_argument(
        "--max-cells", "--max_cells", type=int, default=2000,
        help="maximum cells from each of real/generated data; <=0 means all",
    )
    parser.add_argument(
        "--t-values", "--t_values", default="",
        help="comma-separated raw diffusion timesteps, e.g. 0,499,999",
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_cells == 1 or args.max_cells == 2:
        raise ValueError("--max-cells must be <=0 (all) or >=3")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not args.dry_run:
        _mark_stage(
            run_dir, "running",
            started_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            command=" ".join(sys.argv),
        )
    try:
        result = analyze_run(
            run_dir,
            checkpoint=args.checkpoint,
            sample_path=args.sample_path,
            max_cells=args.max_cells,
            t_values=args.t_values,
            batch_size=args.batch_size,
            device_name=args.device,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        if not args.dry_run:
            _mark_stage(
                run_dir, "failed",
                failed_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if not args.dry_run:
        print(f"ANALYSIS_DIR='{result['output_dir']}'")


if __name__ == "__main__":
    main()
