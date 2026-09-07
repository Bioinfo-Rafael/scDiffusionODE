#!/usr/bin/env python3
"""Create the 20260802-style independent-UMAP facets for three ODE-only runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
SHARED_SCRIPT = REPO_ROOT / "work/20260707_lincomb/integrated_analysis/run_all_models_facets_0707.py"
MODELS = (
    "ode_only_lincomb_softmax__softplus",
    "ode_only_lincomb_softmax__hill_after_softplus",
    "ode_only_lincomb_softmax__exp",
)

spec = importlib.util.spec_from_file_location("facets_20260707_shared", SHARED_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(SHARED_SCRIPT)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate(model: str, run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "exp_config.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    sample_status = manifest.get("stages", {}).get("sample", {}).get("status", "")
    sample_value = manifest.get("sample_path", "")
    sample_path = Path(sample_value) if sample_value else None
    if sample_path is None or not sample_path.is_file():
        samples = sorted(run_dir.glob("samples/*.npz"), key=lambda path: path.stat().st_mtime, reverse=True)
        sample_path = samples[0] if samples else None
    shape = None
    reason = ""
    if not config_path.is_file():
        reason = "missing exp_config.json"
    elif sample_status != "completed":
        reason = f"sample stage is not completed: {sample_status or 'missing'}"
    elif sample_path is None:
        reason = "missing generated sample"
    else:
        shape, shape_error = shared.npz_array_shape(sample_path)
        if shape_error:
            reason = shape_error
        elif len(shape) != 2:
            reason = f"cell_gen is not 2D: shape={shape}"
    checkpoint_step = int(manifest.get("checkpoint_step", 0) or 0)
    return {
        "model": model,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "sample_path": str(sample_path) if sample_path else "",
        "sample_manifest_path": "",
        "manifest_path": str(manifest_path) if manifest_path.is_file() else "",
        "exp_config_path": str(config_path) if config_path.is_file() else "",
        "available_generated_cells": int(shape[0]) if shape else 0,
        "generated_dim": int(shape[1]) if shape and len(shape) == 2 else None,
        "used_generated_cells": 0,
        "used_real_cells": 0,
        "checkpoint_step": checkpoint_step,
        "sample_mtime": sample_path.stat().st_mtime if sample_path else 0.0,
        "skip_reason": reason,
        "selected": not reason,
        "selection_status": "selected" if not reason else "skipped",
        "passed_over": [],
    }


def discover_records() -> list[dict]:
    records = []
    for model in MODELS:
        parent = SUITE_ROOT / "runs" / model
        candidates = [_candidate(model, run) for run in parent.iterdir() if run.is_dir()] if parent.is_dir() else []
        candidates.sort(key=lambda rec: (rec["checkpoint_step"], rec["sample_mtime"], rec["run_id"]), reverse=True)
        valid = [rec for rec in candidates if not rec["skip_reason"]]
        if valid:
            selected = valid[0]
            selected["passed_over"] = [rec for rec in candidates if rec is not selected]
            records.append(selected)
        else:
            records.append({
                "model": model, "run_id": "", "run_dir": "", "sample_path": "",
                "sample_manifest_path": "", "manifest_path": "", "exp_config_path": "",
                "available_generated_cells": 0, "generated_dim": None,
                "used_generated_cells": 0, "used_real_cells": 0,
                "checkpoint_step": 0, "sample_mtime": 0.0,
                "skip_reason": "no completed sampling run", "selected": False,
                "selection_status": "skipped", "passed_over": candidates,
            })
    return records


def data_path(records: list[dict]) -> Path:
    paths = []
    for record in records:
        if record.get("exp_config_path"):
            value = _read_json(Path(record["exp_config_path"]))["data_dir"]
            path = Path(value)
            paths.append(path if path.is_absolute() else REPO_ROOT / path)
    unique = {path.resolve() for path in paths}
    if len(unique) != 1:
        raise RuntimeError(f"the selected runs do not use one common dataset: {sorted(map(str, unique))}")
    return unique.pop()


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--per-model-real-cells", type=int, default=50000)
    parser.add_argument("--per-model-gen-cells", type=int, default=3000)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    records = discover_records()
    if args.dry_run:
        shared.print_dry_run(records)
        return 0
    missing = [record["model"] for record in records if record.get("skip_reason")]
    if missing:
        raise RuntimeError(f"completed samples are required for all three models: {missing}")
    runtime_args = argparse.Namespace(
        work_root=str(SUITE_ROOT), output_root=str(HERE / "outputs"),
        data_dir=str(data_path(records)),
        per_model_real_cells=args.per_model_real_cells,
        per_model_gen_cells=args.per_model_gen_cells,
        n_pcs=args.n_pcs, n_neighbors=args.n_neighbors,
        min_dist=args.min_dist, seed=args.seed, dry_run=False,
        model_order=list(MODELS),
    )
    shared.run_full(runtime_args, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
