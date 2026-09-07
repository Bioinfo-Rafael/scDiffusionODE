#!/usr/bin/env python3
"""Train and analyze the twelve 20260802 norm-control conditions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
PIPELINE = REPO_ROOT / "work" / "20260801" / "scripts" / "run_pipeline_20260801.py"
SUMMARY = REPO_ROOT / "work" / "20260801" / "scripts" / "summarize_results_20260801.py"
INTEGRATED_UMAP = SUITE_ROOT / "integrated_analysis" / "run_integrated_umap_20260802.py"
CONFIG_NAMES = (
    "hybrid_softmax_gate_ratio_reg.json",
    "hybrid_softmax_gate_normed_learned_scale.json",
    "hybrid_softmax_gate_scale_model_x.json",
    "hybrid_softmax_gate_scale_model_ml_emb.json",
    "hybrid_ts_soft_softmax_gate_ratio_reg.json",
    "hybrid_ts_soft_softmax_gate_normed_learned_scale.json",
    "hybrid_ts_soft_softmax_gate_scale_model_x.json",
    "hybrid_ts_soft_softmax_gate_scale_model_ml_emb.json",
    "hybrid_ts_soft_gentle_tau80_softmax_gate_ratio_reg.json",
    "hybrid_ts_soft_gentle_tau80_softmax_gate_normed_learned_scale.json",
    "hybrid_ts_soft_gentle_tau80_softmax_gate_scale_model_x.json",
    "hybrid_ts_soft_gentle_tau80_softmax_gate_scale_model_ml_emb.json",
)


def _config_name(value: str) -> str:
    name = Path(value).name
    if name != value:
        raise ValueError(f"--only accepts config names, not paths: {value}")
    return name if name.endswith(".json") else f"{name}.json"


def _selected_configs(values: list[str]) -> tuple[str, ...]:
    if not values:
        return CONFIG_NAMES
    requested = {_config_name(value) for value in values}
    unknown = requested.difference(CONFIG_NAMES)
    if unknown:
        raise ValueError(f"unknown config(s): {', '.join(sorted(unknown))}")
    return tuple(name for name in CONFIG_NAMES if name in requested)


def _experiment_name(config_name: str) -> str:
    return config_name.removesuffix(".json")


def _run(label: str, command: list[object]) -> None:
    print(f"\n### {label}\n{' '.join(str(value) for value in command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-integrated-umap", action="store_true")
    parser.add_argument("--only", nargs="+", default=[], metavar="CONFIG")
    args, extra = parser.parse_known_args()
    try:
        config_names = _selected_configs(args.only)
    except ValueError as exc:
        parser.error(str(exc))
    batch_id = args.batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    if Path(batch_id).name != batch_id:
        parser.error("--batch-id must be a single path component")

    for config_name in config_names:
        experiment = _experiment_name(config_name)
        command = [
            sys.executable, PIPELINE,
            "--config", SUITE_ROOT / "configs" / config_name,
            "--run-dir", SUITE_ROOT / "runs" / experiment / batch_id,
        ]
        if args.dry_run:
            command.append("--dry-run")
        if args.skip_existing:
            command.append("--skip-existing")
        command.extend(extra)
        _run(config_name, command)

    if args.dry_run:
        return
    _run("comparison summary", [
        sys.executable, SUMMARY, "--suite-root", SUITE_ROOT,
    ])
    if not args.skip_integrated_umap:
        _run("integrated UMAP", [sys.executable, INTEGRATED_UMAP])


if __name__ == "__main__":
    main()
