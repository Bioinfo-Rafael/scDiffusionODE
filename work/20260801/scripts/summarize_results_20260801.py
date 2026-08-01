#!/usr/bin/env python3
"""Collect 20260801 manifests into the standard comparison summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent


def read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    rows = []
    for path in sorted((SUITE_ROOT / "runs").glob("*/*/manifest.json")):
        manifest = read_json(path)
        gate = manifest.get("gate_diagnostics", {})
        baseline = manifest.get("baseline_reference", {})
        rows.append({
            "experiment": manifest.get("experiment"),
            "run_dir": manifest.get("run_directory"),
            "checkpoint_path": manifest.get("checkpoint_path"),
            "sample_path": manifest.get("sample_path"),
            "visualization_path": manifest.get("visualization_path"),
            "regime_gate_mode": manifest.get("regime_gate_mode", "none"),
            "regime_gate_type": manifest.get("regime_gate_type", "sigmoid"),
            "lambda_max": manifest.get("lambda_max"),
            "t_s": manifest.get("t_s"),
            "gate_tau": manifest.get("gate_tau"),
            "w_ode_at_t0": gate.get("w_ode_at_t0"),
            "w_ode_at_ts": gate.get("w_ode_at_ts"),
            "w_ode_at_tmax": gate.get("w_ode_at_tmax"),
            "baseline_run_dir": baseline.get("run_dir"),
            "baseline_checkpoint_path": baseline.get("checkpoint_path"),
            "baseline_sample_path": baseline.get("sample_path"),
        })
    output_dir = SUITE_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    if rows:
        with open(output_dir / "comparison_summary.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"summarized {len(rows)} run(s) -> {output_dir}")


if __name__ == "__main__":
    main()
