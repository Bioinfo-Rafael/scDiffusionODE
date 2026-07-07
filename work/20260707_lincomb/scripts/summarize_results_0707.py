#!/usr/bin/env python3
"""Collect run manifests into comparison CSV/JSON files."""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import read_json, write_json


def main():
    rows = []
    for path in sorted((WORK_ROOT / "runs").glob("*/*/manifest.json")):
        manifest = read_json(path)
        gate = manifest.get("gate_diagnostics", {})
        rows.append({
            "experiment": manifest.get("experiment"),
            "run_dir": manifest.get("run_directory", manifest.get("run_dir")),
            "checkpoint_path": manifest.get("checkpoint_path"),
            "sample_path": manifest.get("sample_path"),
            "visualization_path": manifest.get("visualization_path"),
            "reverse_coef": manifest.get("reverse_coef", False),
            "regime_gate_mode": manifest.get("regime_gate_mode", "none"),
            "regime_gate_type": manifest.get("regime_gate_type", "sigmoid"),
            "lambda_max": manifest.get("lambda_max"),
            "t_s": manifest.get("t_s"),
            "alpha_bar_at_ts": manifest.get("alpha_bar_at_ts"),
            "score_at_ts": manifest.get("score_at_ts"),
            "gate_tau": manifest.get("gate_tau"),
            "w_ode_at_t0": gate.get("w_ode_at_t0"),
            "w_ode_at_ts": gate.get("w_ode_at_ts"),
            "w_ode_at_tmax": gate.get("w_ode_at_tmax"),
            "baseline_run_dir": (manifest.get("baseline_reference") or {}).get("run_dir"),
            "baseline_checkpoint_path": (manifest.get("baseline_reference") or {}).get("checkpoint_path"),
            "baseline_config_path": (manifest.get("baseline_reference") or {}).get("config_path"),
        })
    output = WORK_ROOT / "results"
    output.mkdir(exist_ok=True)
    write_json(output / "comparison_summary.json", rows)
    if rows:
        with open(output / "comparison_summary.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"summarized {len(rows)} run(s) -> {output}")


if __name__ == "__main__":
    main()
