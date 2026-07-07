#!/usr/bin/env python3
"""Run legacy-compatible and 0707-specific visualizations for one run."""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
REPO_ROOT = WORK_ROOT.parent.parent
LEGACY_VIZ = REPO_ROOT / "work" / "20260609_Hybrid5x3" / "viz"
for path in (str(REPO_ROOT), str(WORK_ROOT), str(WORK_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.common import artifact_dirs, command_string, read_json, update_manifest  # noqa: E402


def _run(label, command, keep_going):
    print(f"\n### {label}\n{' '.join(str(v) for v in command)}", flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode and not keep_going:
        raise SystemExit(result.returncode)
    return result.returncode


def _find_one(pattern):
    matches = glob.glob(str(pattern), recursive=True)
    return max(matches, key=lambda value: Path(value).stat().st_mtime) if matches else ""


def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "exp_config.json"
    config = read_json(config_path)
    manifest = read_json(run_dir / "manifest.json")
    paths = artifact_dirs(run_dir)
    checkpoint = args.model_path or manifest.get("checkpoint_path", "")
    sample_path = manifest.get("sample_path", "")
    loss_path = _find_one(paths["train"] / "**" / "loss_details.csv")
    output_dir = paths["viz"]
    skip = {value.strip() for value in args.skip.split(",") if value.strip()}

    legacy_skip = {"velocity"}
    if not loss_path:
        legacy_skip.add("loss")
    commands = []
    if "legacy" not in skip:
        commands.append(("legacy loss/params/eval", [
            sys.executable, LEGACY_VIZ / "run_all_viz.py",
            "--model_path", checkpoint, "--config", config_path,
            "--sample_path", sample_path, "--loss_path", loss_path,
            "--data_dir", config["data_dir"], "--edge_tsv_path", config["edge_tsv_path"],
            "--output_dir", output_dir, "--skip", ",".join(sorted(legacy_skip)),
            "--max_cells", str(args.max_cells),
        ]))
    if "gate" not in skip:
        commands.append(("gate diagnostics", [
            sys.executable, HERE / "plot_gate_diagnostics_0707.py",
            "--run-dir", run_dir, "--max-cells", str(args.diagnostic_max_cells),
        ]))
    if "velocity" not in skip:
        commands.append(("integrated velocity", [
            sys.executable, HERE / "run_velocity_suite_0707.py",
            "--run-dir", run_dir, "--max-cells", str(args.max_cells),
            "--n-jobs", str(args.n_jobs),
        ]))
    if "a_embedding" not in skip:
        commands.append(("LinComb a embedding", [
            sys.executable, LEGACY_VIZ / "plot_lincomb_a_embedding.py",
            "--run_dir", run_dir, "--model_path", checkpoint,
            "--config", config_path, "--data_dir", config["data_dir"],
            "--edge_tsv_path", config["edge_tsv_path"],
            "--output_dir", output_dir / "lincomb_a_embedding",
            "--max_cells", str(args.diagnostic_max_cells),
        ]))
    if "superclass" not in skip:
        commands.append(("LinComb Superclass coefficients", [
            sys.executable, LEGACY_VIZ / "plot_lincomb_superclass_coefficients.py",
            "--run_dir", run_dir, "--model_path", checkpoint,
            "--config", config_path, "--data_dir", config["data_dir"],
            "--edge_tsv_path", config["edge_tsv_path"],
            "--output_dir", output_dir,
            "--max_cells", str(args.diagnostic_max_cells),
        ]))

    if args.dry_run:
        print(json.dumps({
            "command": command_string(), "output_dir": str(output_dir),
            "commands": [[str(value) for value in command] for _, command in commands],
        }, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "commands" / "viz.txt").write_text(command_string() + "\n", encoding="utf-8")
    return_codes = {}
    for label, command in commands:
        return_codes[label] = _run(label, command, args.keep_going)
    stages = manifest.get("stages", {})
    stages["viz"] = {
        "status": "completed" if not any(return_codes.values()) else "partial",
        "finished_at": datetime.now().isoformat(), "output_dir": str(output_dir),
        "return_codes": return_codes,
    }
    update_manifest(run_dir, stages=stages, visualization_path=str(output_dir))
    if any(return_codes.values()):
        raise SystemExit(1)
    print(f"VISUALIZATION_DIR='{output_dir}'")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--diagnostic-max-cells", type=int, default=50000)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--skip", default="")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    main()

