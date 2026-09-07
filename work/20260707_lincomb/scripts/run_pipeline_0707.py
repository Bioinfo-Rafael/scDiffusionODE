#!/usr/bin/env python3
"""Run training, sampling and visualization for one 0707 config."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
REPO_ROOT = WORK_ROOT.parent.parent
for path in (str(REPO_ROOT), str(WORK_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.common import command_string, create_run_dir, load_experiment_config, read_json  # noqa: E402


def _run(label, command):
    print(f"\n### {label}\n{' '.join(str(v) for v in command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main():
    args, extra = build_parser().parse_known_args()
    config = load_experiment_config(args.config)
    if args.dry_run:
        run_dir = Path(args.run_dir) if args.run_dir else WORK_ROOT / "runs" / config["experiment"] / "<timestamp>"
        commands = [
            [sys.executable, HERE / "train_0707.py", "--config", args.config, "--run-dir", run_dir, "--dry-run", *extra],
            [sys.executable, HERE / "sample_0707.py", "--run-dir", run_dir, "--dry-run"],
            [sys.executable, WORK_ROOT / "viz" / "run_all_viz_0707.py", "--run-dir", run_dir, "--dry-run"],
        ]
        print(json.dumps({
            "pipeline_command": command_string(), "run_dir": str(run_dir),
            "commands": [[str(v) for v in command] for command in commands],
        }, indent=2))
        return

    run_dir = Path(args.run_dir).resolve() if args.run_dir else create_run_dir(config["experiment"], args.name)
    (run_dir / "commands").mkdir(parents=True, exist_ok=True)
    (run_dir / "commands" / "pipeline.txt").write_text(command_string() + "\n", encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    def should_run(stage):
        if args.force_stage and args.force_stage != stage:
            return False
        if args.skip_existing and manifest.get("stages", {}).get(stage, {}).get("status") == "completed":
            print(f"[pipeline] skip existing completed stage: {stage}")
            return False
        return True

    train_command = [
        sys.executable, HERE / "train_0707.py", "--config", args.config,
        "--run-dir", run_dir, *extra,
    ]
    if args.resume:
        resume_path = args.resume
        if resume_path == "auto":
            resume_path = manifest.get("checkpoint_path", "")
        if not resume_path:
            raise ValueError("--resume auto requested but manifest has no checkpoint_path")
        train_command += ["--set", f"resume_checkpoint={resume_path}"]
    if should_run("train"):
        _run("train", train_command)
    if not args.skip_sample and should_run("sample"):
        _run("sample", [sys.executable, HERE / "sample_0707.py", "--run-dir", run_dir])
    if not args.skip_viz and should_run("viz"):
        _run("visualization", [
            sys.executable, WORK_ROOT / "viz" / "run_all_viz_0707.py",
            "--run-dir", run_dir,
        ])
    _run("result summary", [sys.executable, HERE / "summarize_results_0707.py"])
    print(f"PIPELINE_RUN_DIR='{run_dir}'")


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-stage", choices=("train", "sample", "viz"), default="")
    parser.add_argument("--resume", default="", help="checkpoint path or 'auto' from manifest")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    main()
