#!/usr/bin/env python3
"""Run one 20260801 experiment using the shared 20260707 implementation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
SHARED_ROOT = REPO_ROOT / "work" / "20260707_lincomb"
SHARED_SCRIPTS = SHARED_ROOT / "scripts"
SHARED_VIZ = SHARED_ROOT / "viz" / "run_all_viz_0707.py"

for path in (REPO_ROOT, SHARED_ROOT, SHARED_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import command_string, load_experiment_config, read_json  # noqa: E402


def create_run_dir(experiment: str, suffix: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        safe_suffix = "".join(char if char.isalnum() or char in "-_." else "_" for char in suffix)
        stamp = f"{stamp}_{safe_suffix}"
    run_dir = SUITE_ROOT / "runs" / experiment / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_stage(label: str, command: list[object]) -> None:
    printable = " ".join(str(value) for value in command)
    print(f"\n### {label}\n{printable}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
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


def main() -> None:
    args, extra = build_parser().parse_known_args()
    config = load_experiment_config(args.config)
    if args.dry_run:
        run_dir = Path(args.run_dir) if args.run_dir else SUITE_ROOT / "runs" / config["experiment"] / "<timestamp>"
        commands = [
            [sys.executable, SHARED_SCRIPTS / "train_0707.py", "--config", args.config, "--run-dir", run_dir, "--dry-run", *extra],
            [sys.executable, SHARED_SCRIPTS / "sample_0707.py", "--run-dir", run_dir, "--dry-run"],
            [sys.executable, SHARED_VIZ, "--run-dir", run_dir, "--dry-run"],
        ]
        print({"pipeline_command": command_string(), "run_dir": str(run_dir), "commands": [[str(value) for value in command] for command in commands]})
        return

    run_dir = Path(args.run_dir).resolve() if args.run_dir else create_run_dir(config["experiment"], args.name)
    (run_dir / "commands").mkdir(parents=True, exist_ok=True)
    (run_dir / "commands" / "pipeline.txt").write_text(command_string() + "\n", encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    def should_run(stage: str) -> bool:
        if args.force_stage and args.force_stage != stage:
            return False
        if args.skip_existing and manifest.get("stages", {}).get(stage, {}).get("status") == "completed":
            print(f"[pipeline] skip completed stage: {stage}")
            return False
        return True

    train_command = [sys.executable, SHARED_SCRIPTS / "train_0707.py", "--config", args.config, "--run-dir", run_dir, *extra]
    if args.resume:
        resume_path = manifest.get("checkpoint_path", "") if args.resume == "auto" else args.resume
        if not resume_path:
            raise ValueError("--resume auto requested but manifest has no checkpoint_path")
        train_command += ["--set", f"resume_checkpoint={resume_path}"]
    if should_run("train"):
        run_stage("train", train_command)
    if not args.skip_sample and should_run("sample"):
        run_stage("sample", [sys.executable, SHARED_SCRIPTS / "sample_0707.py", "--run-dir", run_dir])
    if not args.skip_viz and should_run("viz"):
        run_stage("visualization", [sys.executable, SHARED_VIZ, "--run-dir", run_dir])
    print(f"PIPELINE_RUN_DIR='{run_dir}'")


if __name__ == "__main__":
    main()
