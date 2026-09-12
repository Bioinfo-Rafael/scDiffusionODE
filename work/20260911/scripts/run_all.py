#!/usr/bin/env python3
"""Validate every mapping before launching models sequentially in fresh processes."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from _bootstrap import ROOT, WORK
from discover_runs import discover_all

from src.model_runs import resolve_entry
from src.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--models", type=Path, default=WORK / "configs/models.json")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=50)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="Resolve mappings and print commands only"
    )
    mode.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect all source selections and data paths without loading models",
    )
    mode.add_argument(
        "--restore-only",
        action="store_true",
        help="Restore all selected checkpoints without sampling",
    )
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--edge-path", type=Path)
    args = parser.parse_args()
    config = json.loads(args.models.read_text())
    if Settings(**config["settings"]) != Settings():
        raise ValueError("production experiment fixes T=1000, M=20, S=100, N=20")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    reports = discover_all()
    resolved, problems, keys = [], [], set()
    for entry in config["models"]:
        if not entry.get("enabled", True):
            continue
        key = (entry["suite"], entry["experiment"])
        if key in keys:
            raise ValueError(f"duplicate model mapping: {key}")
        keys.add(key)
        try:
            resolved.append(resolve_entry(entry, reports[entry["suite"]]))
        except (ValueError, KeyError) as exc:
            problems.append(str(exc))
    if problems:
        raise ValueError("No jobs started. Resolve these mappings first:\n" + "\n".join(problems))
    if not resolved:
        raise ValueError("no enabled models in mapping")
    commands = []
    for entry in resolved:
        command = [
            sys.executable,
            str(WORK / "scripts/run_one_model.py"),
            "--suite",
            entry["suite"],
            "--run-dir",
            entry["run_dir"],
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
        ]
        if entry.get("checkpoint"):
            checkpoint = Path(entry["checkpoint"]).expanduser()
            command += [
                "--checkpoint",
                str(checkpoint if checkpoint.is_absolute() else ROOT / checkpoint),
            ]
        for option, value in (("--data-path", args.data_path), ("--edge-path", args.edge_path)):
            if value:
                command += [option, str(value.expanduser().resolve())]
        commands.append(command)
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    # Fail before any sampling if any selected run is invalid, including explicit mappings.
    for command in commands:
        subprocess.run(command + ["--inspect-only"], cwd=ROOT, check=True)
    if args.inspect_only:
        return
    for command in commands:
        subprocess.run(
            command + (["--restore-only"] if args.restore_only else []), cwd=ROOT, check=True
        )


if __name__ == "__main__":
    main()
