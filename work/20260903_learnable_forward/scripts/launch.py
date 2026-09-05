#!/usr/bin/env python3
"""Launch the learnable-forward training configs sequentially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Optional, Sequence


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    CONFIG_ROOT,
    MODEL_CONFIGS,
    MODEL_ORDER,
    load_experiment_config,
    new_batch_id,
    run_directory,
    safe_component,
)


def _selected_configs(args) -> tuple[Path, ...]:
    if args.config:
        return tuple(Path(value) for value in args.config)
    selected_models = tuple(args.model or MODEL_ORDER)
    by_model = dict(zip(MODEL_ORDER, MODEL_CONFIGS))
    return tuple(CONFIG_ROOT / by_model[name] for name in selected_models)


def _commands(args) -> tuple[str, list[list[str]]]:
    batch_id = safe_component(args.batch_id or new_batch_id(), "batch id")
    commands = []
    experiments = set()
    for config_path in _selected_configs(args):
        config, provenance = load_experiment_config(
            config_path,
            base_path=args.base_config,
            set_values=args.set_values,
        )
        experiment = str(config["experiment"])
        if experiment in experiments:
            raise ValueError(f"duplicate experiment selected: {experiment}")
        experiments.add(experiment)
        target = run_directory(experiment, batch_id)
        command = [
            args.python,
            str(HERE / "train.py"),
            "--config",
            provenance["experiment_config_path"],
            "--run-dir",
            str(target),
        ]
        if args.base_config:
            command.extend(["--base-config", provenance["base_config_path"]])
        if args.resume:
            command.extend(["--resume", args.resume])
        for value in args.set_values:
            command.extend(["--set", value])
        commands.append(command)
    if args.resume not in {"", "auto"} and len(commands) != 1:
        raise ValueError(
            "an explicit checkpoint path can only be used with one selected config"
        )
    return batch_id, commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--model",
        action="append",
        choices=MODEL_ORDER,
        help="select one or both canonical models; default runs both in order",
    )
    parser.add_argument(
        "--config",
        action="append",
        help="explicit model config path; overrides --model/default selection",
    )
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--batch-id", default="")
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=""
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="run remaining models after a training failure; exit nonzero if any failed",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    batch_id, commands = _commands(args)
    if args.dry_run:
        print(
            json.dumps(
                {"batch_id": batch_id, "commands": commands,
                 "continue_on_error": args.continue_on_error},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    failed = False
    for index, command in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            if not args.continue_on_error:
                raise
            failed = True
            print(
                f"[{index}/{len(commands)}] FAILED: exit code {error.returncode}.",
                file=sys.stderr,
                flush=True,
            )
            if index < len(commands):
                print("Continuing to the next model.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
