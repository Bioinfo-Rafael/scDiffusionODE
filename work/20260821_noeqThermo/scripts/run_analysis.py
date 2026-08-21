#!/usr/bin/env python3
"""CLI for fixed-UMAP nonequilibrium landscape/flux analysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noeqthermo.config import load_config
from noeqthermo.pipeline import (
    default_output_dir,
    discover_run_dirs,
    run_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="学習済みODEを固定Erythropoietic UMAP上のDynamo landscape/fluxへ接続する",
        allow_abbrev=False,
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", default="", help="modeに対応する完全なJSON設定の上書き")
    parser.add_argument("--run-dir", action="append", default=[], help="比較するrun（複数指定可）")
    parser.add_argument("--runs-root", default="", help="配下の対応runをすべて自動検出")
    parser.add_argument("--output-dir", default="", help="再開可能な解析出力directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="run検出と設定検証だけを行う")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.mode, args.config or None)
    run_dirs = discover_run_dirs(args.run_dir, args.runs_root or None)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(args.mode)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "mode": args.mode,
                    "config": config["_config_path"],
                    "run_dirs": [str(path) for path in run_dirs],
                    "output_dir": str(output_dir),
                    "device": args.device,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    result = run_pipeline(run_dirs, config, output_dir, device_name=args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"NOEQ_THERMO_OUTPUT_DIR='{output_dir}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
