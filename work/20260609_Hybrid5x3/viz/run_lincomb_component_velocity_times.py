#!/usr/bin/env python3
"""Latest lincomb__none runの指定t/component UMAP/PAGAをまとめて実行する。

既定では、名前が ``YYYYMMDD_HHMMSS...`` で始まるrun directoryのうち
日時が最新のものを選び、Erythropoieticについてt=499,999を処理する。
出力は選択runの ``viz/`` 直下に1つのroot directoryを作って集約する。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
DEFAULT_RUNS_DIR = WORK_ROOT / "runs" / "lincomb__none"
RUN_TIMESTAMP_RE = re.compile(r"^(\d{8})_(\d{6})(?:_|$)")


def _command_string(command):
    return " ".join(shlex.quote(str(value)) for value in command)


def _safe_name(value):
    name = str(value).strip() or "unnamed"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "unnamed"


def _safe_group_dir(value):
    name = str(value).strip() or "unnamed"
    return re.sub(r"[\\/:*?\"<>|\s]+", "_", name)


def _timestamp_key(path):
    match = RUN_TIMESTAMP_RE.match(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def select_latest_run(runs_dir):
    runs_dir = Path(runs_dir).expanduser().resolve()
    if not runs_dir.is_dir():
        raise SystemExit(f"[ERROR] runs directory does not exist: {runs_dir}")
    candidates = []
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        timestamp = _timestamp_key(path)
        if timestamp is not None:
            candidates.append((timestamp, path.name, path))
    if not candidates:
        raise SystemExit(
            f"[ERROR] no YYYYMMDD_HHMMSS* run directories found under {runs_dir}"
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2].resolve()


def _parse_t_values(value):
    parsed = []
    for item in str(value).split(","):
        text = item.strip()
        if not text:
            continue
        try:
            number = float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid timestep: {text}") from exc
        if not math.isfinite(number) or number < 0:
            raise argparse.ArgumentTypeError(
                f"timestep must be finite and non-negative: {text}"
            )
        if number not in parsed:
            parsed.append(number)
    if not parsed:
        raise argparse.ArgumentTypeError("at least one timestep is required")
    return parsed


def _t_label(value):
    if float(value).is_integer():
        return str(int(value))
    return (f"{value:g}").replace("-", "m").replace(".", "p")


def _checkpoint_step(path):
    match = re.search(r"(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _require_run_artifacts(run_dir):
    configs = list(run_dir.glob("train/**/exp_config.json"))
    if not configs:
        configs = list(run_dir.glob("**/exp_config.json"))
    checkpoints = list(run_dir.glob("train/**/checkpoints/**/ema_*.pt"))
    if not checkpoints:
        checkpoints = list(run_dir.glob("train/**/checkpoints/**/model*.pt"))
    if not checkpoints:
        checkpoints = list(run_dir.glob("**/ema_*.pt"))
    if not checkpoints:
        checkpoints = list(run_dir.glob("**/model*.pt"))
    missing = []
    if not configs:
        missing.append("exp_config.json")
    if not checkpoints:
        missing.append("checkpoint (ema_*.pt/model*.pt)")
    if missing:
        raise SystemExit(
            f"[ERROR] selected run is incomplete ({', '.join(missing)}): {run_dir}"
        )
    config_path = max(configs, key=lambda path: path.stat().st_mtime)
    checkpoint_path = max(
        checkpoints,
        key=lambda path: (_checkpoint_step(path), path.stat().st_mtime),
    )
    return config_path.resolve(), checkpoint_path.resolve()


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def _run_and_tee(command, log_handle):
    process = subprocess.Popen(
        [str(value) for value in command],
        cwd=str(HERE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_handle.write(line)
        log_handle.flush()
    return process.wait()


def _validate_job_output(job, group, component_mode):
    errors = []
    summary_path = Path(job["output_dir"]) / "summary.json"
    if not summary_path.is_file():
        return [f"missing child summary: {summary_path}"]
    try:
        with open(summary_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read child summary {summary_path}: {type(exc).__name__}: {exc}"]

    if not math.isclose(
        float(payload.get("velocity_t", float("nan"))),
        float(job["t"]),
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        errors.append(
            f"velocity_t mismatch: expected {job['t']}, got {payload.get('velocity_t')}"
        )
    if payload.get("component_mode") != component_mode:
        errors.append(
            "component_mode mismatch: "
            f"expected {component_mode}, got {payload.get('component_mode')}"
        )
    if payload.get("only_groups") != [group]:
        errors.append(
            f"only_groups mismatch: expected {[group]}, got {payload.get('only_groups')}"
        )
    groups = payload.get("groups") or {}
    if set(groups) != {group}:
        errors.append(f"processed groups mismatch: expected {[group]}, got {list(groups)}")
    failed_groups = payload.get("failed_groups") or {}
    if failed_groups:
        errors.append(f"failed_groups is not empty: {failed_groups}")

    group_payload = groups.get(group) or {}
    error_key = "graph_errors" if job["kind"] == "components" else "paga_errors"
    panel_errors = group_payload.get(error_key) or {}
    if panel_errors:
        errors.append(f"{error_key} is not empty: {panel_errors}")

    group_dir = (
        Path(job["output_dir"])
        / "velocity_by_superclass"
        / _safe_group_dir(group)
    )
    expected_names = (
        [
            "1_velocity_stream_grid.png",
            "2_velocity_arrow_grid.png",
            "3_velocity_stream_lineage_grid.png",
            "4_velocity_arrow_lineage_grid.png",
        ]
        if job["kind"] == "components"
        else [
            "1_velocity_paga_grid.png",
            "2_velocity_paga_lineage_grid.png",
        ]
    )
    missing = [name for name in expected_names if not (group_dir / name).is_file()]
    if missing:
        errors.append(f"missing expected PNGs under {group_dir}: {missing}")
    return errors


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "最新lincomb__none runを選び、指定tのLinComb component UMAP/PAGAを"
            "1つのviz output rootへ保存する。"
        )
    )
    parser.add_argument(
        "--runs_dir",
        default=str(DEFAULT_RUNS_DIR),
        help="timestamp run directoriesを含むlincomb__none directory。",
    )
    parser.add_argument(
        "--run_dir",
        default="",
        help="自動選択を使わず対象runを明示する場合に指定。",
    )
    parser.add_argument(
        "--t_values",
        type=_parse_t_values,
        default=_parse_t_values("499,999"),
    )
    parser.add_argument("--group_col", default="Superclass")
    parser.add_argument("--group", default="Erythropoietic")
    parser.add_argument(
        "--component_mode",
        choices=("expert", "contribution"),
        default="expert",
        help="既定expertは既存t=0図と同じraw V_k。contributionならa_k(x,t)V_k(x)。",
    )
    parser.add_argument("--max_cells", type=int, default=0)
    parser.add_argument("--min_cells", type=int, default=15)
    parser.add_argument("--n_jobs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--output_name",
        default="",
        help="選択runのviz/直下に作るdirectory名。未指定ならgroup/tから生成。",
    )
    parser.add_argument(
        "--reuse_output",
        action="store_true",
        help="既存output rootへの再実行を許可する（既存ファイルは削除しない）。",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main():
    args = build_argparser().parse_args()
    if args.max_cells < 0:
        raise SystemExit("[ERROR] --max_cells must be >= 0")
    if args.min_cells < 2:
        raise SystemExit("[ERROR] --min_cells must be >= 2")
    if args.n_jobs <= 0:
        raise SystemExit("[ERROR] --n_jobs must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("[ERROR] --batch_size must be > 0")

    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else select_latest_run(args.runs_dir)
    )
    if not run_dir.is_dir():
        raise SystemExit(f"[ERROR] run directory does not exist: {run_dir}")
    config_path, checkpoint_path = _require_run_artifacts(run_dir)

    t_labels = [_t_label(value) for value in args.t_values]
    output_name = args.output_name or (
        f"velocity_lincomb_{_safe_name(args.group)}_t{'_'.join(t_labels)}"
    )
    output_root = run_dir / "viz" / output_name

    jobs = []
    for timestep, label in zip(args.t_values, t_labels):
        common = [
            "--run_dir", str(run_dir),
            "--config", str(config_path),
            "--model_path", str(checkpoint_path),
            "--velocity_t", f"{timestep:g}",
            "--group_col", args.group_col,
            "--only_groups", args.group,
            "--max_cells", str(args.max_cells),
            "--min_cells", str(args.min_cells),
            "--n_jobs", str(args.n_jobs),
            "--batch_size", str(args.batch_size),
            "--component_mode", args.component_mode,
        ]
        jobs.append({
            "name": f"components_t{label}",
            "t": timestep,
            "kind": "components",
            "output_dir": output_root / f"t{label}" / "velocity_lincomb_components",
            "script": HERE / "plot_lincomb_component_velocity_umap.py",
            "common": common,
        })
        jobs.append({
            "name": f"paga_t{label}",
            "t": timestep,
            "kind": "paga",
            "output_dir": output_root / f"t{label}" / "velocity_lincomb_component_paga",
            "script": HERE / "plot_lincomb_component_velocity_paga.py",
            "common": common,
        })

    for job in jobs:
        job["command"] = [
            sys.executable,
            str(job["script"]),
            *job.pop("common"),
            "--output_dir",
            str(job["output_dir"]),
        ]

    print(f"[lincomb-times] selected_run : {run_dir}")
    print(f"[lincomb-times] config       : {config_path}")
    print(f"[lincomb-times] checkpoint   : {checkpoint_path}")
    print(f"[lincomb-times] group        : {args.group_col}={args.group}")
    print(f"[lincomb-times] t_values     : {', '.join(t_labels)}")
    print(f"[lincomb-times] mode         : {args.component_mode}")
    print(f"[lincomb-times] output_root  : {output_root}")
    for job in jobs:
        print(f"[lincomb-times] {job['name']}: {_command_string(job['command'])}")
    if args.dry_run:
        print("[lincomb-times] dry-run: no directories created and no jobs executed")
        return

    if output_root.exists() and not args.reuse_output:
        raise SystemExit(
            f"[ERROR] output root already exists: {output_root}\n"
            "Use --reuse_output to continue without deleting existing files, or set --output_name."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().isoformat(timespec="seconds")
    wrapper_command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    summary = {
        "status": "running",
        "started_at": started_at,
        "selected_run": str(run_dir),
        "selection_rule": (
            "maximum YYYYMMDD_HHMMSS prefix under runs_dir"
            if not args.run_dir
            else "explicit --run_dir"
        ),
        "runs_dir": str(Path(args.runs_dir).expanduser().resolve()),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "group_col": args.group_col,
        "group": args.group,
        "t_values": args.t_values,
        "component_mode": args.component_mode,
        "output_root": str(output_root),
        "wrapper_command": _command_string(wrapper_command),
        "jobs": [],
    }
    command_lines = [
        f"# started_at: {started_at}",
        f"# cwd: {Path.cwd()}",
        _command_string(wrapper_command),
        "",
        "# child jobs",
        *[_command_string(job["command"]) for job in jobs],
    ]
    (output_root / "command.txt").write_text(
        "\n".join(command_lines) + "\n", encoding="utf-8"
    )
    _write_json(output_root / "summary.json", summary)

    with open(output_root / "run.log", "a", encoding="utf-8") as log_handle:
        for job in jobs:
            print(f"\n[lincomb-times] START {job['name']}")
            log_handle.write(f"\n[lincomb-times] START {job['name']}\n")
            log_handle.write(_command_string(job["command"]) + "\n")
            log_handle.flush()
            job_record = {
                "name": job["name"],
                "kind": job["kind"],
                "t": job["t"],
                "output_dir": str(job["output_dir"]),
                "command": _command_string(job["command"]),
                "status": "running",
            }
            summary["jobs"].append(job_record)
            _write_json(output_root / "summary.json", summary)
            returncode = _run_and_tee(job["command"], log_handle)
            job_record["returncode"] = int(returncode)
            validation_errors = (
                _validate_job_output(job, args.group, args.component_mode)
                if returncode == 0
                else []
            )
            if validation_errors:
                job_record["validation_errors"] = validation_errors
            job_record["status"] = (
                "completed"
                if returncode == 0 and not validation_errors
                else "failed"
            )
            _write_json(output_root / "summary.json", summary)
            if returncode != 0 or validation_errors:
                summary["status"] = "failed"
                summary["failed_job"] = job["name"]
                summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
                _write_json(output_root / "summary.json", summary)
                reason = (
                    f"return code {returncode}"
                    if returncode != 0
                    else "; ".join(validation_errors)
                )
                raise SystemExit(
                    f"[ERROR] {job['name']} failed validation ({reason}); "
                    f"see {output_root / 'run.log'}"
                )
            print(f"[lincomb-times] DONE  {job['name']}")

    summary["status"] = "completed"
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(output_root / "summary.json", summary)
    print(f"[lincomb-times] completed: {output_root}")


if __name__ == "__main__":
    main()
