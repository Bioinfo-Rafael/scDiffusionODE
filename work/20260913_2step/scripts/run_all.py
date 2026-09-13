"""Launch the complete experiment in a detached process; stdlib-only launcher."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback
import uuid

SUITE = Path(__file__).resolve().parents[1]
ROOT = SUITE.parents[1]
CONDITIONS = ["stage1_cellunet"] + [
    f"{family}_{objective}_soft"
    for objective in ("recon", "ot")
    for family in ("hill_after_linear", "centered_signed_hill", "shifted_hill_rho")
]


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def plan(args):
    steps = []

    def add(name, script, argv, capture=None):
        steps.append(
            dict(
                name=name,
                argv=[sys.executable, "-B", "-u", str(SUITE / "scripts" / script)]
                + argv,
                capture=capture or {},
            )
        )

    for condition in CONDITIONS:
        argv = [
            "--campaign",
            args.campaign,
            "--condition",
            condition,
            "--device",
            args.device,
        ]
        if condition == "stage1_cellunet":
            argv += ["--data", args.data, "--edge-tsv", args.edge_tsv]
        add(
            condition + ".train",
            "train.py",
            argv,
            {"EMA_CHECKPOINT": f"@{condition}.checkpoint"},
        )
    for condition in CONDITIONS:
        checkpoint = f"@{condition}.checkpoint"
        trajectory = f"@{condition}.trajectory"
        analysis = f"@{condition}.analysis"
        embedding = f"@{condition}.embedding"
        add(
            condition + ".sample",
            "sample.py",
            [
                "--checkpoint",
                checkpoint,
                "--device",
                args.device,
                "--num-samples",
                str(args.num_samples),
                "--batch-size",
                str(args.batch_size),
                "--post-ode-dt",
                str(args.post_ode_dt),
            ],
            {"TRAJECTORY_DIR": trajectory},
        )
        add(
            condition + ".analyze",
            "analyze.py",
            ["--trajectory", trajectory, "--device", args.device],
            {"ANALYSIS_DIR": analysis},
        )
        add(
            condition + ".embed",
            "embed.py",
            ["--trajectory", trajectory],
            {"ANALYSIS_DIR": embedding},
        )
        add(condition + ".metrics_plot", "plot.py", ["--input", analysis])
        add(condition + ".umap_plot", "plot.py", ["--input", embedding])
    return steps


def run_step(step, artifacts, launch):
    argv = [artifacts.get(arg, arg) for arg in step["argv"]]
    unresolved = [arg for arg in argv if arg.startswith("@")]
    if unresolved:
        raise RuntimeError(f"unresolved preceding outputs: {unresolved}")
    print(f"[{step['name']}] {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(shlex.join(argv), flush=True)
    captures = {key: [] for key in step["capture"]}
    with (launch / (step["name"] + ".log")).open("x") as log:
        with subprocess.Popen(
            argv,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                for key in captures:
                    if line.startswith(key + "="):
                        captures[key].append(line[len(key) + 1 :].strip())
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, argv)
    for key, values in captures.items():
        if len(values) != 1 or not Path(values[0]).exists():
            raise RuntimeError(f"{step['name']}: expected one existing {key}: {values}")
        artifacts[step["capture"][key]] = values[0]


def worker(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    launch = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    artifacts = {}
    os.environ["MPLCONFIGDIR"] = str(launch / "matplotlib")
    (launch / "matplotlib").mkdir(exist_ok=False)
    current = "preflight"
    try:
        if manifest["device"].startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable in the selected Python environment"
                )
            print(
                torch.cuda.get_device_name(torch.device(manifest["device"])), flush=True
            )
        for step in manifest["steps"]:
            current = step["name"]
            run_step(step, artifacts, launch)
            write_json(launch / (current + ".completed.json"), {"artifacts": artifacts})
        write_json(
            launch / "completed.json", {"status": "completed", "artifacts": artifacts}
        )
        print("[COMPLETED] " + manifest["campaign"], flush=True)
    except BaseException as error:
        write_json(
            launch / "failed.json",
            {
                "status": "failed",
                "step": current,
                "error": f"{type(error).__name__}: {error}",
                "artifacts": artifacts,
            },
        )
        traceback.print_exc()
        return 1
    return 0


def parser():
    defaults = json.loads(
        (ROOT / "work/20260803_ODE_hill_exp/configs/base.json").read_text()
    )
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument(
        "--campaign",
        default="two_step_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8],
    )
    p.add_argument("--data", default=defaults["data_dir"])
    p.add_argument("--edge-tsv", default=defaults["edge_tsv_path"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-samples", type=int, default=defaults["num_samples"])
    p.add_argument("--batch-size", type=int, default=defaults["sample_batch_size"])
    p.add_argument("--post-ode-dt", type=float, default=0.001)
    p.add_argument(
        "--foreground",
        action="store_true",
        help="wait for completion instead of detaching",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands only; no data access, jobs or outputs",
    )
    p.add_argument("--worker", help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if Path(args.campaign).name != args.campaign or args.campaign in ("", ".", ".."):
        raise ValueError("campaign must be one safe path component")
    if (
        args.num_samples < 2
        or args.batch_size < 1
        or not math.isfinite(args.post_ode_dt)
        or args.post_ode_dt <= 0
    ):
        raise ValueError(
            "requires >=2 samples, positive batch size and finite positive post-ODE dt"
        )
    args.data = str(Path(args.data).expanduser().resolve())
    args.edge_tsv = str(Path(args.edge_tsv).expanduser().resolve())
    steps = plan(args)
    if args.dry_run:
        for step in steps:
            print(shlex.join(step["argv"]))
        return 0
    for path in (args.data, args.edge_tsv):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if (SUITE / "runs" / args.campaign).exists():
        raise FileExistsError("campaign already exists; choose a new --campaign")
    launch = SUITE / "launches" / args.campaign
    launch.mkdir(parents=True, exist_ok=False)
    manifest = {
        "campaign": args.campaign,
        "device": args.device,
        "steps": steps,
        "python": sys.executable,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    manifest_path = launch / "launch.json"
    write_json(manifest_path, manifest)
    with (launch / "nohup.log").open("x") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                str(manifest_path),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=not args.foreground,
        )
    (launch / "pid").write_text(str(process.pid) + "\n")
    print(f"PID={process.pid}\nCAMPAIGN={args.campaign}\nLAUNCH_DIR={launch}")
    print("Progress: tail -f " + shlex.quote(str(launch / "nohup.log")))
    return process.wait() if args.foreground else 0


if __name__ == "__main__":
    raise SystemExit(main())
