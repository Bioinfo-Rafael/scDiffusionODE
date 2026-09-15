"""Launch the complete experiment in a detached process; stdlib-only launcher."""

import argparse
from datetime import datetime, timezone
import json
import importlib
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
    for objective in ("recon", "trajectory_ot")
    for family in ("hill_after_linear", "centered_signed_hill", "shifted_hill_rho")
]

LEGACY_OT_CONDITIONS = [
    f"{family}_ot_soft"
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

    completed_only = bool(getattr(args, "analyze_completed_campaign", None))
    legacy_only = bool(getattr(args, "analyze_legacy_ot_campaign", None))
    analysis_only = (
        bool(getattr(args, "analyze_campaign", None)) or completed_only or legacy_only
    )
    conditions = CONDITIONS if completed_only or not analysis_only else CONDITIONS[:4]
    if legacy_only:
        conditions = LEGACY_OT_CONDITIONS
    occupation_enabled = completed_only or not analysis_only

    def add_cache(role):
        add(
            "x50." + role + "_cache",
            "cache_x50.py",
            [
                "--campaign",
                args.campaign,
                "--role",
                role,
                "--device",
                args.device,
                "--cache-size",
                str(
                    args.source_cache_size if role == "train" else args.eval_cache_size
                ),
                "--batch-size",
                str(args.batch_size),
            ],
            {"X50_CACHE": "@x50." + role},
        )

    for condition in [] if analysis_only else conditions:
        if condition == CONDITIONS[4]:
            add_cache("train")
        argv = [
            "--campaign",
            args.campaign,
            "--condition",
            condition,
            "--device",
            args.device,
        ]
        if condition.endswith("_trajectory_ot_soft"):
            argv += [
                "--ot-max-iterations",
                str(args.ot_max_iterations),
                "--source-cache",
                "@x50.train",
            ]
        if condition == "stage1_cellunet":
            argv += ["--data", args.data, "--edge-tsv", args.edge_tsv]
        add(
            condition + ".train",
            "train.py",
            argv,
            {"EMA_CHECKPOINT": f"@{condition}.checkpoint"},
        )
    if occupation_enabled:
        add_cache("evaluation")
    occupation_outputs = []
    for condition in conditions:
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
            [
                "--trajectory",
                trajectory,
                "--device",
                args.device,
                "--ot-max-iterations",
                str(args.ot_max_iterations),
            ],
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
        if occupation_enabled and condition != "stage1_cellunet":
            occupation = f"@{condition}.occupation"
            add(
                condition + ".occupation",
                "occupation.py",
                [
                    "--checkpoint",
                    checkpoint,
                    "--source-cache",
                    "@x50.evaluation",
                    "--device",
                    args.device,
                    "--ot-max-iterations",
                    str(args.ot_max_iterations),
                ],
                {"OCCUPATION_DIR": occupation},
            )
            occupation_outputs.append(occupation)
    if occupation_outputs:
        add(
            "occupation.comparison_plot",
            "plot_occupation.py",
            ["--input"] + occupation_outputs,
        )
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


def recovery_steps(manifest):
    """Verify saved training, reuse finals, and continue partial bundles in new runs."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    checkpoints = importlib.import_module("work.20260913_2step.training.checkpoints")
    recovery = importlib.import_module("work.20260913_2step.training.recovery")
    campaign = SUITE / "runs" / manifest["campaign"]
    _, canonical = checkpoints.canonical_stage1(campaign)
    artifacts = {"@stage1_cellunet.checkpoint": canonical["checkpoint"]}
    selections = {"stage1_cellunet": {"completed_checkpoint": canonical["checkpoint"]}}
    steps = []
    if manifest.get("completed_analysis"):
        available = {"stage1_cellunet"}
        for condition in CONDITIONS[1:]:
            selected = recovery.select_training(
                campaign, condition, canonical, completed_only=True
            )
            selections[condition] = selected
            if "completed_checkpoint" in selected:
                available.add(condition)
                artifacts[f"@{condition}.checkpoint"] = selected["completed_checkpoint"]
        for original in manifest["steps"]:
            name = original["name"]
            if name == "x50.evaluation_cache":
                if len(available) > 1:
                    steps.append(original)
                continue
            if name == "occupation.comparison_plot":
                if len(available) > 1:
                    step = {
                        **original,
                        "argv": [
                            arg
                            for arg in original["argv"]
                            if not arg.startswith("@")
                            or arg.removeprefix("@").removesuffix(".occupation")
                            in available
                        ],
                    }
                    steps.append(step)
                continue
            condition, action = name.rsplit(".", 1)
            if condition not in CONDITIONS or action not in {
                "sample",
                "analyze",
                "embed",
                "metrics_plot",
                "umap_plot",
                "occupation",
            }:
                raise ValueError(
                    "completed-analysis plan contains a forbidden training/cache step"
                )
            if condition in available:
                steps.append(original)
        return steps, artifacts, selections
    if manifest.get("analysis_only"):
        analysis_conditions = (
            LEGACY_OT_CONDITIONS
            if manifest.get("legacy_ot_analysis")
            else CONDITIONS[:4]
        )
        for condition in analysis_conditions:
            if condition == "stage1_cellunet":
                continue
            selected = recovery.select_training(
                campaign, condition, canonical, completed_only=True
            )
            if "completed_checkpoint" not in selected:
                raise RuntimeError(
                    f"analysis-only requires completed training: {condition}; no training will be started"
                )
            artifacts[f"@{condition}.checkpoint"] = selected["completed_checkpoint"]
            selections[condition] = selected
        for step in manifest["steps"]:
            condition, action = step["name"].rsplit(".", 1)
            if condition not in analysis_conditions or action not in {
                "sample",
                "analyze",
                "embed",
                "metrics_plot",
                "umap_plot",
            }:
                raise ValueError("analysis-only plan contains a forbidden step")
        return manifest["steps"], artifacts, selections
    for original in manifest["steps"]:
        step = {**original, "argv": list(original["argv"])}
        if not step["name"].endswith(".train"):
            steps.append(step)
            continue
        condition = step["name"].removesuffix(".train")
        if condition == "stage1_cellunet":
            continue
        selected = recovery.select_training(campaign, condition, canonical)
        selections[condition] = selected
        if "completed_checkpoint" in selected:
            artifacts[f"@{condition}.checkpoint"] = selected["completed_checkpoint"]
        else:
            if "resume_checkpoint" in selected:
                step["argv"] += ["--resume-checkpoint", selected["resume_checkpoint"]]
            steps.append(step)
    return steps, artifacts, selections


def analysis_restart(path):
    """Read a failed analysis launch and reuse only its completed prefix."""
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to((SUITE / "launches").resolve()):
        raise ValueError("resume analysis must be below suite/launches")
    original = json.loads((path / "launch.json").read_text())
    if not original.get("analysis_only") or not (path / "failed.json").is_file():
        raise ValueError(
            "requires a failed analysis-only launch; never resumes training"
        )
    execution = json.loads((path / "execution_plan.json").read_text())
    artifacts = dict(execution["artifacts"])
    steps = execution["steps"]
    allowed = {
        "sample.py",
        "analyze.py",
        "embed.py",
        "plot.py",
        "occupation.py",
        "plot_occupation.py",
        "cache_x50.py",
    }
    for step in steps:
        if len(step["argv"]) < 4 or Path(step["argv"][3]).name not in allowed:
            raise ValueError("resume-analysis plan contains a forbidden command")
    consumed = 0
    for step in steps:
        marker = path / (step["name"] + ".completed.json")
        if not marker.exists():
            break
        recorded = json.loads(marker.read_text())["artifacts"]
        for key in step["capture"].values():
            value = Path(recorded[key])
            if (
                not value.is_dir()
                or not (value / "completed.json").is_file()
                or (value / "failed.json").exists()
            ):
                raise ValueError(
                    f"completed analysis artifact missing or failed: {value}"
                )
            if (
                json.loads((value / "completed.json").read_text())["status"]
                != "completed"
            ):
                raise ValueError(f"incomplete analysis artifact: {value}")
            artifacts[key] = str(value)
        consumed += 1
    for key, value in artifacts.items():
        if not Path(value).exists():
            raise FileNotFoundError(value)
        if key.endswith(".checkpoint"):
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            importlib.import_module(
                "work.20260913_2step.training.recovery"
            ).verified_checkpoint(value)
    if consumed == len(steps):
        raise ValueError("all analysis steps already completed")
    return steps[consumed:], artifacts, original


def worker(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    launch = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("six_ot"):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        return importlib.import_module("work.20260913_2step.scripts.six_ot").worker(
            manifest_path
        )
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
        steps = manifest["steps"]
        if manifest.get("resume_analysis_launch"):
            steps, artifacts, _ = analysis_restart(manifest["resume_analysis_launch"])
            write_json(
                launch / "execution_plan.json", {"steps": steps, "artifacts": artifacts}
            )
            print("RESUMING_ANALYSIS_FROM=" + steps[0]["name"], flush=True)
        elif manifest.get("resume_campaign") or manifest.get("analysis_only"):
            steps, artifacts, selections = recovery_steps(manifest)
            write_json(launch / "recovery_selection.json", selections)
            write_json(
                launch / "execution_plan.json", {"steps": steps, "artifacts": artifacts}
            )
            print("RECOVERY_SELECTION=" + json.dumps(selections), flush=True)
        for step in steps:
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
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-six-ot",
        action="store_true",
        help="new Hybrid500 six-OT campaign, including analysis",
    )
    mode.add_argument(
        "--resume-six-ot-campaign",
        help="resume incomplete Hybrid500 work without overwriting completed outputs",
    )
    p.add_argument(
        "--stage1-campaign", help="read-only canonical Stage1 source for --run-six-ot"
    )
    mode.add_argument(
        "--resume-analysis-launch",
        help="continue a failed analysis launch, reusing completed postprocessing; never train",
    )
    mode.add_argument(
        "--analyze-completed-campaign",
        help="analyze all completed canonical models, including trajectory OT; skip incomplete models and never train",
    )
    mode.add_argument(
        "--analyze-legacy-ot-campaign",
        help="analyze completed legacy OT models only; never train or run trajectory OT",
    )
    mode.add_argument(
        "--analyze-campaign",
        help="sample/analyze/plot completed Stage 1 + three reconstruction conditions only; never train",
    )
    mode.add_argument(
        "--resume-campaign",
        help="reuse Stage1/recon; start new trajectory OT or resume compatible trajectory bundles; never resume legacy OT",
    )
    p.add_argument(
        "--ot-max-iterations",
        type=int,
        default=2000,
        help="hard cap with convergence checks every 10 iterations (default: 2000)",
    )
    p.add_argument("--data", default=defaults["data_dir"])
    p.add_argument("--edge-tsv", default=defaults["edge_tsv_path"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--source-cache-size", type=int, default=8192)
    p.add_argument("--eval-cache-size", type=int, default=2048)
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
    if args.run_six_ot or args.resume_six_ot_campaign:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        return importlib.import_module("work.20260913_2step.scripts.six_ot").launch(
            args
        )
    if args.stage1_campaign:
        raise ValueError("--stage1-campaign requires --run-six-ot")
    restart_steps = None
    if args.resume_analysis_launch:
        restart_steps, _, previous = analysis_restart(args.resume_analysis_launch)
        args.analyze_campaign = previous["campaign"]
        args.device = previous["device"]
    analysis_only = bool(
        args.analyze_campaign
        or args.analyze_completed_campaign
        or args.analyze_legacy_ot_campaign
    )
    existing_campaign = (
        args.resume_campaign
        or args.analyze_campaign
        or args.analyze_completed_campaign
        or args.analyze_legacy_ot_campaign
    )
    if existing_campaign:
        args.campaign = existing_campaign
    if Path(args.campaign).name != args.campaign or args.campaign in ("", ".", ".."):
        raise ValueError("campaign must be one safe path component")
    if (
        args.source_cache_size < 128
        or args.eval_cache_size < 2
        or args.ot_max_iterations < 2000
        or args.num_samples < 2
        or args.batch_size < 1
        or not math.isfinite(args.post_ode_dt)
        or args.post_ode_dt <= 0
    ):
        raise ValueError(
            "requires OT cap >=2000, >=2 samples, positive batch size and finite positive post-ODE dt"
        )
    args.data = str(Path(args.data).expanduser().resolve())
    args.edge_tsv = str(Path(args.edge_tsv).expanduser().resolve())
    steps = restart_steps if restart_steps is not None else plan(args)
    if args.dry_run:
        if existing_campaign and not args.resume_analysis_launch:
            steps, _, selections = recovery_steps(
                {
                    "campaign": args.campaign,
                    "steps": steps,
                    "analysis_only": analysis_only,
                    "completed_analysis": bool(args.analyze_completed_campaign),
                    "legacy_ot_analysis": bool(args.analyze_legacy_ot_campaign),
                }
            )
            print("RECOVERY_SELECTION=" + json.dumps(selections))
        for step in steps:
            print(shlex.join(step["argv"]))
        return 0
    if existing_campaign:
        campaign = SUITE / "runs" / args.campaign
        canonical = json.loads((campaign / "canonical_stage1.json").read_text())
        stage1 = json.loads(
            Path(canonical["checkpoint"]).with_suffix(".json").read_text()
        )
        args.data = stage1["effective_config"]["data_dir"]
        args.edge_tsv = stage1["effective_config"]["edge_tsv_path"]
        for started in [] if analysis_only else campaign.glob("*/*/started.json"):
            if not any(
                (started.parent / name).exists()
                for name in ("completed.json", "failed.json")
            ):
                raise RuntimeError(
                    f"training run has no terminal status: {started.parent}; verify it is stopped"
                )
        launch_root = SUITE / "launches" / args.campaign
        for prior in [] if analysis_only else launch_root.rglob("launch.json"):
            if not any(
                (prior.parent / name).exists()
                for name in ("completed.json", "failed.json")
            ):
                raise RuntimeError(
                    f"launcher has no terminal status: {prior.parent}; verify it is stopped"
                )
    for path in (args.data,) if analysis_only else (args.data, args.edge_tsv):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if not existing_campaign and (SUITE / "runs" / args.campaign).exists():
        raise FileExistsError("campaign already exists; choose a new --campaign")
    launch = SUITE / "launches" / args.campaign
    if existing_campaign:
        launch = launch / (
            ("analysis_" if analysis_only else "recovery_")
            + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8]
        )
    launch.mkdir(parents=True, exist_ok=False)
    manifest = {
        "campaign": args.campaign,
        "resume_analysis_launch": str(Path(args.resume_analysis_launch).resolve())
        if args.resume_analysis_launch
        else None,
        "resume_campaign": bool(args.resume_campaign),
        "analysis_only": analysis_only,
        "completed_analysis": bool(args.analyze_completed_campaign),
        "legacy_ot_analysis": bool(args.analyze_legacy_ot_campaign),
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
