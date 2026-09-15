"""Isolated six-OT campaigns using the existing training and analysis commands."""

import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

PKG = "work.20260913_2step"


def modules():
    return (
        importlib.import_module(PKG + ".common"),
        importlib.import_module(PKG + ".scripts.run_all"),
        importlib.import_module(PKG + ".training.checkpoints"),
        importlib.import_module(PKG + ".training.recovery"),
    )


def plan(args, campaign):
    common, launcher, _, _ = modules()
    spec = common.read_json(common.SUITE / "configs/hybrid500_ot.json")
    steps = []

    def add(name, script, argv, capture):
        steps.append(
            dict(
                name=name,
                argv=[
                    sys.executable,
                    "-B",
                    "-u",
                    str(common.SUITE / "scripts" / script),
                ]
                + argv,
                capture=capture,
            )
        )

    def cache(role, size):
        add(
            "x50." + role + "_cache",
            "cache_x50.py",
            [
                "--campaign",
                campaign,
                "--role",
                role,
                "--cache-size",
                str(size),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
            ],
            {"X50_CACHE": "@x50." + role},
        )

    cache("evaluation", 2048)
    for condition in spec["conditions"]:
        trajectory_ot = "_trajectory_ot_" in condition
        if condition == spec["conditions"][3]:
            cache("train", 8192)
        prefix = "@" + condition
        train_args = [
            "--campaign",
            campaign,
            "--condition",
            condition,
            "--device",
            args.device,
        ]
        if trajectory_ot:
            train_args += ["--source-cache", "@x50.train"]
        add(
            condition + ".train",
            "train.py",
            train_args,
            {"EMA_CHECKPOINT": prefix + ".checkpoint"},
        )
        add(
            condition + ".sample",
            "sample.py",
            [
                "--checkpoint",
                prefix + ".checkpoint",
                "--device",
                args.device,
                "--num-samples",
                str(args.num_samples),
                "--batch-size",
                str(args.batch_size),
                "--post-ode-dt",
                "0.001",
            ],
            {"TRAJECTORY_DIR": prefix + ".trajectory"},
        )
        for action in ("analyze", "embed"):
            add(
                condition + "." + action,
                action + ".py",
                ["--trajectory", prefix + ".trajectory", "--device", args.device],
                {"ANALYSIS_DIR": prefix + "." + action},
            )
        for action, source in (("metrics_plot", "analyze"), ("umap_plot", "embed")):
            add(
                condition + "." + action,
                "plot.py",
                ["--input", prefix + "." + source],
                {"FIGURE_DIR": prefix + "." + action},
            )
        add(
            condition + ".occupation",
            "occupation.py",
            [
                "--checkpoint",
                prefix + ".checkpoint",
                "--source-cache",
                "@x50.evaluation",
                "--device",
                args.device,
            ],
            {"OCCUPATION_DIR": prefix + ".occupation"},
        )
    add(
        "occupation.comparison_plot",
        "plot_occupation.py",
        ["--input"] + ["@" + c + ".occupation" for c in spec["conditions"]],
        {"FIGURE_DIR": "@occupation.comparison"},
    )
    return steps


def checked_artifacts(record, step, recovery):
    result = {}
    for key in step["capture"].values():
        value = Path(record["artifacts"][key])
        if key.endswith(".checkpoint"):
            recovery.verified_checkpoint(value)
        else:
            marker = value / "completed.json"
            if (
                not marker.is_file()
                or (value / "failed.json").exists()
                or json.loads(marker.read_text())["status"] != "completed"
            ):
                raise ValueError(f"missing or failed reusable output: {value}")
        result[key] = str(value)
    return result


def execute(manifest, launch, *, run_step=None):
    common, launcher, checkpoints, recovery = modules()
    run_step = run_step or launcher.run_step
    campaign = common.SUITE / "runs" / manifest["campaign"]
    _, canonical = checkpoints.canonical_stage1(campaign)
    artifacts, failures, skipped, reused, missing_ot = {}, [], [], [], []
    history = []
    for prior in sorted(launch.parent.glob("*/launch.json")):
        if prior.parent != launch:
            old = common.read_json(prior)
            if old.get("six_ot"):
                history.append(prior.parent)
    common.write_json(
        launch / "execution_plan.json", {"steps": manifest["steps"], "artifacts": {}}
    )
    for original in manifest["steps"]:
        step = {**original, "argv": list(original["argv"])}
        name = step["name"]
        if name == "occupation.comparison_plot":
            step["argv"] = [
                a for a in step["argv"] if not a.startswith("@") or a in artifacts
            ]
            if step["argv"][-1] == "--input":
                skipped.append(name)
                continue
        if any(a.startswith("@") and a not in artifacts for a in step["argv"]):
            skipped.append(name)
            common.write_json(
                launch / (name + ".skipped.json"), {"reason": "dependency unavailable"}
            )
            continue
        resolved = [artifacts.get(a, a) for a in step["argv"]]
        try:
            reused_record = None
            for prior in reversed(history):
                marker = prior / (name + ".completed.json")
                if marker.exists():
                    record = common.read_json(marker)
                    if record.get("argv") == resolved:
                        artifacts.update(checked_artifacts(record, step, recovery))
                        reused_record = record
                        break
            if name.endswith(".train"):
                condition = name.rsplit(".", 1)[0]
                selected = recovery.select_training(campaign, condition, canonical)
                common.write_json(launch / (name + ".selection.json"), selected)
                if "completed_checkpoint" in selected:
                    artifacts["@" + condition + ".checkpoint"] = selected[
                        "completed_checkpoint"
                    ]
                    reused_record = True
                elif "resume_checkpoint" in selected:
                    step["argv"] += [
                        "--resume-checkpoint",
                        selected["resume_checkpoint"],
                    ]
            if reused_record:
                reused.append(name)
                print("[REUSED] " + name, flush=True)
            else:
                run_step(step, artifacts, launch)
            common.write_json(
                launch / (name + ".completed.json"),
                {
                    "argv": resolved,
                    "artifacts": dict(artifacts),
                    "reused": bool(reused_record),
                },
            )
            for key in step["capture"].values():
                status = Path(artifacts[key]) / "evaluation_ot_status.json"
                if status.is_file() and common.read_json(status).get(
                    "missing_count", 0
                ):
                    missing_ot.append({"step": name, "status_file": str(status)})
        except Exception as error:
            for key in step["capture"].values():
                artifacts.pop(key, None)
            # A condition failure does not discard other independent work.
            failure = {"step": name, "error": f"{type(error).__name__}: {error}"}
            failures.append(failure)
            common.write_json(launch / (name + ".failed.json"), failure)
            traceback.print_exc()
    summary = dict(
        status="partial_failure" if failures or skipped else "completed",
        artifacts=artifacts,
        failures=failures,
        skipped=skipped,
        reused=reused,
        evaluation_ot_missing=missing_ot,
    )
    common.write_json(
        launch / ("failed.json" if failures or skipped else "completed.json"), summary
    )
    print("SIX_OT_STATUS=" + summary["status"], flush=True)
    return 1 if failures or skipped else 0


def worker(manifest_path):
    import fcntl

    common, _, _, _ = modules()
    manifest_path = Path(manifest_path)
    manifest = common.read_json(manifest_path)
    launch = manifest_path.parent
    campaign = common.SUITE / "runs" / manifest["campaign"]
    try:
        # Kernel lock is released even after interruption; no stale PID heuristics.
        with (campaign / ".worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if manifest["device"].startswith("cuda"):
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable")
            os.environ["MPLCONFIGDIR"] = str(launch / "matplotlib")
            return execute(manifest, launch)
    except BaseException as error:
        if not (launch / "failed.json").exists():
            common.write_json(
                launch / "failed.json",
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
            )
        traceback.print_exc()
        return 1


def launch(args):
    common, launcher, checkpoints, _ = modules()
    resume = args.resume_six_ot_campaign
    if resume and args.stage1_campaign:
        raise ValueError("resume uses its recorded Stage1; omit --stage1-campaign")
    if (
        args.source_cache_size != 8192
        or args.eval_cache_size != 2048
        or args.post_ode_dt != 0.001
        or args.ot_max_iterations != 2000
    ):
        raise ValueError(
            "six-OT mode preserves cache, dt and Sinkhorn settings; remove overrides"
        )
    if args.num_samples < 2 or args.batch_size < 1:
        raise ValueError("requires >=2 samples and positive batch size")
    if resume:
        if Path(resume).name != resume or not resume.startswith("hybrid500_ot_"):
            raise ValueError("invalid six-OT campaign")
        campaign = common.confined(common.SUITE / "runs" / resume)
        spec = common.read_json(campaign / "experiment.json")
        if spec["experiment_mode"] != "hybrid500_ot_v1":
            raise ValueError("not a Hybrid500 campaign")
        # Resume keeps original sampling/device settings and scientific parameters.
        for key, value in spec["launcher_settings"].items():
            setattr(args, key, value)
        checkpoints.canonical_stage1(campaign)
    else:
        if not args.stage1_campaign:
            raise ValueError("--run-six-ot requires --stage1-campaign")
        if Path(
            args.stage1_campaign
        ).name != args.stage1_campaign or args.stage1_campaign in ("", ".", ".."):
            raise ValueError("invalid Stage1 campaign")
        source = common.confined(common.SUITE / "runs" / args.stage1_campaign)
        payload, _ = checkpoints.canonical_stage1(source)
        config = payload["metadata"]["effective_config"]
        for key in ("data_dir", "edge_tsv_path"):
            if not Path(config[key]).is_file():
                raise FileNotFoundError(config[key])
        campaign = common.SUITE / "runs" / ("hybrid500_ot_" + common.run_id())
        spec = common.read_json(common.SUITE / "configs/hybrid500_ot.json")
        spec["stage1_source_campaign"] = str(source)
        spec["launcher_settings"] = {
            k: getattr(args, k) for k in ("device", "num_samples", "batch_size")
        }
    steps = plan(args, campaign.name)
    if args.dry_run:
        for step in steps:
            print(shlex.join(step["argv"]))
        return 0
    if not resume:
        common.new_dir(campaign)
        common.write_json(campaign / "experiment.json", spec)
        checkpoints.register_stage1(source, campaign)
        for condition in spec["conditions"]:
            common.new_dir(campaign / condition)
    output = common.new_dir(
        common.SUITE / "launches" / campaign.name / ("six_ot_" + common.run_id())
    )
    manifest = dict(
        six_ot=True,
        campaign=campaign.name,
        device=args.device,
        steps=steps,
        git_commit=common.git_commit(),
        analysis_only=False,
    )
    common.write_json(output / "launch.json", manifest)
    with (output / "nohup.log").open("x") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-u",
                str(Path(launcher.__file__)),
                "--worker",
                str(output / "launch.json"),
            ],
            cwd=common.ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=not args.foreground,
        )
    (output / "pid").write_text(str(process.pid) + "\n")
    print(f"PID={process.pid}\nCAMPAIGN={campaign.name}\nLAUNCH_DIR={output}")
    print("Progress: tail -f " + shlex.quote(str(output / "nohup.log")))
    return process.wait() if args.foreground else 0
