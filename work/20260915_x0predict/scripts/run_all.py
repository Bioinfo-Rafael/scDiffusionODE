"""One campaign, one START_X Stage1, eight independent Stage2 conditions."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
from ..common import (
    CONDITIONS, STAGE1, ROOT, SUITE, campaign_path, confined, effective_config,
    file_hash, git_commit, new_dir, read_json, run_id, source_provenance, write_json,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--stage1-only", action="store_true")
    mode.add_argument("--stage2-only", action="store_true")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--condition", choices=CONDITIONS)
    selection.add_argument("--all-eight", action="store_true")
    p.add_argument("--resume-campaign", help="existing campaign name or path below this suite/runs")
    p.add_argument("--data", help="relocated canonical preprocessed Embryonic.h5ad; new campaigns only")
    p.add_argument("--edge-tsv", help="canonical TF-target edges; new campaigns only")
    p.add_argument("--device", default="auto")
    p.add_argument("--analysis-device", default="cpu")
    p.add_argument("--dry-run", action="store_true", help="print plan, create nothing, train nothing")
    return p


def campaign_name(value):
    if "/" in value:
        path = confined(value)
        if path.parent != SUITE / "runs":
            raise ValueError("resume path must be a campaign directly under suite/runs")
        value = path.name
    campaign_path(value)
    return value


@contextmanager
def campaign_lock(campaign):
    # Kernel releases the lock after a crash; never unlink the lock inode.
    import fcntl
    with (campaign / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another launcher owns this campaign") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def artifact_ok(record, key):
    path = confined(record["artifact"])
    if key == "EMA_CHECKPOINT":
        from ..training.checkpoints import read_checkpoint
        if file_hash(path) != record["sha256"]:
            raise ValueError("completed artifact changed")
        read_checkpoint(path)
    elif (not (path / "completed.json").exists() or (path / "failed.json").exists()):
        raise ValueError("recorded artifact is incomplete")
    return path


def run_step(campaign, name, argv, key):
    success = campaign / "steps" / name / "completed.json"
    if success.exists():
        return artifact_ok(read_json(success), key)
    attempt = new_dir(campaign / "steps" / name / run_id())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(SUITE / ".mplconfig"))
    command = [sys.executable, "-m", "work.20260915_x0predict.cli", *argv]
    write_json(attempt / "started.json", {"command": command})
    artifact = None
    try:
        with (attempt / "output.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                    if line.startswith(key + "="):
                        artifact = line.strip().split("=", 1)[1]
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
            finally:
                process.stdout.close()
        if code or artifact is None:
            raise RuntimeError(f"{name} failed (exit {code}); see {attempt / 'output.log'}")
        record = {"status": "completed", "artifact": artifact, "attempt": str(attempt)}
        if key == "EMA_CHECKPOINT":
            record["sha256"] = file_hash(artifact)
        result = artifact_ok(record, key)
        write_json(success, record)
        return result
    except BaseException as exc:
        write_json(attempt / "failed.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise


def execute(campaign, args, *, step_fn=run_step):
    errors = []
    selected = [args.condition] if args.condition else CONDITIONS
    common = ["--campaign", campaign.name, "--device", args.device]
    try:
        if args.stage2_only:
            from ..training.checkpoints import canonical_stage1
            _, record = canonical_stage1(campaign)
            stage1 = Path(record["checkpoint"])
        else:
            stage1 = step_fn(campaign, STAGE1 + "_train", ["train", *common, "--condition", STAGE1], "EMA_CHECKPOINT")
    except Exception as exc:
        return [{"step": "stage1", "error": str(exc)}]

    def analyze_condition(condition, checkpoint):
        try:
            trajectory = step_fn(campaign, condition + "_sample",
                                 ["sample", "--checkpoint", str(checkpoint), "--device", args.device], "TRAJECTORY_DIR")
        except Exception as exc:
            errors.append({"step": condition + "_sample", "error": str(exc)})
            return
        # Numeric and UMAP analyses can succeed independently.
        for action in ("analyze", "embed"):
            try:
                result = step_fn(campaign, condition + "_" + action,
                                 [action, "--trajectory", str(trajectory), "--device", args.analysis_device], "ANALYSIS_DIR")
                step_fn(campaign, condition + "_" + action + "_plot",
                        ["plot", "--input", str(result)], "FIGURE_DIR")
            except Exception as exc:
                errors.append({"step": condition + "_" + action, "error": str(exc)})

    analyze_condition(STAGE1, stage1)
    if not args.stage1_only:
        for condition in selected:
            try:
                checkpoint = step_fn(campaign, condition + "_train",
                                     ["train", *common, "--condition", condition], "EMA_CHECKPOINT")
            except Exception as exc:
                errors.append({"step": condition + "_train", "error": str(exc)})
                continue
            analyze_condition(condition, checkpoint)
    return errors


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage1_only and (args.condition or args.all_eight):
        raise ValueError("Stage1-only cannot select Stage2 conditions")
    if args.stage2_only and not args.resume_campaign:
        raise ValueError("Stage2-only requires --resume-campaign with its own final START_X Stage1")
    if args.resume_campaign and (args.data or args.edge_tsv):
        raise ValueError("resume uses immutable campaign data/edge paths")
    name = campaign_name(args.resume_campaign) if args.resume_campaign else "x0predict_" + run_id()
    campaign = campaign_path(name)
    selected = [] if args.stage1_only else ([args.condition] if args.condition else CONDITIONS)
    if args.dry_run:
        print(f"CAMPAIGN={campaign}\nStage1: {'reuse final EMA' if args.stage2_only else 'train/reuse START_X final EMA'}")
        print("Stage2: " + ", ".join(selected))
        print("Analysis: Stage1 and selected completed Stage2; native START_X DDPM, diagnostics, UMAP, plots")
        return 0
    import torch
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.resume_campaign:
        spec = read_json(campaign / "campaign.json")
        if spec["suite_version"] != "20260915_x0predict_v1":
            raise ValueError("resume requires a START_X campaign")
        for condition, expected in spec["config_sha256"].items():
            if file_hash(campaign / "configs" / f"{condition}.json") != expected:
                raise ValueError("immutable campaign config changed")
    else:
        configs = {}
        for condition in [STAGE1, *CONDITIONS]:
            config = effective_config(condition)
            config["output_campaign"] = name
            for argument, key in ((args.data, "data_dir"), (args.edge_tsv, "edge_tsv_path")):
                config[key] = str(Path(argument or config[key]).expanduser().resolve())
                if not Path(config[key]).is_file():
                    raise FileNotFoundError(f"{key}: {config[key]}; provide --data and --edge-tsv for this machine")
            configs[condition] = config
        new_dir(campaign)
        new_dir(campaign / "configs")
        for condition, config in configs.items():
            write_json(campaign / "configs" / f"{condition}.json", config)
        write_json(campaign / "source_sha256.json", source_provenance())
        write_json(campaign / "campaign.json", {
            "suite_version": "20260915_x0predict_v1", "git_commit": git_commit(), "campaign": name,
            "conditions": CONDITIONS, "predict_xstart": True,
            "config_sha256": {c: file_hash(campaign / "configs" / f"{c}.json") for c in configs},
        })
    print(f"CAMPAIGN={campaign}", flush=True)
    with campaign_lock(campaign):
        errors = execute(campaign, args)
        write_json(campaign / ("invocation_" + run_id() + ".json"), {"errors": errors, "selected": selected,
                    "status": "failed" if errors else "completed", "arguments": vars(args)})
    for error in errors:
        print(f"FAILED {error['step']}: {error['error']}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
