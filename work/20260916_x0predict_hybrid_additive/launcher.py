"""Prepare once, freeze the supplied EMA, then run eight independent conditions."""
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
from .common import (SUITE, ROOT, VERSION, STAGE1, CONDITIONS, campaign_path, confined,
                     effective_config, file_hash, git_commit, load_real, new_dir, read_json,
                     run_id, source_provenance, write_json, training_config, condition_step_prefix, GRADIENT_MODES)
from .training.checkpoints import load_stage1, save_checkpoint, canonical_stage1

campaign_lock = importlib.import_module("work.20260915_x0predict.scripts.run_all").campaign_lock


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--condition", choices=CONDITIONS)
    group.add_argument("--all-eight", action="store_true")
    p.add_argument("--resume-campaign")
    p.add_argument("--stage1-checkpoint", default=None)
    p.add_argument("--data")
    p.add_argument("--edge-tsv")
    p.add_argument("--pca-dimension", type=int)
    p.add_argument("--target-size", type=int)
    p.add_argument("--target-refresh-interval", type=int)
    p.add_argument("--training-steps", type=int, help="Stage2 horizon (default 10000); may shorten an existing campaign")
    p.add_argument("--sinkhorn-gradient-mode", choices=GRADIENT_MODES)
    p.add_argument("--device", default="cuda")
    p.add_argument("--analysis-device", default="cpu")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--preflight-only", action="store_true", help="load/validate actual checkpoint and input hashes, no training")
    return p


def inputs(args):
    default = effective_config(STAGE1)
    payload, origin = load_stage1(args.stage1_checkpoint or default["stage1_checkpoint"])
    first = payload["metadata"]["effective_config"]
    data = str(Path(args.data or first["data_dir"]).expanduser().resolve())
    edge = str(Path(args.edge_tsv or first["edge_tsv_path"]).expanduser().resolve())
    if file_hash(data) != origin["data_sha256"] or file_hash(edge) != origin["edge_tsv_sha256"]:
        raise ValueError("data/edges differ from the pretrained Stage1")
    print(f"STAGE1_SHA256={origin['checkpoint_sha256']}\nDATA={data}\nEDGE_TSV={edge}", flush=True)
    return payload, origin, data, edge


def prepare(args):
    payload, origin, data, edge = inputs(args)
    first = payload["metadata"]["effective_config"]
    campaign = campaign_path("additive_" + run_id())
    configs = {}
    for condition in [STAGE1, *CONDITIONS]:
        config = effective_config(condition)
        for key in ("cell_unet_hidden_num", "diffusion_steps", "noise_schedule", "timestep_respacing",
                    "learn_sigma", "use_kl", "predict_xstart", "rescale_timesteps", "rescale_learned_sigmas", "ts_layer"):
            config[key] = first[key]
        config.update(data_dir=data, edge_tsv_path=edge, output_campaign=campaign.name,
                      stage1_checkpoint=origin["checkpoint"])
        if condition != STAGE1:
            config = training_config(config, args.training_steps, gradient_mode=args.sinkhorn_gradient_mode)
        for argument, key in ((args.target_size, "target_size"), (args.target_refresh_interval, "target_refresh_interval")):
            if argument is not None:
                config[key] = argument
        if args.pca_dimension is not None:
            config["pca"]["dimension"] = args.pca_dimension
        if min(config["target_size"], config["target_refresh_interval"], config["pca"]["dimension"]) < 1:
            raise ValueError("positive target/PCA settings required")
        configs[condition] = config
    # Validate row capacity and gene order before creating any campaign artifacts.
    actual, genes, _ = load_real(configs[STAGE1], payload["metadata"]["gene_names"])
    if actual.n_obs < configs[STAGE1]["target_size"] + configs[STAGE1]["batch_size"]:
        raise ValueError("insufficient distinct training rows for source + target; refusing replacement duplicates")
    if configs[STAGE1]["pca"]["dimension"] > min(actual.shape):
        raise ValueError("PCA dimension exceeds dataset dimensions")
    del actual
    new_dir(campaign)
    try:
        new_dir(campaign / "configs")
        for condition, config in configs.items():
            write_json(campaign / "configs" / f"{condition}.json", config)
        new_dir(campaign / "stage1")
        meta = dict(payload["metadata"], effective_config=configs[STAGE1],
                    originating_stage1=origin, model_mean_type="START_X", predict_xstart=True)
        imported = save_checkpoint(campaign / "stage1/ema_0.9999_030000.pt", payload["state_dict"], meta)
        write_json(campaign / "canonical_stage1.json", dict(origin, source_checkpoint=origin["checkpoint"],
                   source_checkpoint_sha256=origin["checkpoint_sha256"], checkpoint=str(imported),
                   checkpoint_sha256=file_hash(imported)))
        canonical_stage1(campaign)
        write_json(campaign / "source_sha256.json", source_provenance())
        write_json(campaign / "campaign.json", dict(suite_version=VERSION, git_commit=git_commit(),
                   stage1=origin, conditions=CONDITIONS, seed=configs[STAGE1]["seed"],
                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   config_sha256={c: file_hash(campaign / "configs" / f"{c}.json") for c in configs}))
    except BaseException as exc:
        write_json(campaign / "failed_preparation.json", dict(error=str(exc)))
        raise
    return campaign


def step(campaign, name, argv, key):
    marker = campaign / "steps" / name / "completed.json"
    if marker.exists():
        record = read_json(marker)
        path = confined(record["artifact"])
        if key == "EMA_CHECKPOINT":
            if file_hash(path) != record["sha256"]:
                raise ValueError("completed checkpoint hash mismatch")
        elif not (path / "completed.json").exists() or (path / "failed.json").exists():
            raise ValueError("completed artifact is no longer valid")
        return path
    attempt = new_dir(campaign / "steps" / name / run_id())
    command = [sys.executable, "-m", __package__ + ".cli", *argv]
    write_json(attempt / "started.json", dict(command=command))
    artifact = None
    try:
        with (attempt / "output.log").open("x") as log:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(SUITE / ".mplconfig"))
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line); log.flush()
                    print(line, end="", flush=True)
                    if line.startswith(key + "="):
                        artifact = confined(line.strip().split("=", 1)[1])
                code = process.wait()
            except BaseException:
                process.terminate(); process.wait()
                raise
            finally:
                process.stdout.close()
        if code or artifact is None:
            raise RuntimeError(f"exit {code}: {attempt / 'output.log'}")
        if key != "EMA_CHECKPOINT" and not (artifact / "completed.json").exists():
            raise ValueError("worker did not complete artifact")
        write_json(marker, dict(status="completed", artifact=str(artifact),
                   sha256=file_hash(artifact) if artifact.is_file() else None, attempt=str(attempt)))
        return artifact
    except BaseException as exc:
        write_json(attempt / "failed.json", dict(error=str(exc)))
        raise


def execute(campaign, args, step_fn=step):
    errors = []
    _, origin = canonical_stage1(campaign)
    def evaluate(condition, checkpoint):
        prefix = condition if condition == STAGE1 else condition_step_prefix(
            campaign, condition, args.training_steps, target_size=args.target_size, gradient_mode=args.sinkhorn_gradient_mode)
        try:
            sampled = step_fn(campaign, prefix + "_sample", ["sample", "--checkpoint", str(checkpoint), "--device", args.device], "TRAJECTORY_DIR")
        except Exception as exc:
            errors.append(dict(step=condition + "_sample", error=str(exc)))
            return
        for action in ("analyze", "embed"):
            try:
                output = step_fn(campaign, prefix + "_" + action, [action, "--trajectory", str(sampled), "--device", args.analysis_device], "ANALYSIS_DIR")
                step_fn(campaign, prefix + "_" + action + "_plot", ["plot", "--input", str(output)], "FIGURE_DIR")
            except Exception as exc:
                errors.append(dict(step=condition + "_" + action, error=str(exc)))
    evaluate(STAGE1, origin["checkpoint"])
    for condition in [args.condition] if args.condition else CONDITIONS:
        try:
            prefix = condition_step_prefix(campaign, condition, args.training_steps,
                                           target_size=args.target_size, gradient_mode=args.sinkhorn_gradient_mode)
            options = [] if args.training_steps is None else ["--training-steps", str(args.training_steps)]
            if args.target_size is not None:
                options += ["--target-size", str(args.target_size)]
            if args.sinkhorn_gradient_mode is not None:
                options += ["--sinkhorn-gradient-mode", args.sinkhorn_gradient_mode]
            checkpoint = step_fn(campaign, prefix + "_train", ["train", "--campaign", campaign.name,
                                  "--condition", condition, "--device", args.device, *options], "EMA_CHECKPOINT")
            evaluate(condition, checkpoint)
        except Exception as exc:
            errors.append(dict(step=condition + "_train", error=str(exc)))
    from .analysis.comparison import compare
    compare(campaign, training_steps=args.training_steps, target_size=args.target_size,
            gradient_mode=args.sinkhorn_gradient_mode)
    return errors


def main(argv=None):
    args = parser().parse_args(argv)
    if args.dry_run:
        print("Stage1 (read-only):", args.stage1_checkpoint or effective_config(STAGE1)["stage1_checkpoint"])
        print("Conditions:", [args.condition] if args.condition else CONDITIONS)
        print("Stage2 training steps:", args.training_steps or 10000)
        print("OT override:", dict(target_size=args.target_size, gradient_mode=args.sinkhorn_gradient_mode))
        print("CellUNet-only baseline + native START_X additive sampling/analysis; no work executed")
        return 0
    if args.preflight_only:
        inputs(args)
        return 0
    if args.resume_campaign:
        if any(x is not None for x in (args.stage1_checkpoint, args.data, args.edge_tsv, args.target_refresh_interval, args.pca_dimension)):
            raise ValueError("resume uses immutable campaign settings")
        campaign = campaign_path(args.resume_campaign)
        manifest = read_json(campaign / "campaign.json")
        if manifest["suite_version"] != VERSION:
            raise ValueError("wrong campaign version")
        for condition, expected in manifest["config_sha256"].items():
            if file_hash(campaign / "configs" / f"{condition}.json") != expected:
                raise ValueError("immutable config changed")
    else:
        campaign = prepare(args)
    print(f"CAMPAIGN={campaign}", flush=True)
    order = [args.condition] if args.condition else CONDITIONS
    print("CONDITION_ORDER=" + ",".join(order), flush=True)
    with campaign_lock(campaign):
        errors = execute(campaign, args)
        write_json(campaign / ("invocation_" + run_id() + ".json"), dict(errors=errors, arguments=vars(args),
                   condition_order=order, status="failed" if errors else "completed"))
    for error in errors:
        print(error, file=sys.stderr)
    return int(bool(errors))
