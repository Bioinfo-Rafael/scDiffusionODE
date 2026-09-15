"""Single-device float32-parameter training; source optimizer and noise conventions."""

import csv
import json
from pathlib import Path
import torch
from ..common import (
    SUITE,
    build_diffusion,
    confined,
    campaign_config,
    effective_config,
    file_hash,
    finite,
    git_commit,
    load_real,
    new_dir,
    run_id,
    seed_all,
    state_hash,
    umap_core,
    write_json,
)
from ..models import (
    assert_frozen,
    build_model,
    freeze_from_stage1,
    optimizer_for,
    update_ema,
)
from .checkpoints import canonical_stage1, save_checkpoint
from .objectives import timestep_sampler, training_loss
from .recovery import load_bundle, resume_config, restore_training_state
from .trajectory import trajectory_loss, independent_batch, validate_trajectory
from .source_cache import load_cache, cache_pointer
from ..common import read_json
import math
import time


def train(args):
    campaign = confined(SUITE / "runs" / args.campaign)
    config = (
        campaign_config(campaign, args.condition)
        if (campaign / "experiment.json").exists()
        else effective_config(args.condition)
    )
    if config.get("experiment_mode"):
        config["output_campaign"] = args.campaign
    origin, stage1_state = None, None
    bundle = None
    resume_path = getattr(args, "resume_checkpoint", None)
    if resume_path and config["objective"] == "stage1":
        raise ValueError("resume requires a Stage-2 condition")
    if config["objective"] == "stage1":
        if args.data:
            config["data_dir"] = str(Path(args.data).expanduser().resolve())
        if args.edge_tsv:
            config["edge_tsv_path"] = str(Path(args.edge_tsv).expanduser().resolve())
        new_dir(campaign)  # one common Stage 1 per campaign; no implicit retrain/resume
    else:
        if args.data or args.edge_tsv:
            raise ValueError(
                "Stage 2 inherits data/edge paths from its canonical Stage 1"
            )
        payload, origin = canonical_stage1(campaign)
        stage1_state = payload["state_dict"]
        first = payload["metadata"]["effective_config"]
        shared = (
            "cell_unet_hidden_num",
            "diffusion_steps",
            "noise_schedule",
            "learn_sigma",
            "use_kl",
            "predict_xstart",
            "rescale_timesteps",
            "rescale_learned_sigmas",
            "timestep_respacing",
            "ts_layer",
        )
        if any(config[key] != first[key] for key in shared):
            raise ValueError(
                "Stage-2 CellUNet/diffusion/representation differs from canonical Stage 1"
            )
        for key in ("data_dir", "edge_tsv_path"):
            config[key] = first[key]
    if resume_path:
        bundle = load_bundle(resume_path, campaign, args.condition, origin)
        config = resume_config(bundle, config)
    if (
        config["objective"] == "ot"
        and config["ot"]["batch_size"] != config["batch_size"]
    ):
        raise ValueError("OT batch_size metadata must equal actual batch_size")
    # Expose solver settings without allowing a different diffusion/model architecture.
    for key in ("epsilon", "max_iterations", "tolerance"):
        value = getattr(args, "ot_" + key)
        if value is not None:
            if config["objective"] not in ("ot", "trajectory_ot"):
                raise ValueError("OT training overrides apply only to OT conditions")
            if bundle and (key != "max_iterations" or value < config["ot"][key]):
                raise ValueError(
                    "continuation may only increase the Sinkhorn iteration cap"
                )
            config["ot"][key] = value
    is_trajectory = config["objective"] == "trajectory_ot"
    source_states, source_metadata = None, None
    overrides = {
        "trajectory_batch_size": "trajectory_batch_size",
        "trajectory_ode_steps": "ode_steps",
        "trajectory_dt": "ode_dt",
        "trajectory_samples_per_path": "samples_per_trajectory",
        "real_ot_points": "real_target_points",
        "trajectory_checkpoint_block": "checkpoint_block",
        "gradient_diagnostic_interval": "gradient_diagnostic_interval",
    }
    for arg, key in overrides.items():
        value = getattr(args, arg, None)
        if value is not None:
            if not is_trajectory or bundle:
                raise ValueError(
                    "trajectory overrides require fresh trajectory OT training"
                )
            config["trajectory_ot"][key] = value
            if key == "samples_per_trajectory":
                config["trajectory_ot"]["temporal_bins"] = value
    if is_trajectory:
        runtime = dict(
            read_json(SUITE / "configs/trajectory_defaults.json")["trajectory_runtime"]
        )
        runtime["field_backend"] = (
            getattr(args, "trajectory_field_backend", None) or runtime["field_backend"]
        )
        for key in ("log_interval", "save_interval"):
            value = getattr(args, "trajectory_" + key, None)
            if value is not None:
                if value < 1:
                    raise ValueError(
                        "trajectory logging/save intervals must be positive"
                    )
                runtime[key] = value
        config["trajectory_runtime"] = runtime
        config["log_interval"] = runtime["log_interval"]
        config["save_interval"] = runtime["save_interval"]
        c = config["trajectory_ot"]
        validate_trajectory(c)
        cache_path = (
            getattr(args, "source_cache", None)
            or read_json(cache_pointer(campaign, "train"))["path"]
        )
        source_states, source_metadata = load_cache(cache_path, origin, role="train")
        canonical_cache = read_json(cache_pointer(campaign, "train"))["path"]
        if Path(cache_path).resolve() != Path(canonical_cache).resolve():
            raise ValueError(
                "all trajectory families must use the same canonical training cache"
            )
        if source_metadata["start_diffusion_t"] != c["start_diffusion_t"]:
            raise ValueError(
                "source cache diffusion state does not match training configuration"
            )
        if (
            bundle
            and bundle["raw"]["metadata"].get("trajectory_source_cache")
            != source_metadata
        ):
            raise ValueError("resume source cache provenance mismatch")
        c["source_cache_size"] = len(source_states)
        c["source_seed"] = source_metadata["seed"]
        config["ot"]["batch_size"] = (
            c["trajectory_batch_size"] * c["samples_per_trajectory"]
        )
    elif getattr(args, "source_cache", None):
        raise ValueError("source-cache applies only to trajectory OT")
    run = new_dir(campaign / args.condition / run_id())
    seed_all(config["seed"])
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    write_json(run / "effective_config.json", config)
    write_json(
        run / "started.json",
        {
            "git_commit": git_commit(),
            "device": str(device),
            "condition": args.condition,
        },
    )
    scale_file = None
    try:
        data, genes, _ = load_real(config)
        if not is_trajectory and data.n_obs < config["batch_size"]:
            raise ValueError("drop_last source loader requires at least one full batch")
        if is_trajectory and source_metadata["gene_names"] != genes:
            raise ValueError("source cache gene order differs from training data")
        gene_hash = umap_core().gene_order_hash(genes)
        if origin is not None:
            if gene_hash != origin["gene_order_hash"]:
                raise ValueError("Stage-2 genes differ from Stage 1")
            if file_hash(config["edge_tsv_path"]) != origin["edge_tsv_sha256"]:
                raise ValueError("campaign TF-target edge file changed")
        real_matrix = data.X if is_trajectory else None
        del data
        diffusion = build_diffusion(config)
        if bundle and genes != bundle["raw"]["metadata"]["gene_names"]:
            raise ValueError("resume genes differ from checkpoint")
        model = build_model(
            config, genes, state=bundle["raw"]["state_dict"] if bundle else None
        )
        frozen_hash = (
            freeze_from_stage1(model, stage1_state)
            if stage1_state is not None
            else None
        )
        del stage1_state
        model.to(device).train()
        opt = optimizer_for(model, config)
        ema = (
            restore_training_state(bundle, model, opt, frozen_hash)
            if bundle
            else {k: v.detach().clone() for k, v in model.state_dict().items()}
        )
        start_step = bundle["provenance"]["step"] if bundle else 0
        continuation = bundle["provenance"] if bundle else None
        if continuation:
            print(
                f"CONTINUING_FROM_STEP={start_step}; RNG/data stream restarts from seed",
                flush=True,
            )
        del bundle
        if not is_trajectory:
            from guided_diffusion.cell_datasets_loader import load_data

            batches = load_data(
                data_dir=config["data_dir"],
                batch_size=config["batch_size"],
                train_vae=True,
                preprocess=False,
                layer=config["ts_layer"],
            )
            sampler = timestep_sampler(config, diffusion)
        checkpoints = new_dir(run / "checkpoints")
        metadata = {
            "continuation": continuation,
            **(
                {
                    "objective": "trajectory_ot",
                    "objective_schema_version": config["objective_schema_version"],
                    "trajectory_source_cache": source_metadata,
                    "target_sampling": dict(
                        source="full empirical X, unchanged float32 source representation",
                        seed=config["trajectory_ot"]["target_sampler_seed"],
                        replacement=real_matrix.shape[0]
                        < config["trajectory_ot"]["real_target_points"],
                        real_count=real_matrix.shape[0],
                        independent_from="Gaussian-generated cache; separate per-step RNG",
                    ),
                }
                if is_trajectory
                else {}
            ),
            "effective_config": config,
            "gene_names": genes,
            "gene_order_hash": gene_hash,
            "seed": config["seed"],
            "git_commit": git_commit(),
            "device": str(device),
            "originating_stage1": origin,
            "frozen_cellunet_hash_before": frozen_hash,
            "edge_tsv_sha256": file_hash(config["edge_tsv_path"]),
            "preprocessing": (
                "independent X -> float32 sampler; same representation as load_data(train_vae=True, preprocess=False, layer=None); unchanged gene ordering"
                if is_trajectory
                else "load_data(train_vae=True, preprocess=False, layer=None); unchanged X and gene ordering"
            ),
        }
        scale_file = (
            (run / "sinkhorn_scales.csv").open("x", newline="")
            if is_trajectory
            else None
        )
        scale_writer = (
            csv.DictWriter(
                scale_file,
                fieldnames=[
                    "step",
                    "term",
                    "epsilon",
                    "iterations",
                    "marginal_residual",
                ],
            )
            if scale_file
            else None
        )
        if scale_writer:
            scale_writer.writeheader()
        with (run / "losses.csv").open("x", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=(
                    [
                        "step",
                        "primary",
                        "trajectory_ot",
                        "soft",
                        "total",
                        "grad_ot",
                        "grad_soft",
                        "grad_ratio_ot_to_soft",
                        "parameter_norm",
                        "update_norm",
                        "epsilon_scaling",
                    ]
                    + [
                        f"sinkhorn_{term}_{metric}"
                        for term in ("cross", "pred_self", "real_self")
                        for metric in ("iterations", "residual")
                    ]
                    + [
                        "cost_mean",
                        "cost_std",
                        "cost_median",
                        "cost_q90",
                        "cost_q99",
                        "cost_max",
                        "cost_cv",
                        "cost_median_over_epsilon",
                        "cost_max_over_epsilon",
                        "trajectory_state_norm_mean",
                        "trajectory_state_norm_max",
                        "trajectory_displacement_mean",
                    ]
                    if is_trajectory
                    else ["step", "primary", "soft", "total", "sinkhorn"]
                ),
            )
            writer.writeheader()
            last_log_time, last_log_step = time.monotonic(), start_step
            for index in range(start_step, config["total_steps"]):
                if not is_trajectory:
                    batch, _ = next(batches)
                    batch = finite("training batch", batch.to(device))
                    t, weights = sampler.sample(len(batch), device)
                if frozen_hash is not None:
                    assert_frozen(
                        model, opt
                    )  # inexpensive mode/optimizer checks every update
                opt.zero_grad(set_to_none=True)
                diagnostic, before = False, None
                if is_trajectory:
                    c = config["trajectory_ot"]
                    source, _ = independent_batch(
                        source_states,
                        c["trajectory_batch_size"],
                        c["source_sampler_seed"],
                        index + 1,
                    )
                    target, _ = independent_batch(
                        real_matrix,
                        c["real_target_points"],
                        c["target_sampler_seed"],
                        index + 1,
                    )
                    loss, values, info, diagnostic = trajectory_loss(
                        model,
                        source.to(device),
                        target.to(device),
                        config,
                        step=index + 1,
                    )
                    if diagnostic:
                        before = [
                            p.detach().clone()
                            for p in model.ode_model.parameters()
                            if p.requires_grad
                        ]
                        for term, detail in info.items():
                            for scale in detail["scales"]:
                                scale_writer.writerow(
                                    dict(step=index + 1, term=term, **scale)
                                )
                        scale_file.flush()
                else:
                    loss, values = training_loss(
                        model, diffusion, batch, t, weights, config
                    )
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        finite(f"gradient {name}", parameter.grad)
                opt.step()
                if diagnostic:
                    parameters = [
                        p for p in model.ode_model.parameters() if p.requires_grad
                    ]
                    values["update_norm"] = math.sqrt(
                        sum(
                            float((p.detach() - old).double().square().sum())
                            for p, old in zip(parameters, before)
                        )
                    )
                update_ema(ema, model, float(config["ema_rate"]))
                # Source TrainLoop anneals AFTER optimize, using zero-based index.
                for group in opt.param_groups:
                    group["lr"] = config["lr"] * (1 - index / config["lr_anneal_steps"])
                step = index + 1
                if not is_trajectory:
                    values["sinkhorn"] = json.dumps(values["sinkhorn"])
                writer.writerow({"step": step, **values})
                if step % config["log_interval"] == 0:
                    f.flush()
                    elapsed = time.monotonic() - last_log_time
                    seconds_per_step = elapsed / (step - last_log_step)
                    eta_hours = seconds_per_step * (config["total_steps"] - step) / 3600
                    print(
                        f"{args.condition} step={step} loss={values['total']:.6g}"
                        + (
                            f" seconds_per_step={seconds_per_step:.3f} eta_hours={eta_hours:.2f}"
                            if is_trajectory or config.get("experiment_mode")
                            else ""
                        ),
                        flush=True,
                    )
                    last_log_time, last_log_step = time.monotonic(), step
                if step % config["save_interval"] == 0 or step == config["total_steps"]:
                    for name, value in model.state_dict().items():
                        finite(name, value)
                    if frozen_hash is not None:
                        assert_frozen(model, opt, frozen_hash)
                        if (
                            state_hash(
                                {
                                    k.removeprefix("ml_model."): v
                                    for k, v in ema.items()
                                    if k.startswith("ml_model.")
                                }
                            )
                            != frozen_hash
                        ):
                            raise AssertionError("EMA mutated frozen CellUNet")
                    snapshot_meta = {
                        **metadata,
                        "step": step,
                        "frozen_cellunet_hash_after": frozen_hash,
                    }
                    save_checkpoint(
                        checkpoints / f"model{step:06d}.pt",
                        model.state_dict(),
                        {**snapshot_meta, "checkpoint_kind": "raw"},
                    )
                    final_ema = save_checkpoint(
                        checkpoints / f"ema_{config['ema_rate']}_{step:06d}.pt",
                        ema,
                        {**snapshot_meta, "checkpoint_kind": "ema"},
                    )
                    with (checkpoints / f"opt{step:06d}.pt").open("xb") as handle:
                        torch.save(opt.state_dict(), handle)
        if scale_file:
            scale_file.close()
        complete = {
            "status": "completed",
            "checkpoint": str(final_ema),
            "checkpoint_sha256": file_hash(final_ema),
            "step": step,
        }
        write_json(run / "completed.json", complete)
        if config["objective"] == "stage1":
            write_json(
                campaign / "canonical_stage1.json",
                {
                    **complete,
                    "cellunet_hash": state_hash(ema),
                    "gene_order_hash": gene_hash,
                    "edge_tsv_sha256": metadata["edge_tsv_sha256"],
                },
            )
        print(f"RUN_DIR={run}\nEMA_CHECKPOINT={final_ema}")
    except BaseException as exc:
        write_json(
            run / "failed.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    finally:
        if scale_file is not None:
            scale_file.close()
    return run
