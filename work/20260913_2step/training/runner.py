"""Single-device float32-parameter training; source optimizer and noise conventions."""

import csv
import json
from pathlib import Path
import torch
from ..common import (
    SUITE,
    build_diffusion,
    confined,
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


def train(args):
    config = effective_config(args.condition)
    campaign = confined(SUITE / "runs" / args.campaign)
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
            if config["objective"] != "ot":
                raise ValueError("OT training overrides apply only to OT conditions")
            if bundle and (key != "max_iterations" or value < config["ot"][key]):
                raise ValueError(
                    "continuation may only increase the Sinkhorn iteration cap"
                )
            config["ot"][key] = value
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
    try:
        data, genes, _ = load_real(config)
        if data.n_obs < config["batch_size"]:
            raise ValueError("drop_last source loader requires at least one full batch")
        gene_hash = umap_core().gene_order_hash(genes)
        if origin is not None:
            if gene_hash != origin["gene_order_hash"]:
                raise ValueError("Stage-2 genes differ from Stage 1")
            if file_hash(config["edge_tsv_path"]) != origin["edge_tsv_sha256"]:
                raise ValueError("campaign TF-target edge file changed")
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
            "effective_config": config,
            "gene_names": genes,
            "gene_order_hash": gene_hash,
            "seed": config["seed"],
            "git_commit": git_commit(),
            "device": str(device),
            "originating_stage1": origin,
            "frozen_cellunet_hash_before": frozen_hash,
            "edge_tsv_sha256": file_hash(config["edge_tsv_path"]),
            "preprocessing": "load_data(train_vae=True, preprocess=False, layer=None); unchanged X and gene ordering",
        }
        with (run / "losses.csv").open("x", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "primary", "soft", "total", "sinkhorn"]
            )
            writer.writeheader()
            for index in range(start_step, config["total_steps"]):
                batch, _ = next(batches)  # unconditional; source CellUNet ignores y
                batch = finite("training batch", batch.to(device))
                t, weights = sampler.sample(len(batch), device)
                if frozen_hash is not None:
                    assert_frozen(
                        model, opt
                    )  # inexpensive mode/optimizer checks every update
                opt.zero_grad(set_to_none=True)
                loss, values = training_loss(
                    model, diffusion, batch, t, weights, config
                )
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        finite(f"gradient {name}", parameter.grad)
                opt.step()
                update_ema(ema, model, float(config["ema_rate"]))
                # Source TrainLoop anneals AFTER optimize, using zero-based index.
                for group in opt.param_groups:
                    group["lr"] = config["lr"] * (1 - index / config["lr_anneal_steps"])
                step = index + 1
                values["sinkhorn"] = json.dumps(values["sinkhorn"])
                writer.writerow({"step": step, **values})
                if step % config["log_interval"] == 0:
                    f.flush()
                    print(
                        f"{args.condition} step={step} loss={values['total']:.6g}",
                        flush=True,
                    )
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
    return run
