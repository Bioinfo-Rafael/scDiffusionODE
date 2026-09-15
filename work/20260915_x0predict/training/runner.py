"""Canonical two-stage training with immutable attempts and recoverable bundles."""

import csv
import json
from pathlib import Path
import torch
from ..common import (
    STAGE1, campaign_path, confined, file_hash, finite, git_commit, load_real,
    new_dir, read_json, run_id, seed_all, source_provenance, state_hash,
    umap_core, write_json, build_diffusion,
)
from ..models import build_model, freeze_from_stage1, assert_frozen, optimizer_for, update_ema
from .checkpoints import canonical_stage1, save_checkpoint, read_checkpoint
from .objectives import timestep_sampler, training_loss


def completed_training(campaign, condition):
    records = sorted((campaign / condition).glob("*/completed.json"))
    if len(records) > 1:
        raise ValueError("multiple completed training attempts")
    if not records:
        return None
    record = read_json(records[0])
    checkpoint = confined(record["checkpoint"])
    if not checkpoint.is_relative_to(campaign / condition):
        raise ValueError("checkpoint belongs to another condition/campaign")
    if file_hash(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError("completed checkpoint hash changed")
    payload = read_checkpoint(checkpoint)
    meta = payload["metadata"]
    if (meta["step"] != meta["effective_config"]["total_steps"]
            or meta["checkpoint_kind"] != "ema"
            or meta["effective_config"]["condition"] != condition):
        raise ValueError("completed checkpoint is not the final condition EMA")
    return record


def latest_bundle(campaign, condition):
    candidates = []
    for marker in (campaign / condition).glob("*/checkpoints/bundle*.json"):
        record = read_json(marker)
        candidates.append((record["step"], str(marker), record))
    if not candidates:
        return None
    _, _, record = max(candidates)
    for key in ("raw", "ema", "optimizer"):
        path = confined(record[key])
        if not path.is_relative_to(campaign / condition):
            raise ValueError("resume bundle escaped condition")
        if file_hash(path) != record[key + "_sha256"]:
            raise ValueError("resume bundle hash changed")
    return record


def register_final_stage1(campaign, record):
    payload = read_checkpoint(record["checkpoint"])
    meta = payload["metadata"]
    canonical = {
        **record, "cellunet_hash": state_hash(payload["state_dict"]),
        "gene_order_hash": meta["gene_order_hash"],
        "edge_tsv_sha256": meta["edge_tsv_sha256"],
        "data_sha256": meta["data_sha256"],
    }
    marker = campaign / "canonical_stage1.json"
    if marker.exists():
        _, actual = canonical_stage1(campaign)
        if actual != canonical:
            raise ValueError("canonical Stage1 changed")
    else:
        write_json(marker, canonical)


def train(args):
    campaign = campaign_path(args.campaign)
    config = read_json(campaign / "configs" / f"{args.condition}.json")
    if config["objective"] == "ot" and config["ot"]["batch_size"] != config["batch_size"]:
        raise ValueError("OT batch-size metadata must equal the actual cell batch size")
    done = completed_training(campaign, args.condition)
    if done:
        if args.condition == STAGE1:
            register_final_stage1(campaign, done)
        print(f"EMA_CHECKPOINT={done['checkpoint']}")
        return Path(done["checkpoint"])
    source = source_provenance()
    if source != read_json(campaign / "source_sha256.json"):
        raise ValueError("source code changed since campaign creation; create a new campaign")
    seed_all(config["seed"])
    device = torch.device(args.device)
    origin, stage1_state = None, None
    if args.condition != STAGE1:
        payload, origin = canonical_stage1(campaign)
        stage1_state = payload["state_dict"]
        first = payload["metadata"]["effective_config"]
        shared = ("data_dir", "edge_tsv_path", "cell_unet_hidden_num", "diffusion_steps",
                  "noise_schedule", "predict_xstart", "timestep_respacing", "ts_layer")
        if any(config[key] != first[key] for key in shared):
            raise ValueError("Stage2 differs from its shared START_X Stage1")
    bundle = latest_bundle(campaign, args.condition)
    run = new_dir(campaign / args.condition / run_id())
    write_json(run / "effective_config.json", config)
    write_json(run / "started.json", {"git_commit": git_commit(), "resume_bundle": bundle})
    try:
        data, genes, _ = load_real(config)
        if data.n_obs < config["batch_size"]:
            raise ValueError("source drop_last loader requires at least one full batch")
        del data
        gene_hash = umap_core().gene_order_hash(genes)
        edge_hash, data_hash = file_hash(config["edge_tsv_path"]), file_hash(config["data_dir"])
        if origin and (gene_hash != origin["gene_order_hash"] or
                       edge_hash != origin["edge_tsv_sha256"] or data_hash != origin["data_sha256"]):
            raise ValueError("campaign dataset/gene order/edge file changed")
        raw = read_checkpoint(bundle["raw"]) if bundle else None
        if raw:
            old = raw["metadata"]
            if (old["effective_config"] != config or old["gene_names"] != genes or
                    old["data_sha256"] != data_hash or old["edge_tsv_sha256"] != edge_hash or
                    old["originating_stage1"] != origin):
                raise ValueError("resume provenance differs from campaign")
        diffusion = build_diffusion(config)
        model = build_model(config, genes, state=raw["state_dict"] if raw else None)
        frozen_hash = freeze_from_stage1(model, stage1_state) if origin else None
        del stage1_state, raw
        model.to(device).train()
        opt = optimizer_for(model, config)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        start = 0
        if bundle:
            restored = read_checkpoint(bundle["ema"])
            ema = {k: v.to(device) for k, v in restored["state_dict"].items()}
            opt.load_state_dict(torch.load(bundle["optimizer"], map_location=device, weights_only=True))
            start = bundle["step"]
        if frozen_hash:
            assert_frozen(model, opt, frozen_hash)
        metadata = {
            **{k: config[k] for k in ("ode_family", "ode_source_suite", "ode_source_file",
                 "ode_class", "ode_components", "expert_gating", "ode_source_sha256") if k in config},
            "effective_config": config, "gene_names": genes, "gene_order_hash": gene_hash,
            "seed": config["seed"], "git_commit": git_commit(), "device": str(device),
            "model_mean_type": diffusion.model_mean_type.name, "predict_xstart": True,
            "source_sha256": source, "originating_stage1": origin,
            "frozen_cellunet_hash_before": frozen_hash, "data_sha256": data_hash,
            "edge_tsv_sha256": edge_hash,
            "continuation": bundle,
            "resume_rng_policy": "restart seeded RNG/data stream, restore raw/EMA/optimizer and update count",
            "preprocessing": "load_data(train_vae=True, preprocess=False, layer=None); unchanged X/gene order",
        }
        write_json(run / "metadata.json", metadata)
        checkpoints = new_dir(run / "checkpoints")
        from guided_diffusion.cell_datasets_loader import load_data
        batches = load_data(data_dir=config["data_dir"], batch_size=config["batch_size"],
                            train_vae=True, preprocess=False, layer=config["ts_layer"])
        sampler = timestep_sampler(config, diffusion)
        final_ema = Path(bundle["ema"]) if bundle else None
        with (run / "losses.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "primary", "soft", "total", "sinkhorn"])
            writer.writeheader()
            for index in range(start, config["total_steps"]):
                batch, _ = next(batches)
                batch = batch.to(device)
                t, weights = sampler.sample(len(batch), device)
                opt.zero_grad(set_to_none=True)
                loss, values = training_loss(model, diffusion, batch, t, weights, config)
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        finite(f"gradient {name}", parameter.grad)
                opt.step()
                update_ema(ema, model, float(config["ema_rate"]))
                # Preserve source TrainLoop's post-update, zero-based annealing.
                for group in opt.param_groups:
                    group["lr"] = config["lr"] * (1 - index / config["lr_anneal_steps"])
                step = index + 1
                writer.writerow({"step": step, **values, "sinkhorn": json.dumps(values["sinkhorn"])})
                if step % config["log_interval"] == 0:
                    handle.flush()
                    print(f"{args.condition} step={step} loss={values['total']:.6g}", flush=True)
                if step % config["save_interval"] == 0 or step == config["total_steps"]:
                    for name, value in model.state_dict().items():
                        finite(name, value)
                    if frozen_hash:
                        assert_frozen(model, opt, frozen_hash)
                        if state_hash({k.removeprefix("ml_model."): v for k, v in ema.items()
                                       if k.startswith("ml_model.")}) != frozen_hash:
                            raise AssertionError("EMA mutated the frozen CellUNet")
                    snapshot = {**metadata, "step": step, "frozen_cellunet_hash_after": frozen_hash}
                    raw_path = save_checkpoint(checkpoints / f"model{step:06d}.pt", model.state_dict(),
                                               {**snapshot, "checkpoint_kind": "raw"})
                    final_ema = save_checkpoint(checkpoints / f"ema_{config['ema_rate']}_{step:06d}.pt",
                                                ema, {**snapshot, "checkpoint_kind": "ema"})
                    opt_path = checkpoints / f"opt{step:06d}.pt"
                    with opt_path.open("xb") as f:
                        torch.save(opt.state_dict(), f)
                    bundle_record = {"step": step}
                    for key, path in (("raw", raw_path), ("ema", final_ema), ("optimizer", opt_path)):
                        bundle_record[key] = str(path)
                        bundle_record[key + "_sha256"] = file_hash(path)
                    # Commit only complete bundles; interrupted writes remain immutable.
                    write_json(checkpoints / f"bundle{step:06d}.json", bundle_record)
        complete = {"status": "completed", "checkpoint": str(final_ema),
                    "checkpoint_sha256": file_hash(final_ema), "step": config["total_steps"]}
        write_json(run / "completed.json", complete)
        if args.condition == STAGE1:
            register_final_stage1(campaign, complete)
    except BaseException as exc:
        write_json(run / "failed.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    print(f"EMA_CHECKPOINT={final_ema}")
    return final_ema
