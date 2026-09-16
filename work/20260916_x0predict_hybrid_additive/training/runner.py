"""Stage2 only; indexed data stream and an immutable shared PCA transform."""
import csv
import hashlib
import json
from pathlib import Path
import torch
from ..common import (STAGE1, build_diffusion, campaign_path, file_hash, finite, git_commit,
                      load_real, new_dir, read_json, run_id, seed_all, source_provenance,
                      state_hash, umap_core, write_json)
from ..models import build_model, freeze_from_stage1, assert_frozen, optimizer_for, update_ema
from ..data import source_batches, DisjointTargets, fit_pca, load_pca
from ..analysis.diagnostics import BranchRecorder
from ..reuse import load
from .checkpoints import canonical_stage1, read_checkpoint, save_checkpoint
from .objectives import timestep_sampler, training_loss

# Reuse recovery/completion verification without the old source loader/train loop.
recovery = load("training._recovery", "training/runner.py")


def pca_for_campaign(campaign, matrix, config, origin):
    marker = campaign / "pca.json"
    if not marker.exists():
        path = fit_pca(matrix, config["pca"], campaign / "pca" / run_id(),
                       dict(data_sha256=origin["data_sha256"], gene_order_hash=origin["gene_order_hash"]))
        write_json(marker, dict(path=str(path)))
    path = Path(read_json(marker)["path"])
    if not path.resolve().is_relative_to(campaign / "pca"):
        raise ValueError("PCA pointer escaped campaign")
    module, real, info = load_pca(path, data_sha256=origin["data_sha256"], gene_order_hash=origin["gene_order_hash"])
    if info["config"] != config["pca"]:
        raise ValueError("PCA config changed")
    return module, real, dict(path=str(path), **info)


def train(args):
    campaign = campaign_path(args.campaign)
    config = read_json(campaign / "configs" / f"{args.condition}.json")
    if config["objective"] == "stage1":
        raise ValueError("this campaign never trains Stage1")
    source = source_provenance()
    if source != read_json(campaign / "source_sha256.json"):
        raise ValueError("campaign source changed; create a new campaign")
    done = recovery.completed_training(campaign, args.condition)
    if done:
        print(f"EMA_CHECKPOINT={done['checkpoint']}")
        return Path(done["checkpoint"])
    seed_all(config["seed"])
    stage1, origin = canonical_stage1(campaign)
    if file_hash(config["data_dir"]) != origin["data_sha256"] or file_hash(config["edge_tsv_path"]) != origin["edge_tsv_sha256"]:
        raise ValueError("Stage1 data/edge hash mismatch")
    data, genes, _ = load_real(config, stage1["metadata"]["gene_names"])
    if umap_core().gene_order_hash(genes) != origin["gene_order_hash"]:
        raise ValueError("gene hash mismatch")
    pca, real_pca, pca_meta = None, None, None
    if config["objective"] == "ot":
        if data.n_obs < config["batch_size"] + config["target_size"]:
            raise ValueError(f"need source batch + {config['target_size']} distinct target cells; no size reduction allowed")
        pca, real_pca, pca_meta = pca_for_campaign(campaign, data.X, config, origin)
        pca = pca.to(args.device)
    bundle = recovery.latest_bundle(campaign, args.condition)
    raw = read_checkpoint(bundle["raw"]) if bundle else None
    if raw and (raw["metadata"]["effective_config"] != config or raw["metadata"]["originating_stage1"] != origin):
        raise ValueError("resume config/Stage1 mismatch")
    model = build_model(config, genes, state=raw["state_dict"] if raw else None)
    frozen_hash = freeze_from_stage1(model, stage1["state_dict"])
    del stage1, raw
    model.to(args.device).train()
    optimizer = optimizer_for(model, config)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    start = 0
    if bundle:
        ema = {k: v.to(args.device) for k, v in read_checkpoint(bundle["ema"])["state_dict"].items()}
        optimizer.load_state_dict(torch.load(bundle["optimizer"], map_location=args.device, weights_only=True))
        start = bundle["step"]
    assert_frozen(model, optimizer, frozen_hash)
    seed_all(config["seed"])  # matched diffusion RNG, independent of field parameter count
    batches = source_batches(data.X, config["batch_size"], config["source_seed"])
    targets = DisjointTargets(data.n_obs, config["target_size"], config["target_refresh_interval"], config["target_seed"]) if pca else None
    diffusion = build_diffusion(config)
    sampler = timestep_sampler(config, diffusion)
    run = new_dir(campaign / args.condition / run_id())
    checkpoints = new_dir(run / "checkpoints")
    metadata = dict(effective_config=config, gene_names=genes, gene_order_hash=origin["gene_order_hash"],
                    data_sha256=origin["data_sha256"], edge_tsv_sha256=origin["edge_tsv_sha256"],
                    originating_stage1=origin, frozen_cellunet_hash_before=frozen_hash,
                    model_mean_type="START_X", predict_xstart=True, git_commit=git_commit(), source_sha256=source,
                    pca_provenance=pca_meta, continuation=bundle,
                    resume_rng_policy="restart seeded indexed source/target/diffusion streams; restore optimizer/EMA/count")
    write_json(run / "metadata.json", metadata)
    write_json(run / "effective_config.json", config)
    recorder = BranchRecorder()
    final_ema = Path(bundle["ema"]) if bundle else None
    try:
        with (run / "losses.csv").open("x", newline="") as f, (run / "index_audit.jsonl").open("x") as audit:
            writer = csv.DictWriter(f, fieldnames=["step", "primary", "soft", "total", "sinkhorn"])
            writer.writeheader()
            for index in range(start, config["total_steps"]):
                x0, ids = next(batches)
                x0 = x0.to(args.device)
                t, weights = sampler.sample(len(x0), args.device)
                target = None
                if targets:
                    target_ids, detail = targets.sample(ids, index)
                    target = torch.from_numpy(real_pca[target_ids].copy()).to(args.device)
                    audit.write(json.dumps(dict(step=index, **detail, source_indices=ids.tolist(),
                                               target_sha256=hashlib.sha256(target_ids.tobytes()).hexdigest())) + "\n")
                optimizer.zero_grad(set_to_none=True)
                model.diagnostic_sink = recorder if (index == 0 or (index + 1) % config["log_interval"] == 0) else None
                loss, values = training_loss(model, diffusion, x0, t, weights, config, pca=pca, target=target)
                model.diagnostic_sink = None
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        finite(name + " gradient", parameter.grad)
                optimizer.step()
                update_ema(ema, model, float(config["ema_rate"]))
                for group in optimizer.param_groups:
                    group["lr"] = config["lr"] * (1 - index / config["lr_anneal_steps"])
                step = index + 1
                writer.writerow(dict(step=step, **{**values, "sinkhorn": json.dumps(values["sinkhorn"])}))
                if step % config["log_interval"] == 0:
                    f.flush(); audit.flush()
                    print(f"{args.condition} step={step} loss={values['total']:.6g}", flush=True)
                if step % config["save_interval"] == 0 or step == config["total_steps"]:
                    assert_frozen(model, optimizer, frozen_hash)
                    if state_hash({k.removeprefix("ml_model."): v for k, v in ema.items() if k.startswith("ml_model.")}) != frozen_hash:
                        raise AssertionError("EMA changed frozen CellUNet")
                    for name, tensor in ema.items():
                        finite(name, tensor)
                    meta = dict(metadata, step=step, frozen_cellunet_hash_after=frozen_hash)
                    raw_path = save_checkpoint(checkpoints / f"model{step:06d}.pt", model.state_dict(), dict(meta, checkpoint_kind="raw"))
                    final_ema = save_checkpoint(checkpoints / f"ema_{config['ema_rate']}_{step:06d}.pt", ema, dict(meta, checkpoint_kind="ema"))
                    opt = checkpoints / f"opt{step:06d}.pt"
                    with opt.open("xb") as handle:
                        torch.save(optimizer.state_dict(), handle)
                    record = dict(step=step)
                    for key, path in (("raw", raw_path), ("ema", final_ema), ("optimizer", opt)):
                        record[key], record[key + "_sha256"] = str(path), file_hash(path)
                    write_json(checkpoints / f"bundle{step:06d}.json", record)
        if recorder.rows:
            recorder.save(run, "training")
        write_json(run / "completed.json", dict(status="completed", checkpoint=str(final_ema),
                   checkpoint_sha256=file_hash(final_ema), step=config["total_steps"]))
    except BaseException as exc:
        write_json(run / "failed.json", dict(error=f"{type(exc).__name__}: {exc}"))
        raise
    print(f"EMA_CHECKPOINT={final_ema}")
    return final_ema
