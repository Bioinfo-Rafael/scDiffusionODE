"""Stage2 only; indexed data stream and an immutable shared PCA transform."""
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from ..common import (STAGE1, SUITE, build_diffusion, campaign_path, file_hash, finite, git_commit,
                      load_real, new_dir, read_json, run_id, seed_all, source_provenance,
                      state_hash, umap_core, write_json, training_config, confined, same_training_config, ot_settings)
from ..models import build_model, freeze_from_stage1, assert_frozen, optimizer_for, update_ema
from ..data import source_batches, DisjointTargets, fit_pca, load_pca
from ..analysis.diagnostics import BranchRecorder
from .checkpoints import canonical_stage1, read_checkpoint, save_checkpoint
from .objectives import timestep_sampler, training_loss

def validate_resume_source(original, current):
    """Permit only hash-pinned versions of the reviewed Sinkhorn retry patch."""
    changed = {path: dict(before=original.get(path), after=current.get(path))
               for path in original.keys() | current.keys()
               if original.get(path) != current.get(path)}
    if not changed:
        return None
    policy = read_json(SUITE / "audit/resume_compatibility.json")
    for path, hashes in changed.items():
        allowed = policy["files"].get(path)
        if (allowed is None or hashes["before"] not in allowed["previous_sha256"]
                or hashes["after"] != allowed["current_sha256"]):
            raise ValueError(f"campaign source changed outside compatible Sinkhorn retry patch: {path}; create a new campaign")
    print(f"[resume] applying {policy['id']}; verified {len(changed)} source changes", flush=True)
    return dict(id=policy["id"], changed=changed,
                policy_sha256=file_hash(SUITE / "audit/resume_compatibility.json"))


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


def latest_bundle_at_or_before(campaign, condition, target, *, config=None, saved_config=None):
    candidates, originals = [], []
    for marker in (campaign / condition).glob("*/checkpoints/bundle*.json"):
        record = read_json(marker)
        if record["step"] <= target:
            item = (record["step"], str(marker), record)
            if config is None:
                candidates.append(item)
                continue
            raw_path = confined(record["raw"])
            if not raw_path.is_relative_to(campaign / condition):
                raise ValueError("resume bundle escaped condition")
            meta = read_json(raw_path.with_suffix(".json"))
            if same_training_config(meta["effective_config"], config):
                candidates.append(item)
            elif (saved_config is not None and record["step"] < target
                  and same_training_config(meta["effective_config"], saved_config)):
                originals.append(item)
    # Once the requested variant has a saved checkpoint, continue that branch.
    # A legacy final checkpoint cannot be relabelled as a newly trained variant.
    candidates = candidates or originals
    if not candidates:
        return None
    record = max(candidates)[2]
    for key in ("raw", "ema", "optimizer"):
        path = confined(record[key])
        if not path.is_relative_to(campaign / condition) or file_hash(path) != record[key + "_sha256"]:
            raise ValueError("resume bundle escaped condition or failed hash verification")
    return record


def completed_at_target(campaign, condition, config):
    for marker in sorted((campaign / condition).glob("*/completed.json"), reverse=True):
        record = read_json(marker)
        if record["step"] != config["total_steps"]:
            continue
        path = confined(record["checkpoint"])
        if not path.is_relative_to(campaign / condition) or file_hash(path) != record["checkpoint_sha256"]:
            raise ValueError("completed checkpoint changed/escaped condition")
        meta = read_checkpoint(path)["metadata"]
        if not same_training_config(meta["effective_config"], config):
            continue
        if (meta["step"] != config["total_steps"] or meta["effective_config"]["total_steps"] != config["total_steps"]
                or meta["checkpoint_kind"] != "ema"):
            raise ValueError("completed checkpoint metadata mismatch")
        return record
    return None


def train(args):
    campaign = campaign_path(args.campaign)
    saved_config = read_json(campaign / "configs" / f"{args.condition}.json")
    config = training_config(saved_config, getattr(args, "training_steps", None),
                             target_size=getattr(args, "target_size", None),
                             gradient_mode=getattr(args, "sinkhorn_gradient_mode", None))
    if config["objective"] == "stage1":
        raise ValueError("this campaign never trains Stage1")
    source = source_provenance()
    source_migration = validate_resume_source(read_json(campaign / "source_sha256.json"), source)
    done = completed_at_target(campaign, args.condition, config)
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
    bundle = latest_bundle_at_or_before(campaign, args.condition, config["total_steps"], config=config, saved_config=saved_config)
    raw = read_checkpoint(bundle["raw"]) if bundle else None
    if raw and (not any(same_training_config(raw["metadata"]["effective_config"], candidate)
                       for candidate in (config, saved_config)) or raw["metadata"]["originating_stage1"] != origin):
        raise ValueError("resume config/Stage1 mismatch")
    transitions = list(raw["metadata"].get("ot_transitions", [])) if raw else []
    if raw and config["objective"] == "ot" and ot_settings(raw["metadata"]["effective_config"]) != ot_settings(config):
        transition = dict(after_step=bundle["step"], first_new_update=bundle["step"]+1,
                          before=ot_settings(raw["metadata"]["effective_config"]), after=ot_settings(config),
                          source_checkpoint=bundle["raw"], source_checkpoint_sha256=bundle["raw_sha256"],
                          optimizer_policy="preserve AdamW moments and step, EMA and LR schedule")
        transitions.append(transition)
        print(f"[OT transition] {json.dumps(transition)}", flush=True)
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
                    pca_provenance=pca_meta, continuation=bundle, source_migration=source_migration,
                    saved_campaign_total_steps=saved_config["total_steps"],
                    ot_transitions=transitions,
                    resume_rng_policy="restart seeded indexed source/target/diffusion streams; restore optimizer/EMA/count")
    write_json(run / "metadata.json", metadata)
    write_json(run / "effective_config.json", config)
    recorder = BranchRecorder()
    final_ema = Path(bundle["ema"]) if bundle else None
    def checkpoint_at(step):
        assert_frozen(model, optimizer, frozen_hash)
        if state_hash({k.removeprefix("ml_model."): v for k, v in ema.items() if k.startswith("ml_model.")}) != frozen_hash:
            raise AssertionError("EMA changed frozen CellUNet")
        for name, tensor in ema.items():
            finite(name, tensor)
        meta = dict(metadata, step=step, frozen_cellunet_hash_after=frozen_hash)
        raw_path = save_checkpoint(checkpoints / f"model{step:06d}.pt", model.state_dict(), dict(meta, checkpoint_kind="raw"))
        ema_path = save_checkpoint(checkpoints / f"ema_{config['ema_rate']}_{step:06d}.pt", ema, dict(meta, checkpoint_kind="ema"))
        opt = checkpoints / f"opt{step:06d}.pt"
        with opt.open("xb") as handle:
            torch.save(optimizer.state_dict(), handle)
        record = dict(step=step)
        for key, path in (("raw", raw_path), ("ema", ema_path), ("optimizer", opt)):
            record[key], record[key + "_sha256"] = str(path), file_hash(path)
        write_json(checkpoints / f"bundle{step:06d}.json", record)
        return ema_path

    print(f"{args.condition} start_step={start} total_steps={config['total_steps']} git={metadata['git_commit']}", flush=True)
    if config["objective"] == "ot":
        print(f"[OT settings] {json.dumps(ot_settings(config))}", flush=True)
    log_started, logged_step = time.perf_counter(), start
    def clock(profile):
        if profile and torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(torch.device(args.device))
        return time.perf_counter()
    try:
        with (run / "losses.csv").open("x", newline="") as f, (run / "index_audit.jsonl").open("x") as audit, \
             (run / "timing.jsonl").open("x") as timing:
            writer = csv.DictWriter(f, fieldnames=["step", "timestamp_utc", "primary", "soft", "total", "sinkhorn"])
            writer.writeheader()
            if start == config["total_steps"]:
                # Repackage the exact intermediate EMA at the new horizon; no optimizer update.
                final_ema = checkpoint_at(start)
            for index in range(start, config["total_steps"]):
                profile = index - start < 3
                before_data = clock(profile)
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
                before_forward = clock(profile)
                loss, values = training_loss(model, diffusion, x0, t, weights, config, pca=pca, target=target)
                model.diagnostic_sink = None
                before_backward = clock(profile)
                loss.backward()
                before_update = clock(profile)
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        finite(name + " gradient", parameter.grad)
                optimizer.step()
                update_ema(ema, model, float(config["ema_rate"]))
                for group in optimizer.param_groups:
                    group["lr"] = config["lr"] * (1 - index / config["lr_anneal_steps"])
                step = index + 1
                if profile:
                    phases = dict(step=step, data_seconds=before_forward-before_data,
                                  forward_seconds=before_backward-before_forward,
                                  backward_seconds=before_update-before_backward,
                                  update_seconds=clock(True)-before_update,
                                  cuda_synchronized=torch.device(args.device).type == "cuda")
                    timing.write(json.dumps(phases) + "\n"); timing.flush()
                    print(f"[timing] {args.condition} {json.dumps(phases)}", flush=True)
                writer.writerow(dict(step=step, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                     **{**values, "sinkhorn": json.dumps(values["sinkhorn"])}))
                if step % config["log_interval"] == 0 or step == config["total_steps"]:
                    f.flush(); audit.flush()
                    elapsed = time.perf_counter() - log_started
                    speed = elapsed / (step - logged_step)
                    eta = (config["total_steps"] - step) * speed / 3600
                    counts = {k: v["iterations"] for k, v in values["sinkhorn"].items()
                              if isinstance(v, dict) and "iterations" in v}
                    print(f"{args.condition} step={step}/{config['total_steps']} loss={values['total']:.6g} "
                          f"seconds_per_step={speed:.3f} eta_hours={eta:.2f} ot_iterations={counts}", flush=True)
                    log_started, logged_step = time.perf_counter(), step
                if step % config["save_interval"] == 0 or step == config["total_steps"]:
                    final_ema = checkpoint_at(step)
        if recorder.rows:
            recorder.save(run, "training")
        write_json(run / "completed.json", dict(status="completed", checkpoint=str(final_ema),
                   checkpoint_sha256=file_hash(final_ema), step=config["total_steps"]))
    except BaseException as exc:
        write_json(run / "failed.json", dict(error=f"{type(exc).__name__}: {exc}"))
        raise
    print(f"EMA_CHECKPOINT={final_ema}")
    return final_ema
