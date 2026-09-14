"""Read-only discovery and strict restoration of existing Stage-2 bundles."""

import copy
from pathlib import Path

import torch

from ..common import file_hash, finite, read_json, state_hash
from ..models import assert_frozen
from .checkpoints import read_checkpoint


def verified_checkpoint(path):
    path = Path(path).resolve()
    sidecar = read_json(path.with_suffix(".json"))
    if file_hash(path) != sidecar["checkpoint_sha256"]:
        raise ValueError(f"checkpoint SHA-256 mismatch: {path}")
    payload = read_checkpoint(path)
    return payload


def check_origin(meta, condition, canonical):
    if meta["effective_config"]["condition"] != condition:
        raise ValueError("checkpoint condition mismatch")
    if meta["gene_order_hash"] != canonical["gene_order_hash"]:
        raise ValueError("checkpoint gene-order mismatch")
    if (
        meta["originating_stage1"]["checkpoint_sha256"]
        != canonical["checkpoint_sha256"]
    ):
        raise ValueError("checkpoint uses a different canonical Stage 1")
    if meta["frozen_cellunet_hash_before"] != canonical["cellunet_hash"]:
        raise ValueError("checkpoint frozen CellUNet hash mismatch")


def load_bundle(raw_path, campaign, condition, canonical):
    raw_path = Path(raw_path).resolve()
    expected_parent = Path(campaign).resolve() / condition
    if raw_path.parent.name != "checkpoints" or raw_path.parents[2] != expected_parent:
        raise ValueError("resume checkpoint must belong to the same campaign/condition")
    raw = verified_checkpoint(raw_path)
    meta = raw["metadata"]
    check_origin(meta, condition, canonical)
    config = meta["effective_config"]
    step = meta["step"]
    if meta["checkpoint_kind"] != "raw" or not 0 < step < config["total_steps"]:
        raise ValueError("resume requires an intermediate RAW checkpoint")
    if raw_path.name != f"model{step:06d}.pt":
        raise ValueError("raw checkpoint filename/step mismatch")
    ema_path = raw_path.parent / f"ema_{config['ema_rate']}_{step:06d}.pt"
    opt_path = raw_path.parent / f"opt{step:06d}.pt"
    ema = verified_checkpoint(ema_path)
    if ema["metadata"] != {**meta, "checkpoint_kind": "ema"}:
        raise ValueError("raw/EMA checkpoint metadata mismatch")
    frozen_state = {
        k.removeprefix("ml_model."): v
        for k, v in raw["state_dict"].items()
        if k.startswith("ml_model.")
    }
    frozen_ema = {
        k.removeprefix("ml_model."): v
        for k, v in ema["state_dict"].items()
        if k.startswith("ml_model.")
    }
    if (
        state_hash(frozen_state) != canonical["cellunet_hash"]
        or state_hash(frozen_ema) != canonical["cellunet_hash"]
    ):
        raise ValueError("raw/EMA frozen weights differ from canonical CellUNet")
    return {
        "raw": raw,
        "ema": ema,
        "optimizer": torch.load(opt_path, map_location="cpu", weights_only=True),
        "provenance": {
            "step": step,
            "raw_checkpoint": str(raw_path),
            "ema_checkpoint": str(ema_path),
            "optimizer_checkpoint": str(opt_path),
            "file_sha256": {
                str(p): file_hash(p) for p in (raw_path, ema_path, opt_path)
            },
            "rng_policy": "restart RNG/data stream from configured seed; old checkpoints do not save RNG/loader state",
            "bitwise_continuation": False,
        },
    }


def resume_config(bundle, current):
    # Use the stored experiment configuration, with only an increased numerical
    # iteration cap. Never silently change epsilon, tolerance or the objective.
    config = copy.deepcopy(bundle["raw"]["metadata"]["effective_config"])
    if config["objective"] == "stage1":
        raise ValueError("Stage-1 continuation is not supported")
    if config["objective"] == "ot":
        config["ot"]["max_iterations"] = max(
            config["ot"]["max_iterations"], current["ot"]["max_iterations"]
        )
    return config


def restore_training_state(bundle, model, optimizer, expected_hash):
    model.load_state_dict(bundle["raw"]["state_dict"], strict=True)
    optimizer.load_state_dict(bundle["optimizer"])
    assert_frozen(model, optimizer, expected_hash)
    # Verify EMA structure without replacing the trainable raw weights.
    ema = bundle["ema"]["state_dict"]
    if ema.keys() != model.state_dict().keys() or any(
        ema[k].shape != v.shape for k, v in model.state_dict().items()
    ):
        raise ValueError("EMA state structure mismatch")
    for name, value in ema.items():
        finite(f"resumed EMA {name}", value)
    device = next(model.parameters()).device
    return {k: v.detach().to(device).clone() for k, v in ema.items()}


def select_training(campaign, condition, canonical, *, completed_only=False):
    """Return one completed final EMA or the latest complete intermediate bundle."""
    parent = Path(campaign) / condition
    completed = []
    for marker in sorted(parent.glob("*/completed.json")):
        if (marker.parent / "failed.json").exists():
            continue
        record = read_json(marker)
        path = Path(record["checkpoint"]).resolve()
        if path.parent != marker.parent.resolve() / "checkpoints":
            raise ValueError("completed checkpoint is outside its run")
        payload = verified_checkpoint(path)
        meta = payload["metadata"]
        check_origin(meta, condition, canonical)
        if (
            meta["checkpoint_kind"] != "ema"
            or meta["step"] != meta["effective_config"]["total_steps"]
        ):
            raise ValueError("completed run does not point to final EMA")
        if file_hash(path) != record["checkpoint_sha256"]:
            raise ValueError("completed-run checkpoint checksum mismatch")
        completed.append(str(path))
    if len(completed) > 1:
        raise ValueError(
            f"multiple completed trials for {condition}; explicit selection required"
        )
    if completed:
        return {"completed_checkpoint": completed[0]}
    if completed_only:
        return {"missing_completed_checkpoint": True}
    candidates = []
    for path in parent.glob("*/checkpoints/model*.pt"):
        sidecar_path = path.with_suffix(".json")
        if not sidecar_path.is_file():
            continue
        meta = read_json(sidecar_path)
        step = meta["step"]
        if step >= meta["effective_config"]["total_steps"]:
            continue
        rate = meta["effective_config"]["ema_rate"]
        if all(
            (path.parent / name).is_file()
            for name in (
                f"ema_{rate}_{step:06d}.pt",
                f"ema_{rate}_{step:06d}.json",
                f"opt{step:06d}.pt",
            )
        ):
            candidates.append((step, path.parents[1].name, path))
    if not candidates:
        return {"start_fresh": True}
    path = max(candidates)[2]
    bundle = load_bundle(path, campaign, condition, canonical)
    return {"resume_checkpoint": str(path), "step": bundle["provenance"]["step"]}
