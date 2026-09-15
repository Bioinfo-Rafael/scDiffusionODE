"""Exclusive self-contained checkpoints, including immutable provenance."""

from pathlib import Path
import torch
from ..common import confined, file_hash, read_json, state_hash, write_json, validate_config
from ..models import assert_frozen, build_model


def save_checkpoint(path, state, metadata):
    path = confined(path)
    cpu = {k: v.detach().cpu() for k, v in state.items()}
    payload = {"state_dict": cpu, "metadata": metadata}
    with path.open("xb") as f:
        torch.save(payload, f)
    write_json(
        path.with_suffix(".json"),
        {**metadata, "checkpoint": str(path), "checkpoint_sha256": file_hash(path)},
    )
    return path


def read_checkpoint(path):
    payload = torch.load(confined(path), map_location="cpu", weights_only=True)
    if set(payload) != {"state_dict", "metadata"}:
        raise ValueError("expected a self-contained two-stage checkpoint")
    config = payload["metadata"]["effective_config"]
    validate_config(config)
    if config["suite_version"] != "20260915_x0predict_v1":
        raise ValueError("checkpoint is not from the START_X campaign")
    return payload


def restore(path, device="cpu"):
    payload = read_checkpoint(path)
    meta = payload["metadata"]
    model = (
        build_model(
            meta["effective_config"], meta["gene_names"], state=payload["state_dict"]
        )
        .to(device)
        .eval()
    )
    if hasattr(model, "ml_model"):
        assert_frozen(model, expected=meta["frozen_cellunet_hash_before"])
    return model, meta


def canonical_stage1(campaign):
    campaign = confined(campaign)
    record = read_json(campaign / "canonical_stage1.json")
    path = confined(record["checkpoint"])
    if not path.is_relative_to(campaign / "stage1_cellunet"):
        raise ValueError("Stage1 must have been trained in this START_X campaign")
    if file_hash(path) != record["checkpoint_sha256"]:
        raise ValueError("canonical Stage-1 EMA checkpoint hash changed")
    payload = read_checkpoint(path)
    meta = payload["metadata"]
    if (
        meta["effective_config"]["objective"] != "stage1"
        or meta["checkpoint_kind"] != "ema"
        or meta["step"] != meta["effective_config"]["total_steps"]
    ):
        raise ValueError("canonical checkpoint must be final Stage-1 EMA")
    if state_hash(payload["state_dict"]) != record["cellunet_hash"]:
        raise ValueError("canonical CellUNet hash mismatch")
    return payload, record
