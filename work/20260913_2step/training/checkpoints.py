"""Exclusive self-contained checkpoints, including immutable provenance."""

from pathlib import Path
import torch
from ..common import confined, file_hash, read_json, state_hash, write_json
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
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if set(payload) != {"state_dict", "metadata"}:
        raise ValueError("expected a self-contained two-stage checkpoint")
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
    record = read_json(Path(campaign) / "canonical_stage1.json")
    path = Path(record["checkpoint"])
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


def register_stage1(source_campaign, target_campaign):
    """Read/verify the source and write only into a new campaign."""
    payload, record = canonical_stage1(source_campaign)
    if Path(source_campaign).resolve() == Path(target_campaign).resolve():
        raise ValueError("Stage1 source must remain read-only")
    write_json(Path(target_campaign) / "canonical_stage1.json", record)
    return payload, record
