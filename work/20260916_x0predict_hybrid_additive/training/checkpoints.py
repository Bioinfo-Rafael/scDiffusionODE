"""Read-only external Stage1 import; exclusive local Stage2 checkpoints."""
from pathlib import Path
import torch
from ..common import (confined, file_hash, read_json, write_json, state_hash,
                      validate_config, old, VERSION)
from ..models import build_model, assert_frozen


def load_stage1(path):
    path = Path(path).expanduser().resolve()
    before = file_hash(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload["metadata"]
    config = meta["effective_config"]
    old.validate_config(config)
    if (config["objective"] != "stage1" or not config["predict_xstart"] or
            meta["checkpoint_kind"] != "ema" or meta["step"] != 30000 or
            meta["step"] != config["total_steps"]):
        raise ValueError("requires final 30k START_X Stage1 EMA")
    # Strict architecture/key/shape load, and freeze before returning anything.
    model = __import__("importlib").import_module("work.20260915_x0predict.models").build_model(
        config, meta["gene_names"], state=payload["state_dict"])
    model.requires_grad_(False).eval()
    if before != file_hash(path):
        raise ValueError("Stage1 checkpoint changed while reading")
    sidecar = path.with_suffix(".json")
    if sidecar.exists() and read_json(sidecar).get("checkpoint_sha256") != before:
        raise ValueError("Stage1 sidecar SHA256 mismatch")
    origin = dict(checkpoint=str(path), checkpoint_sha256=before,
                  cellunet_hash=state_hash(model), gene_order_hash=meta["gene_order_hash"],
                  data_sha256=meta["data_sha256"], edge_tsv_sha256=meta["edge_tsv_sha256"])
    return payload, origin


def save_checkpoint(path, state, metadata):
    path = confined(path)
    with path.open("xb") as f:
        torch.save(dict(state_dict={k: v.detach().cpu() for k, v in state.items()}, metadata=metadata), f)
    write_json(path.with_suffix(".json"), dict(**metadata, checkpoint_sha256=file_hash(path)))
    return path


def read_checkpoint(path):
    payload = torch.load(confined(path), map_location="cpu", weights_only=True)
    validate_config(payload["metadata"]["effective_config"])
    return payload


def restore(path, device="cpu"):
    payload = read_checkpoint(path)
    meta = payload["metadata"]
    model = build_model(meta["effective_config"], meta["gene_names"], state=payload["state_dict"]).to(device).eval()
    if hasattr(model, "ml_model"):
        assert_frozen(model, expected=meta["frozen_cellunet_hash_before"])
    return model, meta


def canonical_stage1(campaign):
    record = read_json(confined(campaign) / "canonical_stage1.json")
    path = confined(record["checkpoint"])
    if not path.is_relative_to(Path(campaign)) or file_hash(path) != record["checkpoint_sha256"]:
        raise ValueError("campaign Stage1 checkpoint changed/escaped campaign")
    payload = read_checkpoint(path)
    if state_hash(payload["state_dict"]) != record["cellunet_hash"]:
        raise ValueError("campaign Stage1 state mismatch")
    return payload, record
