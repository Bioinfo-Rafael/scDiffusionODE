"""Strict Stage1 input reuse and self-contained comparison checkpoints."""
import importlib
from pathlib import Path
import torch
from ..common import confined, file_hash, write_json, validate_config
from ..models import build_model, assert_frozen
load_stage1 = importlib.import_module("work.20260916_x0predict_hybrid_additive.training.checkpoints").load_stage1


def save_checkpoint(path, state, metadata):
    path = confined(path)
    with path.open("xb") as f:
        torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in state.items()}, metadata=metadata), f)
    write_json(path.with_suffix(".json"), dict(**metadata, checkpoint_sha256=file_hash(path)))
    return path


def restore(path, device="cpu"):
    path = confined(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload["metadata"]
    validate_config(meta["effective_config"])
    model = build_model(meta["effective_config"], meta["gene_names"], state=payload["state_dict"]).to(device).eval()
    if hasattr(model, "ml_model"):
        assert_frozen(model, expected=meta["frozen_cellunet_hash_before"])
    return model, meta
