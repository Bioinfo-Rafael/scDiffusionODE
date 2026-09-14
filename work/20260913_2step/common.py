"""Configuration, source adapters, provenance, and exclusive artifact creation."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import random
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SUITE = Path(__file__).resolve().parent
ROOT = SUITE.parents[1]
FAMILIES = {
    "hill_after_linear": (
        "20260803_ODE_hill_exp",
        "standard_hybrid_single__hill_after_linear",
    ),
    "centered_signed_hill": ("20260816", "linear_centered_signed_hill"),
    "shifted_hill_rho": ("20260816", "linear_shifted_hill_rho"),
}
CONDITIONS = ["stage1_cellunet"] + [
    f"{f}_{o}_soft" for o in ("recon", "trajectory_ot") for f in FAMILIES
]

LEGACY_CONDITIONS = [f"{f}_ot_soft" for f in FAMILIES]


def read_json(path):
    return json.loads(Path(path).read_text())


def confined(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(SUITE) or path == SUITE:
        raise ValueError(f"outputs must be below {SUITE}: {path}")
    return path


def new_dir(path):
    path = confined(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, data):
    with confined(path).open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False, default=str)
        f.write("\n")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with confined(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_id():
    return (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(model_or_state):
    state = (
        model_or_state.state_dict()
        if hasattr(model_or_state, "state_dict")
        else model_or_state
    )
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(f"{name}:{t.dtype}:{tuple(t.shape)}".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite(name, value):
    ok = (
        torch.isfinite(value).all().item()
        if torch.is_tensor(value)
        else np.isfinite(value).all()
    )
    if not ok:
        raise FloatingPointError(f"{name} contains NaN/Inf")
    return value


def source_factory(config):
    return importlib.import_module(f"work.{config['source_suite']}.models.factory")


def umap_core():
    return importlib.import_module("work.20260911.src.source_imports").umap_core()


def metric_helpers():
    return importlib.import_module("work.20260816.viz.analysis_helpers")


def effective_config(condition):
    if condition not in CONDITIONS + LEGACY_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    condition_config = read_json(SUITE / "configs" / f"{condition}.json")
    source = ROOT / "work" / condition_config["source_suite"] / "configs"
    config = {
        **read_json(source / "base.json"),
        **read_json(source / f"{condition_config['source_experiment']}.json"),
    }
    config.update(read_json(SUITE / "configs/base.json"))
    if condition_config["objective"] == "trajectory_ot":
        config.update(read_json(SUITE / "configs/trajectory_defaults.json"))
    config.update(condition_config)
    config["experiment"] = condition
    config["source_config_sha256"] = {
        str(p.relative_to(ROOT)): file_hash(p)
        for p in (
            source / "base.json",
            source / f"{condition_config['source_experiment']}.json",
        )
    }
    if config["objective"] == "stage1":
        config.update(
            model_family="pure_cellunet",
            ode_type=None,
            SoftReg=False,
            use_mask_reg=False,
            ode_reg_lambda=0.0,
        )
    validate_config(config)
    return config


def validate_config(config):
    required = {
        "diffusion_steps": 1000,
        "noise_schedule": "linear",
        "timestep_respacing": "",
        "learn_sigma": False,
        "use_kl": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
        "rescale_learned_sigmas": False,
        "class_cond": False,
        "schedule_sampler": "uniform",
        "ts_layer": None,
        "use_ddim": False,
        "clip_denoised": False,
        "hybrid_norm_mode": "none",
        "reverse_coef": False,
        "regime_gate_mode": "none",
        "use_fp16": False,
        "microbatch": -1,
    }
    differences = {k: config.get(k) for k, v in required.items() if config.get(k) != v}
    if differences:
        raise ValueError(
            f"canonical model/diffusion/representation settings changed: {differences}"
        )
    if config["objective"] != "stage1":
        for key, expected in {
            "SoftReg": True,
            "use_mask_reg": True,
            "off_mask_lambda": 5.0,
            "ode_reg_lambda": 1.0,
            "ode_reg_norm": "l1",
            "sparse_lambda": 0.0,
            "entropy_lambda": 0.0,
            "ratio_reg_weight": 0.0,
        }.items():
            if config.get(key) != expected:
                raise ValueError(f"source soft-constraint setting changed: {key}")


def build_diffusion(config):
    from guided_diffusion.script_util import create_gaussian_diffusion

    validate_config(config)
    keys = (
        "learn_sigma",
        "noise_schedule",
        "use_kl",
        "predict_xstart",
        "rescale_timesteps",
        "rescale_learned_sigmas",
        "timestep_respacing",
    )
    d = create_gaussian_diffusion(
        steps=config["diffusion_steps"], **{k: config[k] for k in keys}
    )
    if d.num_timesteps != 1000 or d.timestep_map != list(range(1000)):
        raise ValueError("requires the unrespaced 1000-step diffusion")
    return d


def load_real(config, genes=None, erythropoietic=False):
    import scanpy as sc

    core = umap_core()
    data = sc.read_h5ad(config["data_dir"])
    actual_genes = core.gene_names_from_adata(data)
    if genes is not None and actual_genes != list(genes):
        raise ValueError("gene order differs from checkpoint")
    if config.get("ts_layer") is not None:
        raise ValueError(
            "canonical experiment uses the source X representation, ts_layer=None"
        )
    selection = {}
    if erythropoietic:
        data, selection = core.select_hematopoietic_subset(
            data, superclasses=("Erythropoietic",)
        )
    return data, actual_genes, selection
