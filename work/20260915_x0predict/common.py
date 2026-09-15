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
    "simple_softplus": ("20260830", "SimpleSoftplus20260830"),
    "hill_after_linear": ("20260830", "HillAfterLinear20260830"),
    "centered_signed_hill": ("20260830", "CenteredSignedHill20260830"),
    "shifted_hill_rho": ("20260830", "ShiftedHillRho20260830"),
}
CONDITIONS = [f"{family}_{objective}_soft"
              for family in FAMILIES for objective in ("start_x", "ot")]
STAGE1 = "stage1_cellunet"


def campaign_path(name):
    if Path(name).name != name or not name.startswith("x0predict_"):
        raise ValueError("campaign must be a single x0predict_... directory name")
    return confined(SUITE / "runs" / name)


def source_provenance():
    paths = [
        "guided_diffusion/gaussian_diffusion.py", "guided_diffusion/script_util.py",
        "guided_diffusion/cell_model.py", "guided_diffusion/cell_datasets_loader.py",
        "guided_diffusion/nn.py", "guided_diffusion/respace.py", "guided_diffusion/resample.py",
        "ODE/ode_20260609_mathmlp.py", "ODE/ode_20260609_hybrid5x3.py",
        "work/20260830/models/ode_fields_20260830.py", "work/20260830/models/factory.py",
        "work/20260913_2step/models/hybrid500.py",
        "work/20260913_2step/losses/sinkhorn.py",
        "work/20260913_2step/common.py", "work/20260911/src/source_imports.py",
        "work/20260830/hematopoietic_viz/core.py", "work/20260816/viz/analysis_helpers.py",
    ]
    paths += [str(p.relative_to(ROOT)) for p in SUITE.rglob("*.py")]
    return {p: file_hash(ROOT / p) for p in sorted(paths)}


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
    if condition not in CONDITIONS + [STAGE1]:
        raise ValueError(f"unknown condition: {condition}")
    legacy = importlib.import_module("work.20260913_2step.common")
    if condition == STAGE1:
        config = legacy.effective_config(STAGE1)
    else:
        family = next(f for f in FAMILIES if condition.startswith(f + "_"))
        # Keep the already-audited two-stage training settings (30k updates).
        # Reuse only the explicit field factory from 20260830, not its 100k
        # CellUNet-consistency experiment or its wrapper/training objective.
        config = legacy.effective_config("hill_after_linear_recon_soft")
        config.update(source_suite="20260830", source_experiment=family,
                      model_family="standard_hybrid_single", ode_type=family,
                      K=1, gate_mode="none", theta_init=1.0, regulation_A_init=1.0,
                      alpha_init_std=None, rho_init_std=None, target_chunk_size=16)
        config["reference_sources"] = {
            "training_defaults": "work/20260913_2step (30k START_X adaptation)",
            "ode_fields": "work/20260830/models/ode_fields_20260830.py",
            "ode_factory": "work/20260830/models/factory.py::build_ode_from_config",
            "hybrid_schedule": "work/20260913_2step/models/hybrid500.py::Hybrid500Mixin",
            "cell_unet": "guided_diffusion/cell_model.py::Cell_Unet",
        }
    config.update(read_json(SUITE / "configs/base.json"))
    config.update(read_json(SUITE / "configs" / f"{condition}.json"))
    config["experiment"] = condition
    config["ts_cache_path"] = str(SUITE / "runs" / "unused_ts_cache.json")
    if condition != STAGE1:
        family = config["ode_type"]
        source, cls = FAMILIES[family]
        config.update(
            ode_implementation=f"work.{source}.models.ode_fields_20260830.{cls}",
            actual_ode_components=1, ode_family=family,
            ode_source_suite="work/20260830",
            ode_source_file="work/20260830/models/ode_fields_20260830.py",
            ode_class=cls, ode_components=1, expert_gating=False,
        )
        config["ode_source_sha256"] = file_hash(ROOT / config["ode_source_file"])
    validate_config(config)
    return config


def validate_config(config):
    if config.get("condition") not in CONDITIONS + [STAGE1]:
        raise ValueError("unknown x0predict condition")
    if config.get("objective") not in ("stage1", "start_x", "ot"):
        raise ValueError("only stage1, start_x, and ot objectives are supported")
    expected_objective = ("stage1" if config["condition"] == STAGE1 else
                          "ot" if config["condition"].endswith("_ot_soft") else "start_x")
    if config["objective"] != expected_objective:
        raise ValueError("condition/objective mismatch")
    if config.get("hybrid_schedule") != read_json(SUITE / "configs/base.json")["hybrid_schedule"]:
        raise ValueError("requires original-timestep Hybrid500")
    required = {
        "diffusion_steps": 1000,
        "noise_schedule": "linear",
        "timestep_respacing": "",
        "learn_sigma": False,
        "use_kl": False,
        "predict_xstart": True,
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
        family = config.get("ode_type")
        if family not in FAMILIES or not config["condition"].startswith(family + "_"):
            raise ValueError("condition/ODE family mismatch")
        source, _ = FAMILIES[family]
        if config["source_suite"] != source:
            raise ValueError("ODE source suite changed")
        if config["K"] != 1 or config["gate_mode"] != "none":
            raise ValueError("canonical ODE component/gate configuration changed")
        if (config.get("ode_components") != 1 or config.get("expert_gating") is not False
                or config.get("ode_class") != FAMILIES[family][1]
                or config.get("ode_source_file") != "work/20260830/models/ode_fields_20260830.py"):
            raise ValueError("explicit single-ODE provenance is required")
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
    assert_start_x(d)
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



def assert_start_x(diffusion):
    from guided_diffusion.gaussian_diffusion import ModelMeanType, LossType
    if diffusion.model_mean_type != ModelMeanType.START_X:
        raise AssertionError("x0predict requires ModelMeanType.START_X")
    if diffusion.loss_type != LossType.MSE:
        raise AssertionError("x0predict requires ordinary MSE diffusion")


def result_root(config):
    campaign_path(config["output_campaign"])
    return confined(SUITE / "results" / config["output_campaign"])
