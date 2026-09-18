"""Inherited settings, local artifact confinement, and native START_X diffusion."""
import copy
import importlib
import json
import math
from pathlib import Path
from .reuse import load

SUITE = Path(__file__).resolve().parent
ROOT = SUITE.parents[1]
VERSION = "20260917_v1"
MODELS = ("lowrank", "matsum", "lora", "direct_message", "multihop_graph_filter")
LOSSES = ("reconstruction", "knn")
CONDITIONS = tuple(f"{m}_{loss}" for m in MODELS for loss in LOSSES)
STAGE1 = "cellunet_only"
old = importlib.import_module("work.20260916_x0predict_hybrid_additive.common")
for _name in ("read_json", "file_hash", "state_hash", "seed_all", "finite", "run_id", "git_commit", "load_real", "umap_core", "metric_helpers", "assert_start_x"):
    globals()[_name] = getattr(old, _name)


def confined(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(SUITE) or path == SUITE:
        raise ValueError(f"all outputs must be below {SUITE}: {path}")
    return path


def new_dir(path):
    path = confined(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    with confined(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False, default=str)
        f.write("\n")


def write_csv(path, rows, fields=None):
    import csv
    rows = list(rows)
    with confined(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def effective_config(model_type="lowrank", aux_loss="reconstruction", overrides=None):
    config = merge(read_json(SUITE / "configs/base.json"), overrides or {})
    config.update(model_type=model_type, aux_loss=aux_loss, condition=f"{model_type}_{aux_loss}",
                  objective="start_x" if aux_loss == "reconstruction" else "knn")
    validate_config(config)
    return config


def validate_config(c):
    if c["suite_version"] != VERSION or c["model_type"] not in MODELS or c["aux_loss"] not in LOSSES:
        raise ValueError("invalid suite/model/loss")
    if c["condition"] not in (f"{c['model_type']}_{c['aux_loss']}", STAGE1):
        raise ValueError("condition mismatch")
    required = dict(diffusion_steps=1000, noise_schedule="linear", predict_xstart=True,
                    rescale_timesteps=False, timestep_respacing="", learn_sigma=False,
                    use_kl=False, rescale_learned_sigmas=False, schedule_sampler="uniform",
                    use_ddim=False, clip_denoised=False, ts_layer=None, field_dropout=0.,
                    split="all_cells_no_validation", ode_reg_norm="l1", ratio_reg_weight=0.)
    if any(c[k] != v for k, v in required.items()):
        raise ValueError("canonical diffusion/data/objective settings changed")
    for key in ("batch_size", "total_steps", "log_interval", "save_interval", "rank", "K", "time_dim", "field_hidden", "graph_hidden", "edge_chunk_size", "num_samples", "sample_batch_size"):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["total_steps"] > c["lr_anneal_steps"]:
        raise ValueError("training exceeds inherited LR horizon")
    if c["epochs"] is not None and (not isinstance(c["epochs"], int) or c["epochs"] < 1):
        raise ValueError("epochs must be null or a positive integer")
    k = c["knn"]
    for key in ("k", "num_negatives", "query_batch_size", "edge_chunk_size", "integration_steps"):
        if not isinstance(k[key], int) or k[key] < 1:
            raise ValueError(f"knn.{key} must be positive integer")
    for key in ("delta_t", "a", "b", "epsilon"):
        if not math.isfinite(k[key]) or k[key] <= 0:
            raise ValueError(f"knn.{key} must be positive finite")
    for value in (k["lambda_knn"], k["lambda_neg"], c["ode_reg_lambda"], c["off_mask_lambda"], c["weight_decay"]):
        if not math.isfinite(value) or value < 0:
            raise ValueError("loss/decay weights must be nonnegative finite")
    if k["solver"] != "euler" or k["epsilon"] >= 1 or not 0 < c["lr"]:
        raise ValueError("invalid solver, epsilon or learning rate")


def build_diffusion(config):
    validate_config(config)
    from guided_diffusion.script_util import create_gaussian_diffusion
    keys = ("learn_sigma", "noise_schedule", "use_kl", "predict_xstart", "rescale_timesteps", "rescale_learned_sigmas", "timestep_respacing")
    d = create_gaussian_diffusion(steps=config["diffusion_steps"], **{k: config[k] for k in keys})
    assert_start_x(d)
    if d.num_timesteps != 1000 or d.timestep_map != list(range(1000)):
        raise ValueError("requires original 1000-step diffusion")
    return d


def result_root(config):
    return confined(SUITE / "results" / config["output_campaign"])


def source_provenance():
    paths = set(old.source_provenance())
    paths.update(str(p.relative_to(ROOT)) for p in SUITE.rglob("*.py") if not any(s in p.parts for s in ("results", "cache")))
    paths.update(str(p.relative_to(ROOT)) for p in (SUITE / "configs").glob("*.json"))
    return {p: file_hash(ROOT / p) for p in sorted(paths)}
