"""Campaign-local I/O and explicit adapters to the frozen 20260915 conventions."""
import csv
import importlib
import json
from pathlib import Path
from . import reuse

old = importlib.import_module("work.20260915_x0predict.common")
SUITE = Path(__file__).resolve().parent
ROOT = SUITE.parents[1]
VERSION = "20260916_x0predict_hybrid_additive_v1"
STAGE1 = "cellunet_only"
CONDITIONS = [f"{family}_{loss}" for family in
              ("centered_hill", "shifted_hill", "hill_after_linear", "softplus")
              for loss in ("reconst", "ot")]
read_json, file_hash, state_hash = old.read_json, old.file_hash, old.state_hash
seed_all, finite, run_id, git_commit = old.seed_all, old.finite, old.run_id, old.git_commit
umap_core, metric_helpers, load_real = old.umap_core, old.metric_helpers, old.load_real
assert_start_x = old.assert_start_x


def confined(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(SUITE) or path == SUITE:
        raise ValueError(f"output must be below {SUITE}: {path}")
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
    rows = list(rows)
    with confined(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def campaign_path(name):
    if Path(name).name != name or not name.startswith("additive_"):
        raise ValueError("campaign requires a single additive_... name")
    return confined(SUITE / "runs" / name)


def effective_config(condition):
    if condition == STAGE1:
        config = old.effective_config(old.STAGE1)
        item = dict(condition=STAGE1, objective="stage1")
    else:
        if condition not in CONDITIONS:
            raise ValueError(condition)
        item = read_json(SUITE / "configs" / f"{condition}.json")
        config = old.effective_config(f"{item['ode_type']}_{item['objective']}_soft")
    config["base_condition"] = config["condition"]
    config.update(read_json(SUITE / "configs/base.json"))
    config.update(item)
    if condition != STAGE1:
        config["total_steps"] = config["stage2_training_steps"]
    config["experiment"] = condition
    config["ts_cache_path"] = str(SUITE / "runs/unused_ts_cache.json")
    return config


def training_config(config, requested=None):
    """Shorten the run horizon without rewriting the campaign or LR schedule."""
    target = min(config["total_steps"], 10000) if requested is None else requested
    if not isinstance(target, int) or not 1 <= target <= config["total_steps"]:
        raise ValueError("training steps must be positive and cannot exceed the saved campaign horizon")
    return dict(config, total_steps=target)


def condition_step_prefix(campaign, condition, requested=None):
    config = read_json(campaign / "configs" / f"{condition}.json")
    target = training_config(config, requested)["total_steps"]
    return condition if target == config["total_steps"] else f"{condition}_s{target}"


def legacy_config(config):
    # Only translate the identifying envelope, never diffusion/model settings.
    result = dict(config)
    result.update(condition=config["base_condition"], suite_version="20260915_x0predict_v1",
                  hybrid_schedule=old.effective_config(old.STAGE1)["hybrid_schedule"])
    return result


def validate_config(config):
    if config["suite_version"] != VERSION or config["condition"] not in [STAGE1, *CONDITIONS]:
        raise ValueError("not an additive-suite config")
    if config["hybrid_mode"] != "additive" or config["hybrid_schedule"] != read_json(SUITE / "configs/base.json")["hybrid_schedule"]:
        raise ValueError("campaign requires raw-timestep additive Hybrid500")
    if config["condition"] != STAGE1:
        expected = read_json(SUITE / "configs" / f"{config['condition']}.json")
        if any(config[k] != expected[k] for k in ("objective", "ode_type")):
            raise ValueError("condition/model/loss mismatch")
    old.validate_config(legacy_config(config))


def build_diffusion(config):
    validate_config(config)
    return old.build_diffusion(legacy_config(config))


def result_root(config):
    campaign_path(config["output_campaign"])
    return confined(SUITE / "results" / config["output_campaign"])


def source_provenance():
    paths = set(old.source_provenance())
    paths.update(str(p.relative_to(ROOT)) for p in SUITE.rglob("*.py") if ".test_tmp" not in p.parts)
    paths.update(str(p.relative_to(ROOT)) for p in (SUITE / "configs").glob("*.json"))
    return {p: file_hash(ROOT / p) for p in sorted(paths)}
