"""Root/identity adapter over the tested 20260803 run-management helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SOURCE = REPO_ROOT / "work/20260803_ODE_hill_exp/scripts/common.py"
spec = importlib.util.spec_from_file_location("common_20260803_reused", SOURCE)
if spec is None or spec.loader is None: raise ImportError(SOURCE)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

FAMILIES = ("ode_only_lincomb_softmax", "standard_hybrid_lincomb_softmax")
ODES = ("softplus", "hill_after_softplus", "exp")
EXPERIMENT_ORDER = tuple(f"{family}__{ode}" for family in FAMILIES for ode in ODES)


def _canonical_experiment(value):
    name = str(value).removesuffix(".json")
    if name not in EXPERIMENT_ORDER: raise base.ConfigurationError(f"unknown experiment: {value}")
    return name


def validate_config(config):
    required = ("experiment", "model_family", "ode_type", "data_dir", "edge_tsv_path")
    missing = [key for key in required if not config.get(key)]
    if missing: raise base.ConfigurationError(f"missing config keys: {missing}")
    expected = f"{config['model_family']}__{config['ode_type']}"
    if expected != config["experiment"] or expected not in EXPERIMENT_ORDER:
        raise base.ConfigurationError(f"invalid experiment identity: {config.get('experiment')}")
    if int(config.get("K", -1)) != 8 or config.get("gate_mode") != "softmax":
        raise base.ConfigurationError("all conditions require K=8 and softmax gate")
    if str(config.get("regime_gate_mode", "none")).lower() != "none":
        raise base.ConfigurationError("TS regime gates are not used")
    if int(config["total_steps"]) != int(config["lr_anneal_steps"]):
        raise base.ConfigurationError("total_steps must equal lr_anneal_steps")


base.SUITE_ROOT = SUITE_ROOT; base.WORK_ROOT = SUITE_ROOT
base.CONFIG_ROOT = SUITE_ROOT / "configs"; base.RUNS_ROOT = SUITE_ROOT / "runs"
base.BATCHES_ROOT = SUITE_ROOT / "batches"; base.validate_config = validate_config
base.EXPERIMENT_ORDER = EXPERIMENT_ORDER
base.EXPERIMENTS = EXPERIMENT_ORDER
base.EXPERIMENT_TO_PARTS = {name: tuple(name.split("__", 1)) for name in EXPERIMENT_ORDER}
base._canonical_experiment = _canonical_experiment
for name in dir(base):
    if not name.startswith("__"): globals().setdefault(name, getattr(base, name))
globals().update({"SUITE_ROOT": SUITE_ROOT, "REPO_ROOT": REPO_ROOT,
                  "CONFIG_ROOT": base.CONFIG_ROOT, "RUNS_ROOT": base.RUNS_ROOT,
                  "BATCHES_ROOT": base.BATCHES_ROOT, "EXPERIMENT_ORDER": EXPERIMENT_ORDER})


def select_experiments(families=None, experiments=None):
    family_set, experiment_set = set(families or ()), set(experiments or ())
    if family_set.difference(FAMILIES): raise base.ConfigurationError("unknown family")
    if experiment_set.difference(EXPERIMENT_ORDER): raise base.ConfigurationError("unknown experiment")
    result = tuple(name for name in EXPERIMENT_ORDER if (not family_set or name.split("__")[0] in family_set) and (not experiment_set or name in experiment_set))
    if not result: raise base.ConfigurationError("filters select no experiments")
    return result


def experiment_config_path(experiment):
    if experiment not in EXPERIMENT_ORDER: raise base.ConfigurationError(experiment)
    return base.CONFIG_ROOT / f"{experiment}.json"


_factory_spec = importlib.util.spec_from_file_location(
    "raw_count_lincomb_model_factory", SUITE_ROOT / "models/factory.py"
)
if _factory_spec is None or _factory_spec.loader is None:
    raise ImportError(SUITE_ROOT / "models/factory.py")
_factory_module = importlib.util.module_from_spec(_factory_spec)
_factory_spec.loader.exec_module(_factory_module)


def build_model_from_config(config, gene_list, timesteps, device):
    """Always use this suite's six-condition factory, independent of sys.path order."""
    return _factory_module.build_model_from_config(
        dict(config), list(gene_list), timesteps, device
    )
