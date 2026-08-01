"""Shared run/config helpers for the 2026-07-07 pipelines."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
REPO_ROOT = WORK_ROOT.parent.parent


def bool_value(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise ValueError(f"invalid boolean: {value}")


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_experiment_config(config_path):
    base = read_json(WORK_ROOT / "configs" / "base.json")
    selected = read_json(config_path)
    base.update(selected)
    validate_config(base)
    return base


def validate_config(config):
    mode = str(config.get("regime_gate_mode", "none"))
    reverse = bool_value(config.get("reverse_coef", False))
    if mode.lower() != "none" and reverse:
        raise ValueError("reverse_coef=true cannot be combined with an enabled regime gate")
    if mode.lower() != "none" and bool_value(config.get("rescale_timesteps", False)):
        raise ValueError("regime gate currently requires rescale_timesteps=false")
    gate_mode = str(config.get("gate_mode", "raw")).lower()
    if float(config.get("sparse_lambda", 0.0)) > 0 and gate_mode != "raw":
        raise ValueError("sparse_lambda > 0 requires gate_mode=raw")
    if float(config.get("entropy_lambda", 0.0)) > 0 and gate_mode != "softmax":
        raise ValueError("entropy_lambda > 0 requires gate_mode=softmax")
    if float(config.get("gate_tau", 20.0)) <= 0:
        raise ValueError("gate_tau must be > 0")


def create_run_dir(experiment, suffix=""):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in suffix)
        stamp += "_" + safe
    run_dir = WORK_ROOT / "runs" / str(experiment) / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def artifact_dirs(run_dir):
    run_dir = Path(run_dir).resolve()
    suite_root = WORK_ROOT
    try:
        relative = run_dir.relative_to((WORK_ROOT / "runs").resolve())
        if len(relative.parts) < 2:
            raise ValueError
        experiment, run_id = relative.parts[0], relative.parts[1]
        external_root = None
    except ValueError:
        candidate_runs_root = run_dir.parent.parent
        if candidate_runs_root.name == "runs":
            relative = run_dir.relative_to(candidate_runs_root)
            if len(relative.parts) < 2:
                raise ValueError(f"run directory must be <suite>/runs/<experiment>/<run_id>: {run_dir}")
            suite_root = candidate_runs_root.parent
            experiment, run_id = relative.parts[0], relative.parts[1]
            external_root = None
        else:
            experiment, run_id = run_dir.parent.name, run_dir.name
            external_root = run_dir / "artifacts"
    paths = {
        "run": run_dir,
        "train": run_dir / "train",
        "samples": (external_root / "samples" if external_root else suite_root / "samples" / experiment / run_id),
        "viz": (external_root / "viz" if external_root else suite_root / "viz" / experiment / run_id),
        "logs": (external_root / "logs" if external_root else suite_root / "logs" / experiment / run_id),
        "results": (external_root / "results" if external_root else suite_root / "results" / experiment / run_id),
    }
    return paths


def command_string(argv=None):
    argv = argv or sys.argv
    return " ".join(shlex.quote(str(v)) for v in argv)


def git_state():
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        ).stdout.strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short"),
        "diff": run("diff", "--no-ext-diff"),
    }


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def update_manifest(run_dir, **updates):
    path = Path(run_dir) / "manifest.json"
    payload = read_json(path) if path.exists() else {}
    payload.update(updates)
    write_json(path, payload)
    return payload


def build_model_from_config(config, gene_list, timesteps, device):
    from ODE.ode_20260707_lincomb import build_denoiser_0707

    return build_denoiser_0707(
        gene_list=gene_list,
        edge_tsv_path=config["edge_tsv_path"],
        timesteps=timesteps,
        denoiser_mode=config.get("denoiser_mode", "lincomb_only"),
        hybrid_norm_mode=config.get("hybrid_norm_mode", "none"),
        reverse_coef=bool_value(config.get("reverse_coef", False)),
        regime_gate_mode=config.get("regime_gate_mode", "none"),
        regime_gate_type=config.get("regime_gate_type", "sigmoid"),
        t_s=config.get("t_s"), gate_tau=float(config.get("gate_tau", 20.0)),
        K=int(config.get("K", 8)),
        use_mask=bool_value(config.get("use_mask_reg", True)),
        soft=bool_value(config.get("SoftReg", True)),
        time_dim=int(config.get("time_dim", 64)),
        field_hidden=int(config.get("field_hidden", 256)),
        field_dropout=float(config.get("field_dropout", 0.0)),
        use_decay=bool_value(config.get("use_decay", True)),
        gate_mode=config.get("gate_mode", "raw"),
        gate_temperature=float(config.get("gate_temperature", 1.0)),
        off_mask_lambda=float(config.get("off_mask_lambda", 0.0)),
        sparse_lambda=float(config.get("sparse_lambda", 0.0)),
        entropy_lambda=float(config.get("entropy_lambda", 0.0)),
        ratio_reg_weight=float(config.get("ratio_reg_weight", 0.0)),
        ratio_reg_target=float(config.get("ratio_reg_target", 1.0)),
        hybrid_scale_init=float(config.get("hybrid_scale_init", 1.0)),
        hybrid_scale_eps=float(config.get("hybrid_scale_eps", 1e-8)),
        scale_model_type=config.get("scale_model_type", "none"),
        scale_input_source=config.get("scale_input_source", "ml_emb"),
        ode_input_source=config.get("ode_input_source", "none"),
        scale_hidden=int(config.get("scale_hidden", 128)),
        scale_eps=float(config.get("scale_eps", 1e-8)),
        device=device,
    )
