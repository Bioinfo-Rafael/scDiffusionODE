#!/usr/bin/env python3
"""Plot the covariance-derived hybrid ODE/Cell_Unet mixing weights.

Reused implementation sources:
  - work/20260707_lincomb/configs/base.json
  - work/20260707_lincomb/configs/hybrid_ts_soft_lincomb.json
  - work/20260707_lincomb/scripts/common.py::load_experiment_config
  - work/20260707_lincomb/utils/regime_time.py::
      estimate_lambda_max_from_adata, estimate_ts_from_lambda
  - guided_diffusion/script_util.py::create_model_and_diffusion
  - ODE/ode_20260609_hybrid5x3.py::
      UnifiedODEMLHybrid._regime_ode_weight
  - work/20260707_lincomb/viz/plot_gate_diagnostics_0707.py::
      _save_gate_curve
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import anndata as ad


OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[2]
WORK_0707 = REPO_ROOT / "work" / "20260707_lincomb"
DATA_PATH = REPO_ROOT / "work" / "20260215_embryonic" / "data" / "Embryonic.h5ad"
CONFIG_PATH = WORK_0707 / "configs" / "hybrid_ts_soft_lincomb.json"

for import_path in (REPO_ROOT, WORK_0707):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

# Reused from work/20260707_lincomb/scripts/common.py.
from scripts.common import load_experiment_config  # noqa: E402

# Reused from work/20260707_lincomb/utils/regime_time.py.
from utils.regime_time import (  # noqa: E402
    estimate_lambda_max_from_adata,
    estimate_ts_from_lambda,
)

# Reused from guided_diffusion/script_util.py.
from guided_diffusion.script_util import (  # noqa: E402
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

# Reused from ODE/ode_20260609_hybrid5x3.py.
from ODE.ode_20260609_hybrid5x3 import UnifiedODEMLHybrid  # noqa: E402


def main() -> None:
    """Estimate T_s and write the hybrid weight curve and summary artifacts."""
    config = load_experiment_config(CONFIG_PATH)
    config["data_dir"] = str(DATA_PATH)  # replace the archived Linux path locally

    print(f"Loading AnnData: {DATA_PATH}")
    adata = ad.read_h5ad(DATA_PATH)
    print(f"AnnData shape: {adata.shape}")

    # Reused exactly from train_0707.py::_resolve_ts.
    ts_seed = config.get("ts_seed")
    if ts_seed is None:
        ts_seed = config.get("seed", 0)
    lambda_info = estimate_lambda_max_from_adata(
        adata,
        layer=config.get("ts_layer"),
        n_cells=int(config.get("ts_n_cells", 200_000)),
        seed=int(ts_seed),
        use_randomized_pca=True,
    )

    # Reused from train_0707.py: construct the same diffusion schedule.
    diffusion_args = model_and_diffusion_defaults()
    diffusion_args["diffusion_steps"] = int(config["diffusion_steps"])
    _, diffusion = create_model_and_diffusion(**diffusion_args)
    ts_info = estimate_ts_from_lambda(
        diffusion.alphas_cumprod, lambda_info["lambda_max"]
    )
    t_s = int(ts_info["t_s"])

    # Call the existing method directly. It only needs these gate attributes;
    # model parameters and a checkpoint do not affect this time-only weight.
    gate_state = SimpleNamespace(
        regime_gate_mode=config["regime_gate_mode"],
        regime_gate_type=config["regime_gate_type"],
        t_s=t_s,
        gate_tau=float(config["gate_tau"]),
    )
    t = np.arange(diffusion.num_timesteps, dtype=np.int64)
    r_ode = UnifiedODEMLHybrid._regime_ode_weight(
        gate_state, torch.from_numpy(t), torch.device("cpu"), torch.float32
    ).squeeze(-1).numpy()
    r_cellunet = 1.0 - r_ode
    alpha_bar = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
    covariance_score = alpha_bar * float(lambda_info["lambda_max"])

    curve = pd.DataFrame(
        {
            "t": t,
            "alpha_bar": alpha_bar,
            "alpha_bar_lambda_max": covariance_score,
            "r_ode_lincomb": r_ode,
            "r_cellunet": r_cellunet,
        }
    )
    curve_path = OUTPUT_DIR / "hybrid_ts_soft_weight_curve.csv"
    curve.to_csv(curve_path, index=False)

    summary = {
        "data_path": str(DATA_PATH),
        "config_path": str(CONFIG_PATH),
        "diffusion_steps": int(diffusion.num_timesteps),
        "regime_gate_mode": config["regime_gate_mode"],
        "regime_gate_type": config["regime_gate_type"],
        "gate_tau": float(config["gate_tau"]),
        **lambda_info,
        **ts_info,
        "r_ode_lincomb_at_t0": float(r_ode[0]),
        "r_ode_lincomb_at_ts": float(r_ode[t_s]),
        "r_ode_lincomb_at_tmax": float(r_ode[-1]),
        "r_cellunet_at_t0": float(r_cellunet[0]),
        "r_cellunet_at_ts": float(r_cellunet[t_s]),
        "r_cellunet_at_tmax": float(r_cellunet[-1]),
    }
    summary_path = OUTPUT_DIR / "ts_estimate_and_weight_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Layout adapted from plot_gate_diagnostics_0707.py::_save_gate_curve.
    fig, (ax_weight, ax_score) = plt.subplots(1, 2, figsize=(13, 4.2))
    ax_weight.plot(t, r_ode, color="tab:blue", label="r(t): ODE / LinComb weight")
    ax_weight.plot(
        t, r_cellunet, color="tab:orange", label="1-r(t): Cell_Unet weight"
    )
    ax_weight.axvline(t_s, color="tab:red", linestyle="--", label=f"T_s = {t_s}")
    ax_weight.set(
        xlabel="diffusion timestep t",
        ylabel="hybrid mixing weight",
        ylim=(-0.02, 1.02),
        title="Hybrid mixing weights across diffusion steps",
    )
    ax_weight.legend()

    ax_score.plot(
        t,
        covariance_score,
        color="tab:green",
        label=r"$\bar{\alpha}_t \lambda_{max}$",
    )
    ax_score.axhline(1.0, color="0.35", linestyle=":", label="target = 1")
    ax_score.axvline(t_s, color="tab:red", linestyle="--", label=f"T_s = {t_s}")
    ax_score.set(
        xlabel="diffusion timestep t",
        ylabel=r"$\bar{\alpha}_t \lambda_{max}$",
        title="Covariance criterion used to choose T_s",
    )
    ax_score.legend()
    fig.suptitle(
        "hybrid_ts_soft_lincomb: "
        f"lambda_max={lambda_info['lambda_max']:.4g}, "
        f"T_s={t_s}, gate_tau={config['gate_tau']}",
        y=1.03,
    )
    fig.tight_layout()
    figure_path = OUTPUT_DIR / "hybrid_ts_soft_weight_curve.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"lambda_max={lambda_info['lambda_max']:.10g}")
    print(f"T_s={t_s}")
    print(f"w_ode(t=0)={r_ode[0]:.8f}")
    print(f"w_ode(t=T_s)={r_ode[t_s]:.8f}")
    print(f"w_ode(t=max)={r_ode[-1]:.8f}")
    print(f"CSV={curve_path}")
    print(f"SUMMARY={summary_path}")
    print(f"FIGURE={figure_path}")


if __name__ == "__main__":
    main()
