"""Loss schema normalization and contribution analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = (
    "diffusion_loss_raw",
    "ode_regularization_raw",
    "cell_ode_consistency_raw_20260830",
)
WEIGHTED_COLUMNS = (
    "diffusion_loss_raw",
    "ode_regularization_weighted",
    "cell_ode_consistency_weighted_20260830",
)


def _component_files(run_dir: Path) -> list[Path]:
    return sorted(
        run_dir.glob("checkpoints/segment_*/model/loss_components_20260830.csv")
    )


def load_loss_history(
    run_dir,
    config: dict,
    *,
    rolling_window: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map saved names to explicit raw/weighted analysis names.

    ``ode_soft_constraint`` already includes ``off_mask_lambda`` inside the ODE.
    ``ode_soft_constraint_weighted`` additionally includes the outer
    ``ode_reg_lambda`` and is the final total-loss contribution.
    """

    run = Path(run_dir)
    files = _component_files(run)
    if not files:
        raise FileNotFoundError(
            f"no loss_components_20260830.csv below {run / 'checkpoints'}"
        )
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["loss_source_path"] = str(path.resolve())
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    required = {
        "step",
        "diffusion_loss",
        "ode_soft_constraint",
        "ode_soft_constraint_weighted",
        "cell_ode_consistency_20260830",
        "cell_ode_consistency_weighted_20260830",
        "total_loss",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise KeyError("loss component CSV is missing: " + ", ".join(missing))

    off_mask_lambda = float(config.get("off_mask_lambda", 5.0))
    ode_reg_lambda = float(config.get("ode_reg_lambda", 1.0))
    cell_lambda = float(config["cell_ode_reg_lambda_20260830"])
    ode_base = (
        source["ode_soft_constraint"].astype(float) / off_mask_lambda
        if off_mask_lambda > 0 else source["ode_soft_constraint"].astype(float) * 0.0
    )
    history = pd.DataFrame({
        "training_step": source["step"].astype(np.int64),
        "diffusion_loss_raw": source["diffusion_loss"].astype(float),
        "ode_regularization_base_before_off_mask_lambda": ode_base,
        "ode_regularization_raw": ode_base,
        "ode_regularization_weighted_once_by_off_mask_lambda": source[
            "ode_soft_constraint"
        ].astype(float),
        "ode_regularization_weighted": source["ode_soft_constraint_weighted"].astype(float),
        "cell_ode_consistency_raw_20260830": source[
            "cell_ode_consistency_20260830"
        ].astype(float),
        "cell_ode_consistency_weighted_20260830": source[
            "cell_ode_consistency_weighted_20260830"
        ].astype(float),
        "total_loss": source["total_loss"].astype(float),
        "off_mask_lambda_internal": off_mask_lambda,
        "ode_reg_lambda_outer": ode_reg_lambda,
        "cell_ode_reg_lambda_20260830": cell_lambda,
        "loss_source_path": source["loss_source_path"],
    })
    history = (
        history.sort_values(["training_step", "loss_source_path"])
        .drop_duplicates("training_step", keep="last")
        .reset_index(drop=True)
    )
    window = max(int(rolling_window), 1)
    for column in (*RAW_COLUMNS, "ode_regularization_weighted", "cell_ode_consistency_weighted_20260830", "total_loss"):
        rolling = history[column].rolling(window=window, min_periods=1)
        history[f"{column}_rolling_median_w{window}"] = rolling.median()
        history[f"{column}_rolling_q25_w{window}"] = rolling.quantile(0.25)
        history[f"{column}_rolling_q75_w{window}"] = rolling.quantile(0.75)

    denominator = (
        history["diffusion_loss_raw"]
        + history["ode_regularization_weighted"]
        + history["cell_ode_consistency_weighted_20260830"]
    )
    safe = denominator.where(denominator.abs() > 1e-30, np.nan)
    fractions = pd.DataFrame({
        "training_step": history["training_step"],
        "loss_sum_for_fraction": denominator,
        "diffusion_contribution_fraction": history["diffusion_loss_raw"] / safe,
        "ode_regularization_contribution_fraction": history[
            "ode_regularization_weighted"
        ] / safe,
        "cell_ode_contribution_fraction_20260830": history[
            "cell_ode_consistency_weighted_20260830"
        ] / safe,
    }).fillna(0.0)
    return history, fractions


def late_loss_summary(history: pd.DataFrame, fraction: pd.DataFrame) -> dict[str, float]:
    if history.empty:
        return {}
    cutoff = float(history["training_step"].max()) * 0.75
    late = history[history["training_step"] >= cutoff]
    late_fraction = fraction[fraction["training_step"] >= cutoff]
    if late.empty:
        late = history.tail(1)
        late_fraction = fraction.tail(1)
    return {
        "late_diffusion_contribution": float(late["diffusion_loss_raw"].mean()),
        "late_ode_regularization_contribution": float(
            late["ode_regularization_weighted"].mean()
        ),
        "late_cell_ode_contribution_20260830": float(
            late["cell_ode_consistency_weighted_20260830"].mean()
        ),
        "late_diffusion_fraction": float(
            late_fraction["diffusion_contribution_fraction"].mean()
        ),
        "late_ode_regularization_fraction": float(
            late_fraction["ode_regularization_contribution_fraction"].mean()
        ),
        "late_cell_ode_fraction_20260830": float(
            late_fraction["cell_ode_contribution_fraction_20260830"].mean()
        ),
    }


__all__ = ["late_loss_summary", "load_loss_history"]
