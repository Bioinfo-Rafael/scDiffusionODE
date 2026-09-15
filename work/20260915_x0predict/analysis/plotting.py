"""Render saved CSVs only. Importing this module never creates figures."""

import numpy as np
from ..common import confined, new_dir, read_json, run_id, write_json
from .diagnostics import timestep_grids


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def annotate_diffusion(ax, low):
    ax.axvspan(
        0, 49, color="#56b4e9", alpha=0.12, label="START_X OT region: t=0..49"
    )
    ax.axvspan(
        0,
        10,
        color="#e69f00",
        alpha=0.18,
        label="Very low noise: t=0..10 (sigma ≲0.05)",
    )
    ax.set_xlim(0, 50 if low else 999)
    ax.set_xlabel("Forward diffusion timestep t")


def annotate_trajectory(ax, maximum):
    ax.axvspan(0, 1000, color="#56b4e9", alpha=0.08, label="Diffusion")
    ax.axvline(1000, color="#555555", linestyle="--", label="Terminal diffusion t=0")
    if maximum > 1000:
        ax.axvspan(
            1000, maximum, color="#009e73", alpha=0.12, label="Post-hoc ODE dynamics"
        )
    ax.set_xlabel("Completed updates (diffusion 1..1000; ODE 1001..1100)")


def _save(fig, ax, output, filename):
    plt = _pyplot()
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    path = output / filename
    # Exclusive handle is an additional protection even inside a fresh directory.
    with path.open("xb") as f:
        fig.savefig(f, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_numeric(source, output):
    import pandas as pd

    plt = _pyplot()
    full, low = timestep_grids()
    for filename in ("true_x0_metrics", "cellunet_vs_ode_metrics", "norm_metrics"):
        frame = pd.read_csv(source / f"{filename}.csv")
        for metric in frame.metric.unique():
            groups = [(metric, frame[frame.metric == metric])]
            if filename == "norm_metrics" and metric == "l2":
                selected = frame[frame.metric == metric]
                weighted = selected.series.str.startswith("weighted_")
                groups = [
                    ("raw_l2", selected[~weighted]),
                    ("weighted_l2", selected[weighted]),
                ]
            for group_name, group in groups:
                if group.empty:
                    continue
                for view, times in (("full", full), ("low_noise", low)):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    subset = group[group.t.isin(times)]
                    for series, rows in subset.groupby("series", sort=False):
                        rows = rows.sort_values("t")
                        ax.plot(rows.t, rows["mean"], label=series)
                    annotate_diffusion(ax, view == "low_noise")
                    ax.set(
                        title=f"{filename}: {group_name} ({view})",
                        ylabel=f"Mean per-cell {metric}",
                    )
                    _save(fig, ax, output, f"{filename}_{group_name}_{view}.png")
    for filename, columns, ylabel in (
        (
            "trajectory_diversity",
            ["mean", "median"],
            "Pairwise Euclidean distance in original gene space",
        ),
        (
            "sinkhorn_to_real",
            ["sinkhorn_divergence"],
            "Sinkhorn divergence to real Erythropoietic cells",
        ),
    ):
        frame = pd.read_csv(source / f"{filename}.csv")
        fig, ax = plt.subplots(figsize=(10, 5))
        for column in columns:
            ax.plot(
                frame.reverse_step,
                pd.to_numeric(frame[column], errors="coerce"),
                marker="o",
                label=column,
            )
        if "ot_status" in frame and (frame.ot_status == "not_converged").any():
            ax.text(
                0.02,
                0.95,
                f"Uncomputed Sinkhorn snapshots: {(frame.ot_status == 'not_converged').sum()}",
                transform=ax.transAxes,
                va="top",
            )
        annotate_trajectory(ax, int(frame.reverse_step.max()))
        ax.set(ylabel=ylabel, title=filename + " (state after update)")
        _save(fig, ax, output, filename + ".png")
    sw_path = source / "sliced_wasserstein_snapshots.csv"
    if sw_path.exists():
        frame = pd.read_csv(sw_path)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            frame.reverse_step,
            frame.sliced_wasserstein2,
            marker="o",
            label="Snapshot SW2",
        )
        annotate_trajectory(ax, int(frame.reverse_step.max()))
        ax.set(
            ylabel="Sliced Wasserstein-2",
            title="Endpoint/snapshot distribution (state after update)",
        )
        _save(fig, ax, output, "sliced_wasserstein_snapshots.png")
    convergence = source / "post_ode_convergence.csv"
    if convergence.exists():
        frame = pd.read_csv(convergence)
        for name, columns, ylabel in (
            (
                "post_ode_displacement",
                ["mean_displacement", "median_displacement"],
                "Per-cell displacement L2",
            ),
            (
                "post_ode_vector_field_norm",
                ["mean_field_norm", "max_field_norm"],
                "ODE vector-field L2 norm",
            ),
        ):
            fig, ax = plt.subplots(figsize=(10, 5))
            for column in columns:
                ax.plot(frame.ode_step, frame[column], label=column)
            ax.set(
                xlabel="Post-hoc ODE dynamics step k (diffusion conditioning held at t=0)",
                ylabel=ylabel,
                title=name + " (post-hoc dynamics analysis)",
            )
            _save(fig, ax, output, name + ".png")


def plot_embeddings(source, output):
    import pandas as pd

    plt = _pyplot()
    for fit in read_json(source / "umap_metadata.json"):
        frame = pd.read_csv(source / (fit["fit"] + ".csv"))
        fig, ax = plt.subplots(figsize=(9, 7))
        real = frame.time_label == "Real Erythropoietic"
        ax.scatter(
            frame.loc[real, "UMAP1"],
            frame.loc[real, "UMAP2"],
            color="#bdbdbd",
            s=4,
            alpha=0.4,
            linewidths=0,
            label="Real Erythropoietic",
        )
        # One coherent sequential palette for terminal diffusion -> ODE continuation.
        colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(fit["labels"])))
        for name, color in zip(fit["labels"], colors):
            mask = frame.time_label == name
            ax.scatter(
                frame.loc[mask, "UMAP1"],
                frame.loc[mask, "UMAP2"],
                color=color,
                s=8,
                alpha=0.6,
                linewidths=0,
                label=name,
            )
        title = (
            "Terminal timepoints: one joint UMAP fit"
            if fit["fit"] == "joint_terminal"
            else "Independent UMAP: "
            + fit["labels"][0]
            + "\nCoordinates are not aligned across independent fits"
        )
        ax.set(title=title, xlabel="UMAP1", ylabel="UMAP2")
        _save(fig, ax, output, fit["fit"] + ".png")


def plot(args):
    source = confined(args.input)
    if (
        read_json(source / "completed.json")["status"] != "completed"
        or (source / "failed.json").exists()
    ):
        raise ValueError("refusing partial numerical results")
    output = new_dir(source / "figures" / run_id())
    if (source / "umap_metadata.json").exists():
        plot_embeddings(source, output)
    else:
        plot_numeric(source, output)
    write_json(
        output / "completed.json",
        {"status": "completed", "numerical_input": str(source)},
    )
    print(f"FIGURE_DIR={output}")
    return output
