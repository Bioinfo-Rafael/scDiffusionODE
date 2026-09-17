"""Completed terminal diffusion metrics beside the same shared CellUNet baseline."""
from ..common import STAGE1, CONDITIONS, SUITE, new_dir, read_json, run_id, write_csv, write_json, condition_step_prefix


def compare(campaign, training_steps=None):
    import pandas as pd
    rows = []
    for condition in [STAGE1, *CONDITIONS]:
        prefix = condition if condition == STAGE1 else condition_step_prefix(campaign, condition, training_steps)
        marker = campaign / "steps" / (prefix + "_analyze") / "completed.json"
        if not marker.exists():
            continue
        path = read_json(marker)["artifact"]
        row = dict(condition=condition, baseline=STAGE1, numerical_source=path, step_prefix=prefix)
        for file, columns in {
            "collapse_coverage": ["aggregate_gene_variance", "mean_distance_from_population_centroid", "mean_knn_distance", "added_real_knn_radius_coverage"],
            "trajectory_diversity": ["mean", "median"],
            "sinkhorn_to_real": ["sinkhorn_divergence", "ot_status"],
            "sliced_wasserstein_snapshots": ["sliced_wasserstein2"],
        }.items():
            frame = pd.read_csv(path + "/" + file + ".csv")
            terminal = frame[frame.reverse_step == 1000].iloc[0]
            for column in columns:
                value = terminal[column]
                row[file + "_" + column] = None if pd.isna(value) else value
        rows.append(row)
    output = new_dir(SUITE / "results" / campaign.name / "comparisons" / run_id())
    if rows:
        write_csv(output / "terminal_comparison.csv", rows)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        metrics = ["collapse_coverage_aggregate_gene_variance", "collapse_coverage_added_real_knn_radius_coverage",
                   "sliced_wasserstein_snapshots_sliced_wasserstein2"]
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
        for ax, metric in zip(axes, metrics):
            ax.bar([r["condition"] for r in rows], [r[metric] for r in rows])
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", labelrotation=30)
        with (output / "terminal_comparison.png").open("xb") as f:
            fig.savefig(f, format="png", dpi=120)
        plt.close(fig)
    write_json(output / "completed.json", dict(status="completed", included=[r["condition"] for r in rows],
               missing=[c for c in [STAGE1, *CONDITIONS] if c not in {r["condition"] for r in rows}],
               baseline_policy="one identical frozen CellUNet baseline shared by all eight conditions"))
    return output
