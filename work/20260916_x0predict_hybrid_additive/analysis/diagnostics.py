"""Reuse x0 metrics; additionally record every raw sampling timestep."""
import numpy as np
import torch
from ..common import finite, metric_helpers, write_csv
from ..reuse import load

legacy = load("analysis._diagnostics", "analysis/diagnostics.py")
per_cell_metrics = legacy.per_cell_metrics
diagnostics = legacy.diagnostics
timestep_grids = legacy.timestep_grids


class BranchRecorder:
    def __init__(self, deduplicate=False):
        self.rows = {}
        self.deduplicate = deduplicate
        self.last_time = None

    @torch.no_grad()
    def __call__(self, t, branches):
        t = torch.as_tensor(t).reshape(-1)
        if self.deduplicate and len(t.unique()) == 1:
            if self.last_time == int(t[0]):
                return
            self.last_time = int(t[0])
        ml = branches["ml_raw"].detach().double()
        ode = branches["ode_raw"].detach().double()
        output = branches["output"].detach().double()
        if len(t) == 1:
            t = t.expand(len(ml))
        metrics = torch.stack([ml.norm(dim=1), ode.norm(dim=1),
                               branches["ode_contribution"].detach().double().norm(dim=1),
                               metric_helpers()._sample_corr(ml, ode)], dim=1)
        for time in t.unique():
            selected = t == time
            key = int(time)
            if key not in self.rows:
                self.rows[key] = dict(n=0, sums=np.zeros(4), gene_sum=np.zeros(output.shape[1]),
                                      gene_square=np.zeros(output.shape[1]))
            row = self.rows[key]
            values = output[selected]
            row["n"] += len(values)
            row["sums"] += finite("branch metrics", metrics[selected].sum(0)).cpu().numpy()
            row["gene_sum"] += values.sum(0).cpu().numpy()
            row["gene_square"] += values.square().sum(0).cpu().numpy()

    def save(self, output, prefix="sampling"):
        rows, variances = [], []
        for t, values in sorted(self.rows.items()):
            n = values["n"]
            means = values["sums"] / n
            variance = np.maximum(0, values["gene_square"] / n - (values["gene_sum"] / n)**2)
            variances.append(variance)
            rows.append(dict(raw_t=t, r=max(0, 1-t/500), cell_weight=1, n_cells=n,
                             cell_norm=means[0], ode_norm=means[1], weighted_ode_norm=means[2],
                             cell_ode_pearson=means[3], aggregate_gene_variance=float(variance.mean()),
                             rms_distance_from_prediction_centroid=float(np.sqrt(variance.sum()))))
        write_csv(output / (prefix + "_branch_metrics.csv"), rows)
        with (output / (prefix + "_gene_variance.npz")).open("xb") as f:
            np.savez(f, raw_t=sorted(self.rows), gene_variance=np.array(variances))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
        for key in ("cell_norm", "ode_norm", "weighted_ode_norm"):
            axes[0].plot([r["raw_t"] for r in rows], [r[key] for r in rows], label=key)
        axes[0].set_ylabel("Mean per-cell L2 norm")
        axes[0].legend()
        for ax, key in zip(axes[1:], ("cell_ode_pearson", "aggregate_gene_variance")):
            ax.plot([r["raw_t"] for r in rows], [r[key] for r in rows])
            ax.set_ylabel(key)
        for ax in axes:
            ax.axvline(500, color="grey", linestyle="--")
        axes[-1].set_xlabel("Raw diffusion timestep (reverse diffusion: 999 to 0)")
        with (output / (prefix + "_branch_metrics.png")).open("xb") as f:
            fig.savefig(f, format="png", dpi=120)
        plt.close(fig)


class RecordingCellOnly(torch.nn.Module):
    def __init__(self, model, recorder):
        super().__init__()
        self.model, self.recorder = model, recorder

    def forward(self, x, t, y=None):
        value = self.model(x, t, y)
        self.recorder(t, dict(ml_raw=value, ode_raw=torch.zeros_like(value),
                             ode_contribution=torch.zeros_like(value), output=value))
        return value
