"""Render completed occupation CSVs and recorded gradients, without model calls."""

from pathlib import Path
from ..common import SUITE, confined, new_dir, read_json, run_id, write_json
from .plotting import _pyplot, _save


def plot_occupation(args):
    import pandas as pd

    sources = [confined(p) for p in args.input]
    frames, metadata = [], []
    for source in sources:
        if (
            read_json(source / "completed.json")["status"] != "completed"
            or (source / "failed.json").exists()
        ):
            raise ValueError("refusing partial occupation analysis")
        frames.append(pd.read_csv(source / "occupation_analysis.csv"))
        metadata.append(read_json(source / "occupation_metadata.json"))
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["condition", "distribution"]).any():
        raise ValueError(
            "ambiguous multiple evaluations per condition; select one explicitly"
        )
    output = new_dir(SUITE / "results" / "occupation_comparison" / run_id())
    plt = _pyplot()
    try:
        with (output / "occupation_comparison.csv").open("x") as f:
            frame.to_csv(f, index=False)
        for metric in ("sinkhorn_divergence", "sliced_wasserstein2"):
            fig, ax = plt.subplots(figsize=(12, 6))
            table = frame.pivot(
                index="condition", columns="distribution", values=metric
            )
            table = table.apply(pd.to_numeric, errors="coerce")
            table.plot.bar(ax=ax)
            if table.isna().any().any():
                ax.text(
                    0.02,
                    0.95,
                    "Missing distances are uncomputed, not zero",
                    transform=ax.transAxes,
                    va="top",
                )
            ax.set(
                ylabel=metric,
                title="Independent x50 ODE trajectories: occupation vs endpoint",
            )
            _save(fig, ax, output, metric + ".png")
        for meta in metadata:
            loss_path = Path(meta["training_loss_history"])
            if not loss_path.exists():
                continue
            loss = pd.read_csv(loss_path)
            if "grad_ot" not in loss:
                continue
            loss = loss.dropna(subset=["grad_ot", "grad_soft"])
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(loss.step, loss.grad_ot, label="Trajectory OT gradient norm")
            ax.plot(loss.step, loss.grad_soft, label="Weighted soft gradient norm")
            ax.set(
                xlabel="Optimizer step",
                ylabel="ODE parameter gradient L2 norm",
                title=meta["condition"],
            )
            _save(fig, ax, output, meta["condition"] + "_gradients.png")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(loss.step, loss.grad_ratio_ot_to_soft, label="OT / weighted soft")
            ax.set(
                xlabel="Optimizer step",
                ylabel="Gradient norm ratio",
                title=meta["condition"],
            )
            _save(fig, ax, output, meta["condition"] + "_gradient_ratio.png")
        write_json(
            output / "completed.json",
            dict(status="completed", sources=[str(p) for p in sources]),
        )
    except BaseException as exc:
        write_json(output / "failed.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    print(f"FIGURE_DIR={output}")
    return output
