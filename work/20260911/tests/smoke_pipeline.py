"""Small CPU Stage 1 integration with synthetic expression; no model sampling.

Run explicitly to exercise real Scanpy PCA/neighbors/UMAP twenty times and render
all breakpoint/UMAP figures. Stage 2 is never run here. Outputs remain ignored.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WORK), str(WORK.parents[1])]
from src.artifacts import result_path, write_csv, write_json
from src.breakpoint import analyze
from src.settings import Settings
from src.source_imports import umap_core
from src.trajectory import snapshot_table
from src.umap_adapter import plot_independent


def main(output: Path) -> None:
    output = result_path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = Settings(N=2, S=5)
    rng = np.random.default_rng(311)
    genes = [f"g{i:02d}" for i in range(12)]
    real = rng.normal(size=(48, len(genes))).astype(np.float32)
    obs = pd.DataFrame(
        {
            "Superclass": ["Erythropoietic"] * 40 + ["Immune"] * 8,
            "celltype": ["Ery_A"] * 20 + ["Ery_B"] * 20 + ["Immune"] * 8,
        },
        index=[f"cell{i}" for i in range(48)],
    )
    data = ad.AnnData(real, obs=obs, var=pd.DataFrame({"gene_name": genes}, index=genes))
    data_path = output / "synthetic_real.h5ad"
    data.write_h5ad(data_path)
    predictions = rng.normal(
        scale=0.2, size=(settings.trajectories, settings.M, len(genes))
    ).astype(np.float32)
    for k in range(settings.M):
        predictions[:, k] += 0.01 * k if k <= 8 else 0.08 + (k - 8) * 0.09
    core = umap_core()
    meta = {
        "settings": asdict(settings),
        "gene_count": len(genes),
        "gene_names": genes,
        "gene_order_hash": core.gene_order_hash(genes),
        "data_path": str(data_path),
        "seed": 311,
        "experiment": "synthetic_CPU_smoke_only",
        "status": "synthetic_fixture",
    }
    write_json(output / "metadata.json", meta)
    trajectory = output / "trajectory"
    trajectory.mkdir()
    np.save(trajectory / "pred_xstart.npy", predictions)
    table = snapshot_table(settings)
    write_csv(trajectory / "snapshot_table.csv", table)
    write_json(
        trajectory / "sampling_metadata.json",
        {"status": "completed", "synthetic_fixture": True, "timestep_map": list(range(1000))},
    )
    analyze(output)
    info = plot_independent(output)
    assert info["fit_count"] == 20
    assert info["real_selection"]["selected_cell_count"] == 40
    assert all(fit["generated_cells"] == 10 for fit in info["fits"])
    assert len(list((output / "umap_independent").glob("snapshot_*.png"))) == 20
    assert not (output / "umap_selected_joint").exists()
    print(f"SMOKE_PASSED={output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
