#!/usr/bin/env python3
"""End-to-end synthetic smoke test for the independent visualization pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch


SUITE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for path in (REPO_ROOT, SUITE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.runner import create_diffusion  # noqa: E402
from hematopoietic_viz.core import file_sha256  # noqa: E402
from hematopoietic_viz.runner import (  # noqa: E402
    HematopoieticVizOptions,
    run_visualization,
)
from models import build_model_from_config  # noqa: E402
from scripts.common import load_experiment_config, write_json  # noqa: E402


REQUIRED_FIGURES = {
    "01_sampling_umap_real_hema_vs_generated.png",
    "02_sampling_umap_celltypes_plus_generated.png",
    "10_ode_velocity_stream.png",
    "11_ode_velocity_arrow.png",
    "12_ode_velocity_grid.png",
    "20_cellunet_t000_velocity_stream.png",
    "21_cellunet_t000_velocity_arrow.png",
    "22_cellunet_t000_velocity_grid.png",
    "20_cellunet_t009_velocity_stream.png",
    "21_cellunet_t009_velocity_arrow.png",
    "22_cellunet_t009_velocity_grid.png",
    "30_ode_vs_cellunet_stream.png",
}


def main() -> int:
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    with tempfile.TemporaryDirectory(prefix="hema-viz-smoke-") as directory:
        root = Path(directory)
        run = root / "01_centered_signed_hill_lambda0p1" / "smoke"
        data_path = root / "Embryonic.h5ad"
        genes = [f"gene_{index}" for index in range(6)]
        obs = pd.DataFrame(
            {
                "Superclass": (
                    ["Erythropoietic"] * 20 + ["Immune"] * 20 + ["Other"] * 10
                ),
                "celltype": (
                    ["HSC"] * 10 + ["Erythroblast"] * 10
                    + ["Macrophages"] * 10 + ["Monocytes"] * 10
                    + ["Neuron"] * 10
                ),
            },
            index=[f"cell_{index}" for index in range(50)],
        )
        real = ad.AnnData(
            X=rng.lognormal(0.0, 0.25, (50, 6)).astype(np.float32),
            obs=obs,
            var=pd.DataFrame({"gene_name": genes}, index=genes),
        )
        real.write_h5ad(data_path)

        config = load_experiment_config("01_centered_signed_hill_lambda0p1")
        config.update(
            {
                "data_dir": str(data_path.resolve()),
                "edge_tsv_path": "",
                "use_mask_reg": False,
                "cell_unet_hidden_num": [16, 12, 8, 8],
                "cell_unet_dropout": 0.0,
                "device": "cpu",
            }
        )
        write_json(run / "exp_config.json", config)
        diffusion = create_diffusion(config)
        model = build_model_from_config(config, genes, diffusion.num_timesteps, "cpu")
        checkpoint_dir = run / "checkpoints/segment_000/model"
        checkpoint_dir.mkdir(parents=True)
        raw = checkpoint_dir / "model100000.pt"
        optimizer = checkpoint_dir / "opt100000.pt"
        ema = checkpoint_dir / "ema_0.9999_100000.pt"
        torch.save(model.state_dict(), raw)
        torch.save({}, optimizer)
        torch.save(model.state_dict(), ema)

        samples = run / "samples"
        samples.mkdir(parents=True)
        sample = samples / "samples_ema_0.9999_100000.npz"
        np.savez_compressed(
            sample,
            cell_gen=rng.lognormal(0.0, 0.25, (12, 6)).astype(np.float32),
        )
        write_json(
            sample.with_suffix(".json"),
            {
                "checkpoint": str(ema.resolve()),
                "sample_path": str(sample.resolve()),
                "ode_forward_call_count": 0,
            },
        )
        sample_hash = file_sha256(sample)

        result = run_visualization(
            run,
            HematopoieticVizOptions(
                timesteps=(0, 9),
                batch_size=11,
                device="cpu",
                n_jobs=1,
                pca_components=5,
                neighbors=8,
                neighbor_pcs=4,
                save_h5ad=True,
                paga=False,
            ),
        )
        assert result["status"] == "completed", result
        assert sample_hash == file_sha256(sample)
        output = run / "hematopoietic_viz"
        observed = {path.name for path in (output / "figures").glob("*.png")}
        assert REQUIRED_FIGURES.issubset(observed), sorted(REQUIRED_FIGURES - observed)
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["selected_cell_count"] == 40
        assert metadata["selected_superclasses"] == ["Erythropoietic", "Immune"]
        assert metadata["gene_count"] == 6
        assert metadata["shared_vector_umap_asserted"] is True
        assert metadata["field_audit"]["optimizer_constructed"] is False
        assert (
            metadata["field_audit"]["model_fingerprint_before"]
            == metadata["field_audit"]["model_fingerprint_after"]
        )
        assert metadata["sample_sha256_before"] == metadata["sample_sha256_after"]
        vector = ad.read_h5ad(output / "h5ad/hematopoietic_vector_fields.h5ad")
        assert vector.layers["velocity_ode_20260830"].shape == (40, 6)
        assert vector.layers["velocity_cellunet_t000_20260830"].shape == (40, 6)
        assert vector.layers["velocity_cellunet_t009_20260830"].shape == (40, 6)
        expected_umap = pd.read_csv(output / "csv/vector_field_umap_coordinates.csv")
        assert expected_umap.shape[0] == 40
        print(json.dumps({"status": "ok", "figures": len(observed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
