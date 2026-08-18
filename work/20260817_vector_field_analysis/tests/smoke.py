#!/usr/bin/env python3
"""Post-hoc CLI smoke test using an untrained, directly serialized fixture."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ANALYSIS_ROOT = HERE.parent
REPO_ROOT = ANALYSIS_ROOT.parent.parent
SOURCE_SUITE = REPO_ROOT / "work" / "20260803_ODE_hill_exp"
CLI = ANALYSIS_ROOT / "analyze_vector_field.py"
for search_path in (str(REPO_ROOT), str(SOURCE_SUITE)):
    if search_path in sys.path:
        sys.path.remove(search_path)
    sys.path.insert(0, search_path)

from guided_diffusion.script_util import (  # noqa: E402
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from models.factory import build_model_from_config  # noqa: E402


REQUIRED = {
    "cell_metrics.csv",
    "summary_metrics.csv",
    "jacobian_gene_summary.csv",
    "jacobian_top_interactions.csv",
    "real_velocity_pca.png",
    "generated_velocity_pca.png",
    "divergence_pca.png",
    "acceleration_norm_pca.png",
    "speed_pca.png",
    "real_vs_generated_metrics.png",
    "analysis_manifest.json",
}


def _fixture(root: Path) -> tuple[Path, Path]:
    run_dir = root / "run"
    train_dir = run_dir / "train"
    samples_dir = run_dir / "samples"
    train_dir.mkdir(parents=True)
    samples_dir.mkdir(parents=True)
    rng = np.random.default_rng(20260817)
    genes = [f"gene_{index}" for index in range(5)]
    real = rng.lognormal(mean=0.1, sigma=0.35, size=(12, 5)).astype(np.float32)
    generated = rng.lognormal(mean=0.15, sigma=0.4, size=(10, 5)).astype(np.float32)
    data_path = root / "synthetic.h5ad"
    edge_path = root / "edges.tsv"
    obs = pd.DataFrame(
        {"Superclass": ["class_A", "class_B"] * 6},
        index=[f"cell_{index}" for index in range(len(real))],
    )
    var = pd.DataFrame(
        {"gene_name": genes}, index=[f"var_{index}" for index in range(len(genes))]
    )
    ad.AnnData(X=real, obs=obs, var=var).write_h5ad(data_path)
    edge_path.write_text("from\tto\ngene_0\tgene_1\n", encoding="utf-8")
    sample_path = samples_dir / "samples_000001.npz"
    np.savez_compressed(sample_path, cell_gen=generated)

    with (SOURCE_SUITE / "configs" / "base.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(
        {
            "experiment": "standard_hybrid_single__hill_after_linear",
            "model_family": "standard_hybrid_single",
            "ode_type": "hill_after_linear",
            "data_dir": str(data_path),
            "edge_tsv_path": str(edge_path),
            "use_mask_reg": False,
            "SoftReg": False,
            "off_mask_lambda": 0.0,
            "cell_unet_hidden_num": [16, 12, 8, 8],
            "diffusion_steps": 20,
            "noise_schedule": "cosine",
            "device": "cpu",
            "seed": 20260817,
        }
    )
    with (run_dir / "exp_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    defaults = model_and_diffusion_defaults()
    for key in defaults:
        if key in config:
            defaults[key] = config[key]
    _, diffusion = create_model_and_diffusion(**defaults)
    torch.manual_seed(20260817)
    model = build_model_from_config(config, genes, diffusion.num_timesteps, "cpu")
    checkpoint = train_dir / "ema_0.9999_000001.pt"
    torch.save(model.state_dict(), checkpoint)
    manifest = {
        "checkpoint_path": str(checkpoint),
        "sample_path": str(sample_path),
        "data_dir": str(data_path),
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return run_dir, root / "outputs"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="vector-field-smoke-") as temporary:
        run_dir, output_dir = _fixture(Path(temporary))
        command = [
            sys.executable,
            str(CLI),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--max-real-cells",
            "8",
            "--max-generated-cells",
            "8",
            "--jacobian-cells",
            "2",
            "--sensitivity-cells",
            "1",
            "--sensitivity-max-dim",
            "8",
            "--fixed-point-seeds",
            "0",
            "--pca-components",
            "3",
            "--top-interactions",
            "4",
            "--max-arrows",
            "8",
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"smoke CLI failed:\n{completed.stdout}")
        missing = sorted(name for name in REQUIRED if not (output_dir / name).is_file())
        if missing:
            raise AssertionError(f"missing required outputs: {missing}")
        with (output_dir / "analysis_manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("status") != "completed":
            raise AssertionError(f"analysis did not complete: {manifest}")
        if manifest.get("dynamo_refit_performed") is not False:
            raise AssertionError("manifest does not prove the no-refit policy")
        for csv_path in output_dir.glob("*.csv"):
            table = pd.read_csv(csv_path)
            numeric = table.select_dtypes(include=[np.number]).to_numpy()
            if not np.isfinite(numeric).all():
                raise AssertionError(f"{csv_path.name} contains numeric NaN/Inf")
        print(
            json.dumps(
                {
                    "status": "completed",
                    "training_executed": False,
                    "sampling_executed": False,
                    "required_outputs": len(REQUIRED),
                    "created_files": len(manifest["created_files"]),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
