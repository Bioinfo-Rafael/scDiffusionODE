#!/usr/bin/env python3
"""Synthetic Stage1 bundle -> production CLI -> 3 fields/figures/resume guards."""
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def main():
    common = importlib.import_module("work.20260915_x0predict.common")
    models = importlib.import_module("work.20260915_x0predict.models")
    output_root = HERE / "outputs"
    output_root.mkdir(exist_ok=True)
    fixture = Path(tempfile.mkdtemp(prefix="validation_", dir=output_root))
    rng = np.random.default_rng(42)
    X = rng.gamma(2, 1, (120, 64)).astype(np.float32)
    X[:40, :20] += 2
    X[40:80, 20:40] += 2
    genes = [f"gene_{i}" for i in range(X.shape[1])]
    data = fixture / "real.h5ad"
    adata = ad.AnnData(X, obs=pd.DataFrame({
        "Superclass": ["Erythropoietic"] * len(X),
        "celltype": [f"state_{i // 40}" for i in range(len(X))],
    }, index=[f"cell_{i}" for i in range(len(X))]), var=pd.DataFrame(index=genes))
    adata.write_h5ad(data)
    config = common.effective_config(common.STAGE1)
    config.update(data_dir=str(data), cell_unet_hidden_num=[32, 16, 8, 8])
    torch.manual_seed(42)
    model = models.build_model(config, genes)
    checkpoint = fixture / "ema_0.9999_030000.pt"
    meta = dict(effective_config=config, checkpoint_kind="ema", step=30000,
                gene_names=genes, gene_order_hash="synthetic", data_sha256=common.file_hash(data),
                edge_tsv_sha256="synthetic")
    torch.save(dict(state_dict=model.state_dict(), metadata=meta), checkpoint)
    output = fixture / "analysis"
    command = [sys.executable, str(HERE / "scripts/run_analysis.py"), "--mode", "smoke",
               "--checkpoint", str(checkpoint), "--data", str(data), "--device", "cpu",
               "--output-dir", str(output)]
    env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def run(suffix, args=(), success=True):
        with (fixture / f"{suffix}.log").open("w") as log:
            result = subprocess.run(command + list(args), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if (result.returncode == 0) != success:
            raise AssertionError(f"unexpected exit {result.returncode}; see {fixture / f'{suffix}.log'}")

    run("dry_run", ["--dry-run"])
    assert not output.exists(), "dry-run wrote outputs"
    run("smoke")
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["model_count"] == 3
    shared = np.load(output / "common/erythropoietic_fixed_umap.npz")
    fields = []
    scales = []
    for t in (100, 900, 1000):
        directory = output / "models" / f"t_{t:04d}"
        item = json.loads((directory / "model_manifest.json").read_text())
        assert item["projection"]["field"]["timestep"] == t
        assert item["projection"]["field"]["outside_training_timestep_range"] == (t == 1000)
        assert len(item["figures"]) >= 9
        for figure in item["figures"]:
            assert (directory / figure).stat().st_size > 1000
        with np.load(directory / "observed_umap_velocity.npz") as saved:
            np.testing.assert_array_equal(saved["coordinates"], shared["coordinates"])
            np.testing.assert_array_equal(saved["cell_ids"], shared["cell_ids"])
        with np.load(directory / "observed_gene_velocity.npz") as saved:
            with torch.no_grad():
                expected = model.eval()(torch.from_numpy(X), torch.full((len(X), 1), t)).numpy()
            np.testing.assert_allclose(saved["velocity"], expected, rtol=1e-5, atol=1e-6)
        scales.append(item["plot_settings"])
        with np.load(directory / "landscape_flux_arrays.npz") as arrays:
            fields.append({k: arrays[k].copy() for k in arrays.files})
    for key in ("potential_limits", "probability_limits", "curl_scale", "axis_bounds"):
        assert scales[0][key] == scales[1][key] == scales[2][key]
    before = (output / "analysis_manifest.json").read_bytes()
    run("changed_time", ["--timesteps", "100", "900", "999"], success=False)
    assert (output / "analysis_manifest.json").read_bytes() == before
    run("resume")
    for index, t in enumerate((100, 900, 1000)):
        with np.load(output / "models" / f"t_{t:04d}" / "landscape_flux_arrays.npz") as arrays:
            for key, value in fields[index].items():
                np.testing.assert_array_equal(value, arrays[key])
    print(f"PASS: all three fields, shared coordinates/scales, raw outputs, figures, resume and mismatch guard\n{output}")


if __name__ == "__main__":
    main()
