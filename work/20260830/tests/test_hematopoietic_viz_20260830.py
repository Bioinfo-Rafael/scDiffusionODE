from __future__ import annotations

import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import torch


SUITE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for path in (REPO_ROOT, SUITE_ROOT):
    while str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

from analysis.gradients import parameter_fingerprint  # noqa: E402
from hematopoietic_viz.core import (  # noqa: E402
    assert_gene_alignment,
    build_sampling_anndata,
    compute_vector_fields,
    file_sha256,
    gene_order_hash,
    select_hematopoietic_subset,
)
from hematopoietic_viz.plotting import compute_velocity_embeddings  # noqa: E402
from hematopoietic_viz import runner as viz_runner  # noqa: E402
from models import build_model_from_config  # noqa: E402
from scripts.common import EXPERIMENT_ORDER, load_experiment_config, write_json  # noqa: E402
from guided_diffusion.script_util import create_gaussian_diffusion  # noqa: E402


def synthetic_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "Superclass": [
                "Erythropoietic", "Immune", "Other", "Erythropoietic", "Immune"
            ],
            "celltype": ["HSC", "Macrophages", "Neuron", "Erythroblast", "HSC"],
        },
        index=[f"cell_{index}" for index in range(5)],
    )
    var = pd.DataFrame(
        {"gene_name": ["g0", "g1", "g2", "g3"]},
        index=[f"var_{index}" for index in range(4)],
    )
    return ad.AnnData(np.arange(20, dtype=np.float32).reshape(5, 4), obs=obs, var=var)


def tiny_model_and_diffusion():
    config = load_experiment_config("01_centered_signed_hill_lambda0p1")
    config["cell_unet_hidden_num"] = [16, 12, 8, 8]
    config["cell_unet_dropout"] = 0.0
    diffusion = create_gaussian_diffusion(steps=1000, noise_schedule="linear")
    genes = [f"g{index}" for index in range(4)]
    model = build_model_from_config(
        config, genes, diffusion.num_timesteps, mask=torch.zeros(4, 4)
    )
    return model, diffusion


class RecordingDiffusion:
    def __init__(self, diffusion):
        self._diffusion = diffusion
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._diffusion, name)

    def q_sample(self, x_start, t, noise=None):
        self.calls.append(
            {
                "t": t.detach().cpu().clone(),
                "x_start": x_start.detach().cpu().clone(),
                "noise": noise.detach().cpu().clone(),
            }
        )
        return self._diffusion.q_sample(x_start=x_start, t=t, noise=noise)


class HematopoieticVizTests(unittest.TestCase):
    def test_protected_pipeline_files_match_pre_visualization_hashes(self):
        expected = {
            "work/20260830/scripts/launch.py": "ed0fab922ac06d3ea65d23ab1ae1216c661eba7346928205f7cb254ead3ccdc5",
            "work/20260830/scripts/train.py": "5fc976c5a4b23a67bf5d4c6389968e2e0b25ceb78b1d22544e9feb000f51f5db",
            "work/20260830/scripts/sample.py": "e313f830131dded178bad9a34229c80f3109503f545a8bb92ba1310d5e68ea8c",
            "work/20260830/scripts/analyze.py": "819269fa5b20d9fa38da8a8378ed487e8e4a858f0ff5bbc232abda8e894f787b",
            "work/20260830/training/__init__.py": "5a336e8a7fbe61dcf3243d52649e069d4c10d74ba415d108d65390e8278d6920",
            "work/20260830/training/train_loop_20260830.py": "95c1cc9cd39eda5db2d54d635e69b0383bc83193d7e5083174d9e430985ebc06",
        }
        for relative, digest in expected.items():
            actual = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, relative)

    def test_default_union_selection_and_no_leakage(self):
        source = synthetic_adata()
        subset, metadata = select_hematopoietic_subset(source)
        self.assertEqual(
            metadata["selected_superclasses"], ["Erythropoietic", "Immune"]
        )
        self.assertEqual(metadata["selected_superclass_column"], "Superclass")
        self.assertEqual(subset.n_obs, 4)
        self.assertEqual(
            set(subset.obs["Superclass"].astype(str)), {"Erythropoietic", "Immune"}
        )
        self.assertNotIn("Other", subset.obs["Superclass"].astype(str).tolist())

    def test_exact_hematopoietic_value_has_precedence_and_override_is_exact(self):
        source = synthetic_adata()
        source.obs["Superclass"] = ["Hematopoietic", "Immune", "Other", "Hematopoietic", "Immune"]
        subset, metadata = select_hematopoietic_subset(source)
        self.assertEqual(metadata["selected_superclasses"], ["Hematopoietic"])
        self.assertEqual(subset.n_obs, 2)
        immune, selected = select_hematopoietic_subset(source, superclasses=("Immune",))
        self.assertEqual(selected["selected_superclasses"], ["Immune"])
        self.assertEqual(immune.n_obs, 2)

    def test_generated_dimension_and_all_available_gene_order_checks(self):
        source = synthetic_adata()
        subset, _ = select_hematopoietic_subset(source)
        genes = source.var["gene_name"].astype(str).tolist()
        generated = np.ones((3, 4), dtype=np.float32)
        audit = assert_gene_alignment(
            genes,
            source.X,
            generated,
            model_genes=genes,
            generated_genes=genes,
            sample_created_from_run_config=True,
        )
        self.assertEqual(audit["gene_order_hash"], gene_order_hash(genes))
        self.assertTrue(audit["model_gene_names_and_order_match"])
        self.assertTrue(audit["sample_contains_embedded_gene_names"])
        combined = build_sampling_anndata(subset, generated)
        self.assertEqual(combined.shape, (subset.n_obs + 3, 4))
        self.assertEqual(
            int((combined.obs["sampling_origin"].astype(str) == "Generated (unconditional)").sum()),
            3,
        )
        with self.assertRaises(ValueError):
            assert_gene_alignment(
                genes, source.X, generated, model_genes=list(reversed(genes)),
                sample_created_from_run_config=True,
            )
        with self.assertRaises(ValueError):
            assert_gene_alignment(
                genes, source.X, generated, model_genes=genes,
                generated_genes=list(reversed(genes)), sample_created_from_run_config=True,
            )
        with self.assertRaises(ValueError):
            assert_gene_alignment(
                genes, source.X, np.ones((2, 3)), model_genes=genes,
                sample_created_from_run_config=True,
            )

    def test_vector_outputs_use_q_sample_and_do_not_mutate_model(self):
        torch.manual_seed(7)
        model, base_diffusion = tiny_model_and_diffusion()
        diffusion = RecordingDiffusion(base_diffusion)
        matrix = np.random.default_rng(5).uniform(0.1, 1.0, (7, 4)).astype(np.float32)
        before = parameter_fingerprint(model)
        ode, cells, audit = compute_vector_fields(
            model, diffusion, matrix, (0, 9), batch_size=3,
            noise_seed=19, device=torch.device("cpu"),
        )
        self.assertEqual(ode.shape, matrix.shape)
        self.assertEqual(cells[0].shape, matrix.shape)
        self.assertEqual(cells[9].shape, matrix.shape)
        self.assertEqual(before, parameter_fingerprint(model))
        self.assertEqual(audit["model_fingerprint_before"], audit["model_fingerprint_after"])
        self.assertFalse(audit["optimizer_constructed"])
        self.assertFalse(audit["optimizer_step_performed"])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual({int(call["t"][0]) for call in diffusion.calls}, {0, 9})
        self.assertEqual(len(diffusion.calls), 6)
        # The RNG is restarted for each timestep, so the same cells receive the
        # same noise while every t, including t>0, goes through q_sample.
        for low, high in zip(diffusion.calls[:3], diffusion.calls[3:]):
            self.assertTrue(torch.equal(low["x_start"], high["x_start"]))
            self.assertTrue(torch.equal(low["noise"], high["noise"]))

    def test_scvelo_layers_share_one_umap_and_have_exact_shapes(self):
        source = synthetic_adata()
        subset, _ = select_hematopoietic_subset(source)
        subset.obsm["X_umap"] = np.arange(subset.n_obs * 2, dtype=float).reshape(-1, 2)
        subset.layers["ode"] = np.ones(subset.shape, dtype=np.float32)
        subset.layers["cell"] = np.full(subset.shape, 2.0, dtype=np.float32)
        calls = []

        def velocity_graph(adata, **kwargs):
            calls.append(("graph", kwargs.copy()))

        def velocity_embedding(adata, *, basis, vkey):
            calls.append(("embedding", {"basis": basis, "vkey": vkey}))
            adata.obsm[f"{vkey}_umap"] = np.zeros((adata.n_obs, 2), dtype=float)

        fake = types.SimpleNamespace(
            tl=types.SimpleNamespace(
                velocity_graph=velocity_graph,
                velocity_embedding=velocity_embedding,
            )
        )
        original = subset.obsm["X_umap"].copy()
        with mock.patch.dict(sys.modules, {"scvelo": fake}):
            returned = compute_velocity_embeddings(subset, ("ode", "cell"), n_jobs=2)
        self.assertTrue(np.array_equal(original, returned))
        self.assertTrue(np.array_equal(original, subset.obsm["X_umap"]))
        self.assertEqual(subset.layers["ode"].shape, subset.shape)
        self.assertEqual(subset.layers["cell"].shape, subset.shape)
        graph_calls = [kwargs for kind, kwargs in calls if kind == "graph"]
        self.assertEqual([call["vkey"] for call in graph_calls], ["ode", "cell"])
        self.assertTrue(all(call["xkey"] == "X" for call in graph_calls))
        self.assertTrue(all(call["backend"] == "loky" for call in graph_calls))

    def test_existing_sample_hash_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.npz"
            np.savez_compressed(sample, cell_gen=np.arange(12).reshape(3, 4))
            before = file_sha256(sample)
            with np.load(sample, allow_pickle=False) as archive:
                np.asarray(archive["cell_gen"])
            self.assertEqual(before, file_sha256(sample))

    def test_partial_launcher_skips_unfinished_and_completed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            batch = "partial-batch"
            ready = root / EXPERIMENT_ORDER[0] / batch
            completed = root / EXPERIMENT_ORDER[1] / batch
            write_json(ready / "exp_config.json", {"experiment": EXPERIMENT_ORDER[0]})
            write_json(completed / "exp_config.json", {"experiment": EXPERIMENT_ORDER[1]})
            write_json(completed / "hematopoietic_viz/metadata.json", {
                "status": "completed",
                "sample_path": str((completed / "samples/sample.npz").resolve()),
                "checkpoint": str((completed / "checkpoints/model.pt").resolve()),
            })
            executed = []

            def fake_resolve(run, config, explicit=""):
                del config, explicit
                return run / "samples/sample.npz", run / "checkpoints/model.pt", {}

            def fake_execute(run, options):
                del options
                executed.append(run)
                return {"status": "completed", "run_dir": str(run)}

            with mock.patch.object(viz_runner, "RUNS_ROOT", root), mock.patch.object(
                viz_runner, "resolve_current_sample", side_effect=fake_resolve
            ):
                with self.assertWarns(UserWarning):
                    summary = viz_runner.run_all_available(
                        batch, viz_runner.HematopoieticVizOptions(), execute=fake_execute
                    )
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["skipped_completed"], 1)
            self.assertEqual(summary["skipped_unfinished"], 10)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(executed, [ready.resolve()])
            self.assertTrue(Path(summary["summary_path"]).is_file())

    def test_current_repository_dataset_inventory_if_present(self):
        data_path = REPO_ROOT / "work/20260215_embryonic/data/Embryonic.h5ad"
        if not data_path.is_file():
            self.skipTest("matching repository Embryonic.h5ad is not present")
        current = ad.read_h5ad(data_path, backed="r")
        try:
            self.assertIn("Superclass", current.obs.columns)
            values = set(current.obs["Superclass"].astype(str).unique())
            self.assertNotIn("Hematopoietic", values)
            self.assertTrue({"Erythropoietic", "Immune"}.issubset(values))
            selected = current.obs["Superclass"].astype(str).isin(
                ["Erythropoietic", "Immune"]
            )
            self.assertEqual(int(selected.sum()), 45156)
            genes = current.var["gene_name"].astype(str).tolist()
            self.assertEqual(len(genes), 1024)
            self.assertEqual(len(set(genes)), 1024)
        finally:
            current.file.close()


if __name__ == "__main__":
    unittest.main()
