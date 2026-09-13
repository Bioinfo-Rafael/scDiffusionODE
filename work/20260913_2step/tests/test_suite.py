"""CPU-only synthetic tests. No data, experiment sampling, or figure generation."""

import ast
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
PKG = "work.20260913_2step"
common = importlib.import_module(PKG + ".common")
models = importlib.import_module(PKG + ".models")
obj = importlib.import_module(PKG + ".training.objectives")
ot = importlib.import_module(PKG + ".losses.sinkhorn")
cp = importlib.import_module(PKG + ".training.checkpoints")
traj = importlib.import_module(PKG + ".sampling.trajectory")
diag = importlib.import_module(PKG + ".analysis.diagnostics")
dist = importlib.import_module(PKG + ".analysis.distributions")
umaps = importlib.import_module(PKG + ".analysis.umaps")

torch.set_num_threads(1)
TEMP_ROOT = common.SUITE / ".test_tmp"
TEMP_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TEMP_ROOT / "matplotlib"))


class SuiteTests(unittest.TestCase):
    def setUp(self):
        common.seed_all(1234)
        self.tmp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.path = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.edge = self.path / "edges.tsv"
        self.edge.write_text("from\tto\na\tb\nc\ta\n")
        self.genes = ["a", "b", "c", "d"]
        self.stage1_config = self.small("stage1_cellunet")
        self.stage1 = models.build_model(self.stage1_config, self.genes)
        self.diffusion = common.build_diffusion(self.stage1_config)
        self.x = torch.rand(5, 4) + 0.5
        self.noise = torch.randn(5, 4, dtype=torch.float64)

    def small(self, name):
        c = common.effective_config(name)
        c.update(
            cell_unet_hidden_num=[8, 8, 4, 4],
            field_hidden=8,
            time_dim=8,
            edge_tsv_path=str(self.edge),
        )
        return c

    def hybrid(self, name):
        config = self.small(name)
        model = models.build_model(config, self.genes)
        fingerprint = models.freeze_from_stage1(model, self.stage1.state_dict())
        return config, model, fingerprint

    def test_canonical_matrix_and_source_defaults(self):
        self.assertEqual(len(common.CONDITIONS), 7)
        selected = common.read_json(
            ROOT / "work/20260911/configs/selected_models.json"
        )["models"]
        self.assertEqual(
            {x["experiment"] for x in selected},
            {v[1] for v in common.FAMILIES.values()},
        )
        for condition in common.CONDITIONS:
            c = common.effective_config(condition)
            self.assertEqual(c["cell_unet_hidden_num"], [2000, 1000, 500, 500])
            for key, value in {
                "diffusion_steps": 1000,
                "noise_schedule": "linear",
                "batch_size": 128,
                "lr": 1e-4,
                "ema_rate": "0.9999",
                "seed": 1234,
                "weight_decay": 1e-4,
                "clip_denoised": False,
                "predict_xstart": False,
                "ts_layer": None,
            }.items():
                self.assertEqual(c[key], value)

    def test_stage1_has_no_ode(self):
        from guided_diffusion.cell_model import Cell_Unet

        self.assertIs(type(self.stage1), Cell_Unet)
        self.assertFalse(hasattr(self.stage1, "ode_model"))
        self.assertTrue(all(p.requires_grad for p in self.stage1.parameters()))

    def test_stage1_loss_exact_source_mse(self):
        t = torch.tensor([0, 49, 50, 500, 999])
        total, details = obj.training_loss(
            self.stage1,
            self.diffusion,
            self.x,
            t,
            torch.ones(5),
            self.stage1_config,
            self.noise,
        )
        reference = self.diffusion.training_losses(
            self.stage1, self.x, t, noise=self.noise
        )["loss"].mean()
        torch.testing.assert_close(total, reference, rtol=0, atol=0)
        self.assertEqual(details["soft"], 0)
        total.backward()
        self.assertTrue(any(p.grad is not None for p in self.stage1.parameters()))

    def test_full_and_low_noise_samplers(self):
        for name in common.CONDITIONS:
            config = self.small(name)
            sampler = obj.timestep_sampler(config, self.diffusion)
            support = np.flatnonzero(sampler.weights())
            expected = np.arange(50 if config["objective"] == "ot" else 1000)
            np.testing.assert_array_equal(support, expected)
            t, weights = sampler.sample(20000, "cpu")
            self.assertEqual(int(t.min()), 0)
            self.assertEqual(int(t.max()), int(expected[-1]))
            torch.testing.assert_close(weights, torch.ones_like(weights))

    def test_all_six_freeze_optimizer_ema_and_self_contained_restore(self):
        for condition in common.CONDITIONS[1:]:
            with self.subTest(condition=condition):
                c, model, before = self.hybrid(condition)
                self.assertEqual(
                    common.state_hash(model.ml_model), common.state_hash(self.stage1)
                )
                optimizer = models.optimizer_for(model, c)
                ema = copy.deepcopy(model.state_dict())
                model.train()
                model.ml_model.train(True)
                models.assert_frozen(model, optimizer, before)
                t = (
                    torch.tensor([0, 1, 10, 30, 49])
                    if c["objective"] == "ot"
                    else torch.tensor([0, 49, 50, 500, 999])
                )
                loss, _ = obj.training_loss(
                    model, self.diffusion, self.x, t, torch.ones(5), c, self.noise
                )
                loss.backward()
                self.assertTrue(
                    all(p.grad is None for p in model.ml_model.parameters())
                )
                self.assertTrue(
                    any(p.grad is not None for p in model.ode_model.parameters())
                )
                optimizer.step()
                models.update_ema(ema, model, 0.9999)
                models.assert_frozen(model, optimizer, before)
                self.assertEqual(
                    common.state_hash(
                        {
                            k.removeprefix("ml_model."): v
                            for k, v in ema.items()
                            if k.startswith("ml_model.")
                        }
                    ),
                    before,
                )
                metadata = {
                    "effective_config": {
                        **c,
                        "edge_tsv_path": "/nonexistent/edges.tsv",
                    },
                    "gene_names": self.genes,
                    "frozen_cellunet_hash_before": before,
                    "checkpoint_kind": "ema",
                    "step": 1,
                }
                path = self.path / (condition + ".pt")
                cp.save_checkpoint(path, ema, metadata)
                restored, _ = cp.restore(path)
                self.assertEqual(common.state_hash(restored), common.state_hash(ema))
                # A non-sampling forward proves the full mask/gates/frozen ML restore.
                model.load_state_dict(ema)
                model.eval()
                torch.testing.assert_close(
                    restored(self.x, t[:, None]),
                    model(self.x, t[:, None]),
                    rtol=0,
                    atol=0,
                )

    def test_invariants_detect_illegal_cellunet_updates(self):
        c, model, before = self.hybrid(common.CONDITIONS[1])
        with self.assertRaises(AssertionError):
            models.assert_frozen(model, torch.optim.AdamW(model.parameters()))
        with torch.no_grad():
            next(model.ml_model.parameters()).add_(1)
        with self.assertRaises(AssertionError):
            models.assert_frozen(model, expected=before)

    def test_exact_source_interpolation_all_six(self):
        from ODE.ode_20260609_hybrid5x3 import UnifiedODEMLHybrid

        for name in common.CONDITIONS[1:]:
            with self.subTest(condition=name):
                _, model, _ = self.hybrid(name)
                model.eval()
                self.assertIsInstance(model, UnifiedODEMLHybrid)
                self.assertIs(type(model).forward, UnifiedODEMLHybrid.forward)
                t = torch.tensor([0, 1, 49, 500, 999])[:, None]
                branch = model.branch_outputs(self.x, t)
                weight = 1 - t.float() / 999
                torch.testing.assert_close(branch["ode_weight"], weight, rtol=0, atol=0)
                expected = weight * branch["ode_raw"] + (1 - weight) * branch["ml_raw"]
                torch.testing.assert_close(model(self.x, t), expected, rtol=0, atol=0)
                torch.testing.assert_close(expected[0], branch["ode_raw"][0])
                torch.testing.assert_close(expected[-1], branch["ml_raw"][-1])

    def test_soft_constraint_mask_orientation_and_parameter_gradients(self):
        for name in common.CONDITIONS[1:4]:
            c, model, _ = self.hybrid(name)
            field = model.ode_model
            self.assertEqual(float(field.mask[1, 0]), 1)
            self.assertEqual(float(field.mask[0, 1]), 0)
            param = field.penalty_parameter()
            with torch.no_grad():
                param.fill_(0.2)
            penalty = obj.soft_constraint(model, c)
            reference = 5 * (param * (1 - field.mask)).abs().mean()
            torch.testing.assert_close(penalty, reference, rtol=0, atol=0)
            penalty.backward()
            self.assertTrue(
                torch.equal(
                    param.grad[..., 1, 0], torch.zeros_like(param.grad[..., 1, 0])
                )
            )
            self.assertTrue(all(p.grad is None for p in model.ml_model.parameters()))
            for p in field.parameters():
                if p is not param:
                    self.assertIsNone(p.grad)

    def test_reconstruction_never_calls_ot(self):
        c, model, _ = self.hybrid(common.CONDITIONS[1])
        t = torch.tensor([0, 50, 100, 500, 999])
        with patch.object(
            obj, "sinkhorn_divergence", side_effect=AssertionError("OT called")
        ):
            loss, values = obj.training_loss(
                model, self.diffusion, self.x, t, torch.ones(5), c, self.noise
            )
        reference = self.diffusion.training_losses(model, self.x, t, noise=self.noise)[
            "loss"
        ].mean() + obj.soft_constraint(model, c)
        torch.testing.assert_close(loss, reference)
        self.assertEqual(values["sinkhorn"], {})

    def test_ot_no_mse_uses_pred_xstart_and_true_set(self):
        c, model, _ = self.hybrid(common.CONDITIONS[4])
        t = torch.tensor([0, 1, 10, 30, 49])
        with patch.object(
            self.diffusion, "training_losses", side_effect=AssertionError("MSE called")
        ):
            loss, _ = obj.training_loss(
                model, self.diffusion, self.x, t, torch.ones(5), c, self.noise
            )
        xt = self.diffusion.q_sample(self.x, t, noise=self.noise)
        pred = self.diffusion._predict_xstart_from_eps(xt, t, model(xt, t[:, None]))
        reference = ot.sinkhorn_divergence(pred, self.x, c["ot"]) + obj.soft_constraint(
            model, c
        )
        torch.testing.assert_close(loss, reference)
        with self.assertRaises(ValueError):
            obj.training_loss(
                model,
                self.diffusion,
                self.x,
                torch.full((5,), 50),
                torch.ones(5),
                c,
                self.noise,
            )

    def test_sinkhorn_permutation_invariance_and_debias(self):
        settings = self.stage1_config["ot"]
        a, b = self.x.double() / 3, self.x.double().roll(1, 0) / 3 + 0.02
        reference = ot.sinkhorn_divergence(a, b, settings)
        shuffled = ot.sinkhorn_divergence(
            a[[3, 1, 0, 4, 2]], b[[2, 4, 1, 3, 0]], settings
        )
        torch.testing.assert_close(reference, shuffled, atol=1e-12, rtol=1e-9)
        self.assertAlmostEqual(
            float(ot.sinkhorn_divergence(a, a, settings)), 0, places=12
        )
        self.assertGreater(float(reference), 0)

    def test_sinkhorn_gradient_and_gene_normalization(self):
        settings = self.stage1_config["ot"]
        a = (self.x.double() / 3).requires_grad_()
        b = a.detach() + 0.03
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: ot.sinkhorn_divergence(x, b, settings),
                (a,),
                eps=1e-6,
                atol=1e-4,
            )
        )
        loss = ot.sinkhorn_divergence(a, b, settings)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(float(a.grad.norm()), 0)
        torch.testing.assert_close(
            ot.cost_matrix(a, b),
            ot.cost_matrix(a.repeat(1, 3), b.repeat(1, 3)),
            atol=1e-12,
            rtol=1e-9,
        )

    def test_sinkhorn_validation_and_nonconvergence_fail(self):
        config = dict(self.stage1_config["ot"], max_iterations=1, tolerance=1e-15)
        with self.assertRaises(FloatingPointError):
            ot.sinkhorn_divergence(self.x, self.x * 3, config)
        for epsilon in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                ot.sinkhorn_divergence(self.x, self.x, dict(config, epsilon=epsilon))

    def test_snapshot_count_phases_and_physical_time(self):
        table = traj.snapshot_table(True)
        self.assertEqual(len(table), 22)
        self.assertEqual([r["reverse_step"] for r in table], list(range(50, 1101, 50)))
        self.assertEqual(table[18]["diffusion_t"], 50)
        self.assertEqual(table[19]["diffusion_t"], 0)
        self.assertEqual([r["diffusion_t"] for r in table[20:]], [None, None])
        self.assertEqual([r["ode_step"] for r in table[20:]], [50, 100])
        self.assertEqual(table[-1]["integration_time"], 0.1)
        baseline = traj.snapshot_table(False)
        self.assertEqual(len(baseline), 20)
        self.assertTrue(all(r["phase"] == "diffusion" for r in baseline))

    def test_euler_field_only_terminal_time_no_gradient(self):
        calls = []

        def ode(x, t):
            calls.append(t)
            return -x

        x = self.x.requires_grad_()
        after, field, values = traj.euler_step(ode, x, 0.001, 1e6)
        torch.testing.assert_close(after, x.detach() * 0.999)
        torch.testing.assert_close(field, -x.detach())
        self.assertFalse(after.requires_grad)
        self.assertTrue(torch.equal(calls[0], torch.zeros(5, 1, dtype=torch.long)))
        torch.testing.assert_close(
            values["displacement"], (after - x.detach()).double().norm(dim=1)
        )

    def test_nan_inf_and_divergence_fail_loudly(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(FloatingPointError):
                common.finite("bad", torch.tensor([bad]))
            with self.assertRaises(FloatingPointError):
                traj.euler_step(
                    lambda x, t: torch.full_like(x, bad), self.x, 0.001, 1e6
                )
        with self.assertRaises(FloatingPointError):
            traj.euler_step(lambda x, t: x * 1e10, self.x, 0.001, 1e6)
        with self.assertRaises(ValueError):
            traj.euler_step(lambda x, t: x, self.x, -1, 1e6)

    def test_joint_umap_exact_inputs_and_no_baseline_joint(self):
        table = traj.snapshot_table(True)
        self.assertEqual(umaps.joint_indices(table), [18, 19, 20, 21])
        captured = []

        def build(real, generated):
            captured.append(generated)
            return SimpleNamespace(obs={})

        real = SimpleNamespace(n_obs=2)
        states = np.broadcast_to(np.arange(22)[None, :, None], (3, 22, 4))
        preds = np.broadcast_to(100 + np.arange(20)[None, :, None], (3, 20, 4))
        result = umaps.embedding_inputs(
            real,
            states,
            preds,
            table,
            umaps.joint_indices(table),
            core=SimpleNamespace(build_sampling_anndata=build),
        )
        self.assertEqual(captured[0].shape, (12, 4))
        np.testing.assert_array_equal(
            captured[0][:, 0], np.repeat([118, 119, 20, 21], 3)
        )
        self.assertEqual(
            result.obs["time_label"],
            ["Real Erythropoietic"] * 2
            + ["Diffusion 950"] * 3
            + ["Diffusion 1000"] * 3
            + ["ODE +50"] * 3
            + ["ODE +100"] * 3,
        )
        with self.assertRaises(ValueError):
            umaps.joint_indices(traj.snapshot_table(False))

    def test_independent_fit_counts_with_fake_umap(self):
        # Mock dimensionality reduction only; this test creates CSV/JSON, no figures.
        calls = []

        def build(real, generated):
            return SimpleNamespace(
                obs={}, obsm={"X_umap": np.zeros((real.n_obs + len(generated), 2))}
            )

        def compute(combined, seed):
            calls.append(combined)
            return {"seed": seed}

        core = SimpleNamespace(
            build_sampling_anndata=build, compute_common_umap=compute
        )
        for hybrid, expected in ((False, 20), (True, 23)):
            output = common.new_dir(self.path / str(hybrid))
            calls.clear()
            table = traj.snapshot_table(hybrid)
            with patch.object(umaps, "umap_core", return_value=core):
                umaps.compute_embeddings(
                    SimpleNamespace(n_obs=2),
                    np.zeros((3, len(table), 4)),
                    np.zeros((3, 20, 4)),
                    table,
                    output,
                    seed=1234,
                )
            self.assertEqual(len(calls), expected)
            self.assertEqual(len(list(output.glob("*.csv"))), expected)
            self.assertFalse(list(output.glob("*.png")))

    def test_metric_definitions_match_both_source_helpers(self):
        for source in ("20260816", "20260803_ODE_hill_exp"):
            helpers = importlib.import_module(f"work.{source}.viz.analysis_helpers")
            a, b = self.x, self.noise.float()
            values = diag.per_cell_metrics(a, b)
            torch.testing.assert_close(
                values["pearson"], helpers._sample_corr(a, b), rtol=0, atol=0
            )
            torch.testing.assert_close(
                values["mse"], ((a - b) ** 2).mean(1), rtol=0, atol=0
            )
            torch.testing.assert_close(
                values["cosine"],
                torch.nn.functional.cosine_similarity(a, b, dim=1),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                helpers._sample_norm(a), a.norm(dim=1), rtol=0, atol=0
            )
        self.assertTrue(
            torch.equal(
                diag.per_cell_metrics(torch.ones(2, 4), torch.ones(2, 4))["pearson"],
                torch.zeros(2),
            )
        )

    def test_diagnostic_grids_and_schedule_annotation(self):
        full, low = diag.timestep_grids()
        self.assertEqual(full, list(range(0, 1000, 20)) + [999])
        self.assertEqual(low, list(range(51)))
        sigmas = np.sqrt(1 - self.diffusion.alphas_cumprod)
        self.assertLess(sigmas[10], 0.05)
        self.assertGreater(sigmas[49], 0.05)

    def test_diversity_exact_mean_and_squared_identity(self):
        x = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        values = dist.diversity(x, pairs=10000, seed=1234, block_size=2)
        self.assertAlmostEqual(values["mean"], 4)
        self.assertAlmostEqual(values["mean_squared"], 50 / 3)
        self.assertEqual(values["median"], 4)
        self.assertEqual(
            values, dist.diversity(x, pairs=10000, seed=1234, block_size=2)
        )

    def test_no_overwrite_and_output_confinement(self):
        path = self.path / "unique.json"
        common.write_json(path, {"original": True})
        with self.assertRaises(FileExistsError):
            common.write_json(path, {"original": False})
        self.assertTrue(json.loads(path.read_text())["original"])
        with self.assertRaises(FileExistsError):
            common.new_dir(self.path)
        with self.assertRaises(ValueError):
            common.write_json(ROOT / "outside.json", {})
        checkpoint = self.path / "checkpoint.pt"
        cp.save_checkpoint(checkpoint, self.stage1.state_dict(), {})
        before = common.file_hash(checkpoint)
        with self.assertRaises(FileExistsError):
            cp.save_checkpoint(checkpoint, self.stage1.state_dict(), {})
        self.assertEqual(common.file_hash(checkpoint), before)

    def test_canonical_stage1_rejects_tampering(self):
        path = self.path / "ema.pt"
        config = dict(self.stage1_config, total_steps=1)
        meta = {"effective_config": config, "checkpoint_kind": "ema", "step": 1}
        cp.save_checkpoint(path, self.stage1.state_dict(), meta)
        common.write_json(
            self.path / "canonical_stage1.json",
            {
                "checkpoint": str(path),
                "checkpoint_sha256": common.file_hash(path),
                "cellunet_hash": common.state_hash(self.stage1),
            },
        )
        cp.canonical_stage1(self.path)
        with path.open("ab") as f:
            f.write(b"tampered")
        with self.assertRaises(ValueError):
            cp.canonical_stage1(self.path)

    def test_canonical_settings_reject_architecture_or_representation_drift(self):
        for key, value in {
            "predict_xstart": True,
            "noise_schedule": "cosine",
            "ts_layer": "counts",
            "hybrid_norm_mode": "ratio_reg",
            "regime_gate_mode": "Ts_I_vs_II_III",
        }.items():
            with self.assertRaises(ValueError):
                common.validate_config(dict(self.stage1_config, **{key: value}))

    def test_small_diagnostic_csv_schema_without_plotting(self):
        import pandas as pd

        c, model, _ = self.hybrid(common.CONDITIONS[1])
        for name, tested in (("hybrid", model), ("baseline", self.stage1)):
            out = common.new_dir(self.path / name)
            with patch.object(diag, "timestep_grids", return_value=([0, 999], [0, 1])):
                diag.diagnostics(
                    tested,
                    self.diffusion,
                    self.x.numpy(),
                    out,
                    batch_size=3,
                    seed=1234,
                    division_epsilon=1e-12,
                )
            truth = pd.read_csv(out / "true_noise_metrics.csv")
            norms = pd.read_csv(out / "norm_metrics.csv")
            comparison = pd.read_csv(out / "cellunet_vs_ode_metrics.csv")
            self.assertEqual(set(truth.t), {0, 1, 999})
            self.assertEqual(set(truth.metric), {"pearson", "mse", "cosine"})
            self.assertTrue((truth.n_cells == 5).all())
            self.assertEqual(set(truth.phase), {"diffusion"})
            self.assertTrue(np.isfinite(truth["mean"]).all())
            if name == "hybrid":
                self.assertEqual(set(truth.series), {"hybrid", "cellunet", "ode"})
                self.assertTrue(
                    {
                        "weighted_cellunet",
                        "weighted_ode",
                        "hybrid_to_true_noise",
                    }.issubset(set(norms.series))
                )
                self.assertEqual(len(comparison), 9)
            else:
                self.assertEqual(set(truth.series), {"cellunet"})
                self.assertTrue(comparison.empty)
            self.assertFalse(list(out.glob("*.png")))

    def test_python_files_parse_without_executing_workflows(self):
        for path in common.SUITE.rglob("*.py"):
            if ".test_tmp" in path.parts:
                continue
            ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
