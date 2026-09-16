"""Small CPU tests, with one real 32768-target rectangular solve; no full experiment."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

PKG = "work.20260916_x0predict_hybrid_additive"
c = importlib.import_module(PKG + ".common")
m = importlib.import_module(PKG + ".models")
d = importlib.import_module(PKG + ".data")
obj = importlib.import_module(PKG + ".training.objectives")
cp = importlib.import_module(PKG + ".training.checkpoints")
runner = importlib.import_module(PKG + ".training.runner")
launcher = importlib.import_module(PKG + ".launcher")
diag = importlib.import_module(PKG + ".analysis.diagnostics")
torch.set_num_threads(1)
TEMP = c.SUITE / ".test_tmp"
TEMP.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TEMP / "matplotlib"))


class Tests(unittest.TestCase):
    def test_sinkhorn_retry_preserves_settings_and_gradient(self):
        config = c.effective_config("softplus_ot")["pca_ot"]
        pred = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        target = torch.randn(7, 3, dtype=torch.float64)
        original = obj.solver.entropic_ot
        calls = []
        def delayed(x, y, **kwargs):
            calls.append(kwargs.copy())
            if len(calls) < 3:
                raise obj.solver.SinkhornConvergenceError("test iteration limit")
            return original(x, y, **kwargs)
        with patch.object(obj.solver, "entropic_ot", side_effect=delayed):
            loss, info = obj.converged_entropic_ot(pred, target, config)
        gradient = torch.autograd.grad(loss, pred)[0]
        reference, _ = original(pred, target, epsilon=config["epsilon"],
                                tolerance=config["tolerance"], max_iterations=2400)
        torch.testing.assert_close(gradient, torch.autograd.grad(reference, pred)[0], rtol=0, atol=0)
        self.assertEqual([x["max_iterations"] for x in calls], [2000, 2200, 2400])
        self.assertTrue(all(x["epsilon"] == .1 and x["tolerance"] == 1e-5 for x in calls))
        self.assertEqual(len(info["retries"]), 2)
        short_config = dict(config, max_iterations=1)
        recovered, recovery_info = obj.converged_entropic_ot(pred, target, short_config)
        self.assertGreater(len(recovery_info["retries"]), 0)
        self.assertLessEqual(recovery_info["marginal_residual"], config["tolerance"])
        self.assertTrue(torch.isfinite(torch.autograd.grad(recovered, pred)[0]).all())
        with patch.object(obj.solver, "entropic_ot", side_effect=obj.solver.SinkhornConvergenceError("limit")) as failed:
            with self.assertRaises(obj.solver.SinkhornConvergenceError):
                obj.converged_entropic_ot(pred, target, config)
        self.assertEqual(failed.call_count, 71)
        with patch.object(obj.solver, "entropic_ot", side_effect=FloatingPointError("nonfinite")) as invalid:
            with self.assertRaises(FloatingPointError):
                obj.converged_entropic_ot(pred, target, config)
        self.assertEqual(invalid.call_count, 1)

    def setUp(self):
        c.seed_all(1234)
        temporary = tempfile.TemporaryDirectory(dir=TEMP)
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.edge = self.path / "edges.tsv"
        self.edge.write_text("from\tto\na\tb\nc\ta\n")
        self.genes = ["a", "b", "c", "d"]
        self.x = torch.rand(5, 4) + .5

    def config(self, condition):
        config = c.effective_config(condition)
        config.update(cell_unet_hidden_num=[8, 8, 4, 4], edge_tsv_path=str(self.edge))
        return config

    def stage1(self):
        old_config = c.old.effective_config(c.old.STAGE1)
        old_config.update(cell_unet_hidden_num=[8, 8, 4, 4], edge_tsv_path=str(self.edge))
        data = self.path / "training.h5ad"
        if not data.exists():
            import anndata as ad
            a = ad.AnnData(np.random.default_rng(1).uniform(.1, 2., size=(40, 4)).astype(np.float32))
            a.var_names = self.genes
            a.var["gene_name"] = self.genes
            a.obs["celltype"] = "toy"
            a.obs["superclass"] = "Erythropoietic"
            a.write_h5ad(data)
        old_config["data_dir"] = str(data)
        model = m.old.build_model(old_config, self.genes)
        metadata = dict(effective_config=old_config, checkpoint_kind="ema", step=30000,
                        gene_names=self.genes, gene_order_hash=c.umap_core().gene_order_hash(self.genes),
                        data_sha256=c.file_hash(data), edge_tsv_sha256=c.file_hash(self.edge))
        path = self.path / "stage1.pt"
        torch.save(dict(state_dict=model.state_dict(), metadata=metadata), path)
        return path

    def test_all_eight_build_and_diffusion_and_defaults(self):
        self.assertEqual(len(c.CONDITIONS), 8)
        for condition in c.CONDITIONS:
            cfg = self.config(condition)
            model = m.build_model(cfg, self.genes)
            self.assertIsInstance(model, m.AdditiveHybrid500)
            m.assert_single_ode(model.ode_model, cfg["ode_type"])
            self.assertEqual(c.build_diffusion(cfg).model_mean_type.name, "START_X")
            self.assertEqual((cfg["batch_size"], cfg["total_steps"], cfg["seed"]), (128, 30000, 1234))
            self.assertEqual((cfg["target_size"], cfg["target_refresh_interval"], cfg["pca"]["dimension"]), (32768, 10, 50))

    def test_additive_boundary_and_frozen_cell(self):
        times = torch.tensor([999, 501, 500, 250, 0])
        for condition in c.CONDITIONS:
            cfg = self.config(condition)
            model = m.build_model(cfg, self.genes)
            before = c.state_hash(model.ml_model)
            model.train(); model.ml_model.train(True)
            opt = m.optimizer_for(model, cfg)
            m.assert_frozen(model, opt, before)
            parts = model.branch_outputs(self.x, times)
            weights = torch.tensor([0, 0, 0, .5, 1])[:, None]
            torch.testing.assert_close(parts["ml_weight"], torch.ones(5, 1))
            torch.testing.assert_close(model(self.x, times), parts["ml_raw"] + weights * parts["ode_raw"])
            torch.testing.assert_close(model(self.x, times)[:3], parts["ml_raw"][:3], rtol=0, atol=0)

    def test_legacy_blend_unchanged(self):
        cfg = self.config("centered_hill_reconst")
        old = m.old.build_model(c.legacy_config(cfg), self.genes).eval()
        new = m.build_model(cfg, self.genes, state=old.state_dict()).eval()
        t = torch.tensor([0, 250, 500, 750, 999])
        new.mode = "blend"
        torch.testing.assert_close(new(self.x, t), old(self.x, t), rtol=0, atol=0)
        new.mode = "additive"
        self.assertFalse(torch.allclose(new(self.x, t), old(self.x, t)))
        self.assertIs(type(old), m.old.SingleODEHybrid500)

    def test_actual_format_stage1_load_and_epsilon_rejection(self):
        path = self.stage1()
        before = c.file_hash(path)
        payload, origin = cp.load_stage1(path)
        self.assertEqual(origin["checkpoint_sha256"], before)
        self.assertEqual(c.file_hash(path), before)
        payload["metadata"]["effective_config"]["predict_xstart"] = False
        torch.save(payload, self.path / "epsilon.pt")
        with self.assertRaises(ValueError):
            cp.load_stage1(self.path / "epsilon.pt")

    def test_disjoint_32768_refresh_repair_and_reproducibility(self):
        sampler = d.DisjointTargets(40000)
        other = d.DisjointTargets(40000)
        target, info = sampler.sample(np.arange(128), 0)
        expected, _ = other.sample(np.arange(128), 0)
        np.testing.assert_array_equal(target, expected)
        self.assertEqual(len(target), 32768)
        self.assertTrue(info["refreshed"])
        source = target[:128].copy()  # force cached-target overlap
        repaired, info = sampler.sample(source, 1)
        expected, _ = other.sample(source, 1)
        np.testing.assert_array_equal(repaired, expected)
        self.assertEqual(info["repaired"], 128)
        self.assertFalse(info["refreshed"])
        self.assertEqual(len(np.intersect1d(repaired, source)), 0)
        self.assertEqual(len(np.unique(repaired)), 32768)
        np.testing.assert_array_equal(repaired[128:], target[128:])
        for step in range(2, 10):
            current, info = sampler.sample(source, step)
            np.testing.assert_array_equal(current, repaired)
            self.assertFalse(info["refreshed"])
        current, info = sampler.sample(source, 10)
        self.assertTrue(info["refreshed"])
        self.assertFalse(np.array_equal(current, repaired))
        self.assertEqual(len(np.intersect1d(current, source)), 0)
        with self.assertRaises(ValueError):
            d.DisjointTargets(32800).sample(np.arange(128), 0)

    def test_fixed_pca_cache_and_autograd(self):
        real = np.random.default_rng(42).normal(size=(53, 4)).astype(np.float32)
        cfg = dict(dimension=3, batch_size=16, whiten=False)
        provenance = dict(data_sha256="data", gene_order_hash="genes")
        path = d.fit_pca(real, cfg, self.path / "pca", provenance)
        transform, cached, info = d.load_pca(path, **provenance)
        np.testing.assert_allclose(transform(torch.from_numpy(real)).numpy(), cached, atol=1e-6)
        value = torch.tensor(real[:5], requires_grad=True)
        transform(value).square().sum().backward()
        self.assertGreater(value.grad.norm().item(), 0)
        self.assertEqual(list(transform.parameters()), [])
        with self.assertRaises(ValueError):
            d.load_pca(path, data_sha256="changed", gene_order_hash="genes")

    def test_pca_ot_gradient_all_four_odes_and_frozen_ema(self):
        transform = d.FixedPCA(np.zeros(4), np.eye(4)[:3])
        target = torch.rand(20, 3) * .2
        for condition in c.CONDITIONS:
            cfg = self.config(condition)
            model = m.build_model(cfg, self.genes).train()
            before = c.state_hash(model.ml_model)
            ema = copy.deepcopy(model.state_dict())
            optimizer = m.optimizer_for(model, cfg)
            t = torch.tensor([0, 5, 10, 20, 49])
            loss, _ = obj.training_loss(model, c.build_diffusion(cfg), self.x, t, torch.ones(5), cfg,
                                         pca=transform, target=target, noise=torch.zeros_like(self.x))
            if cfg["objective"] == "ot":
                # Prove data-term PCA->OT gradients, independently of soft penalty.
                prediction = model(self.x, t)
                primary, _ = obj.rectangular_sinkhorn(transform(prediction), target, cfg["pca_ot"])
                primary.backward()
                self.assertTrue(any(p.grad is not None and p.grad.norm() > 0 for p in model.ode_model.parameters()))
                optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step(); m.update_ema(ema, model, .9999)
            m.assert_frozen(model, optimizer, before)
            self.assertTrue(all(p.grad is None for p in model.ml_model.parameters()))
            self.assertEqual(c.state_hash({k.removeprefix('ml_model.'):v for k,v in ema.items() if k.startswith('ml_model.')}), before)

    def test_gradient_equivalence_to_full_sinkhorn(self):
        cfg = self.config("softplus_ot")["pca_ot"]
        prediction = (torch.rand(4, 3, dtype=torch.float64) * .2).requires_grad_()
        target = torch.rand(7, 3, dtype=torch.float64) * .2
        shifted, _ = obj.rectangular_sinkhorn(prediction, target, cfg)
        old_config = dict(epsilon=.1, max_iterations=2000, tolerance=1e-5,
                          cost="mean_squared_gene_distance", debias=True, compute_dtype="float64")
        full = obj.solver.sinkhorn_divergence(prediction, target, old_config)
        g1 = torch.autograd.grad(shifted, prediction, retain_graph=True)[0]
        g2 = torch.autograd.grad(full, prediction)[0]
        torch.testing.assert_close(g1, g2, rtol=0, atol=0)
        constant, _ = obj.solver.entropic_ot(target, target, epsilon=.1, max_iterations=2000, tolerance=1e-5)
        torch.testing.assert_close(shifted - full, .5 * constant)

    def test_real_32768_target_solve_has_no_target_square(self):
        config = self.config("softplus_ot")["pca_ot"]
        pred = (torch.rand(4, 50) * .1).requires_grad_()
        target = torch.rand(32768, 50) * .1
        shapes = []
        original = obj.solver.cost_matrix
        def checked(x, y):
            shapes.append((len(x), len(y)))
            self.assertNotEqual((len(x), len(y)), (32768, 32768))
            return original(x, y)
        with patch.object(obj.solver, "cost_matrix", side_effect=checked):
            value, _ = obj.rectangular_sinkhorn(pred, target, config)
            value.backward()
        self.assertEqual(shapes, [(4, 32768), (4, 4)])
        self.assertGreater(pred.grad.norm().item(), 0)

    def test_indexed_source_stream(self):
        real = np.arange(160, dtype=np.float32).reshape(40, 4)
        stream = d.source_batches(real, 5, 1234)
        seen = []
        for _ in range(8):
            values, indices = next(stream)
            np.testing.assert_array_equal(values.numpy(), real[indices])
            seen.extend(indices)
        self.assertEqual(sorted(seen), list(range(40)))

    def test_recorder_boundary_and_sampling_duplicate_guard(self):
        model = m.build_model(self.config("softplus_reconst"), self.genes).eval()
        recorder = diag.BranchRecorder(deduplicate=True)
        model.diagnostic_sink = recorder
        for t in [501, 500, 500, 499, 250, 0]:
            model(self.x, torch.full((5,), t))
        recorder.save(self.path)
        self.assertEqual(recorder.rows[500]["n"], 5)
        import pandas as pd
        frame = pd.read_csv(self.path / "sampling_branch_metrics.csv").set_index("raw_t")
        self.assertEqual(frame.loc[500, "r"], 0)
        self.assertEqual(frame.loc[250, "r"], .5)
        self.assertTrue((frame.cell_weight == 1).all())

    def test_reused_modules_are_isolated_and_native_sampling(self):
        new = importlib.import_module(PKG + ".sampling.trajectory")
        old = importlib.import_module("work.20260915_x0predict.sampling.trajectory")
        self.assertIsNot(new.sample_to_disk.__globals__, old.sample_to_disk.__globals__)
        self.assertIs(new.sample_to_disk.__globals__["new_dir"], c.new_dir)
        model = m.build_model(self.config("softplus_reconst"), self.genes).eval()
        diffusion = c.build_diffusion(self.config("softplus_reconst"))
        with patch.object(diffusion, "_predict_xstart_from_eps", side_effect=AssertionError("epsilon conversion")):
            t = torch.tensor([0, 1, 250, 500, 999])
            out = diffusion.p_mean_variance(model, self.x, t, clip_denoised=False)
            torch.testing.assert_close(out["pred_xstart"], model(self.x, t))
        with self.assertRaises(ValueError):
            c.new_dir(c.ROOT / "work/20260915_x0predict/forbidden")

    def test_checkpoint_prepare_and_two_step_resume(self):
        path = self.stage1()
        args = launcher.parser().parse_args(["--stage1-checkpoint", str(path), "--target-size", "20", "--pca-dimension", "3", "--device", "cpu"])
        # Production batch128 is intentionally reduced only for this synthetic fixture.
        original_config = launcher.effective_config
        def small_config(condition):
            cfg = original_config(condition)
            cfg.update(batch_size=5, total_steps=2, save_interval=1, log_interval=1)
            cfg["ot"]["batch_size"] = 5
            return cfg
        fake_campaign = self.path / "additive_toy"
        with patch.object(launcher, "effective_config", side_effect=small_config), \
             patch.object(launcher, "campaign_path", return_value=fake_campaign):
            campaign = launcher.prepare(args)
        original_bytes = c.file_hash(path)
        with patch.object(runner, "campaign_path", return_value=campaign):
            train_args = SimpleNamespace(campaign=campaign.name, condition="softplus_ot", device="cpu")
            original_loss = runner.training_loss
            counter = []
            def interrupted(*a, **kw):
                counter.append(1)
                if len(counter) == 2:
                    raise RuntimeError("synthetic interruption")
                return original_loss(*a, **kw)
            with patch.object(runner, "training_loss", side_effect=interrupted), self.assertRaises(RuntimeError):
                runner.train(train_args)
            final = runner.train(train_args)
            meta = cp.read_checkpoint(final)["metadata"]
            self.assertEqual(meta["step"], 2)
            self.assertEqual(meta["frozen_cellunet_hash_before"], meta["frozen_cellunet_hash_after"])
            self.assertIsNotNone(meta["pca_provenance"])
            cache_hash = c.file_hash(Path(meta["pca_provenance"]["path"]) / "real_pca.npy")
            train_args.condition = "centered_hill_ot"
            another = cp.read_checkpoint(runner.train(train_args))["metadata"]
            self.assertEqual(another["pca_provenance"]["real_pca_sha256"], cache_hash)
        self.assertEqual(c.file_hash(path), original_bytes)
        restored, _ = cp.restore(final)
        m.assert_frozen(restored, expected=meta["frozen_cellunet_hash_before"])

    def test_full_1000_timestep_sampling_smoke(self):
        cli = importlib.import_module(PKG + ".cli")
        cfg = self.config("softplus_reconst")
        cfg.update(num_samples=2, sample_batch_size=2, output_campaign="additive_smoke")
        model = m.build_model(cfg, self.genes).eval()
        meta = dict(effective_config=cfg, gene_names=self.genes, frozen_cellunet_hash_before=c.state_hash(model.ml_model))
        path = cp.save_checkpoint(self.path / "hybrid.pt", model.state_dict(), meta)
        with patch.object(cli, "result_root", return_value=self.path / "samples"):
            cli.main(["sample", "--checkpoint", str(path), "--device", "cpu"])
        outputs = list((self.path / "samples/softplus_reconst").glob("*/sampling_branch_metrics.csv"))
        import pandas as pd
        frame = pd.read_csv(outputs[0])
        self.assertEqual(len(frame), 1000)
        self.assertTrue((frame.n_cells == 2).all())
        self.assertTrue((frame.cell_weight == 1).all())

    def test_protected_files_unchanged(self):
        baseline = c.read_json(c.SUITE / "audit/baseline.json")
        # The author's unrelated dirty files are not committed. Accept their
        # original HEAD bytes on clean/remote checkouts, without weakening any
        # experiment/core file check. Final local audit separately checks exact
        # preservation of every worktree baseline byte.
        head = c.read_json(c.SUITE / "audit/preexisting_head_hashes.json")
        changed = [p for p, expected in baseline.items()
                   if c.file_hash(c.ROOT / p) not in {expected, head.get(p)}]
        self.assertEqual(changed, [])

    def test_launcher_reuses_one_baseline_and_continues_failures(self):
        calls = []
        def fake(campaign, name, argv, key):
            calls.append(name)
            if name == "centered_hill_reconst_train":
                raise RuntimeError("synthetic failure")
            return self.path / name
        args = launcher.parser().parse_args(["--all-eight", "--device", "cpu"])
        comparison = importlib.import_module(PKG + ".analysis.comparison")
        with patch.object(launcher, "canonical_stage1", return_value=({}, {"checkpoint": "stage1.pt"})), \
             patch.object(comparison, "compare"):
            errors = launcher.execute(self.path, args, step_fn=fake)
        self.assertEqual(len(errors), 1)
        self.assertEqual(calls.count("cellunet_only_sample"), 1)
        self.assertNotIn("cellunet_only_train", calls)
        self.assertEqual(len([x for x in calls if x.endswith("_train")]), 8)
        self.assertIn("softplus_ot_embed_plot", calls)

    def test_evaluation_numeric_plot_and_small_umap(self):
        import anndata as ad
        dist = importlib.import_module(PKG + ".analysis.distributions")
        plotting = importlib.import_module(PKG + ".analysis.plotting")
        umaps = importlib.import_module(PKG + ".analysis.umaps")
        cfg = self.config("softplus_reconst")
        cfg.update(analysis_pairs=20, analysis_ot_cells=5)
        cfg["evaluation"].update(sliced_wasserstein_projections=4, sliced_wasserstein_points=5)
        real = np.random.default_rng(3).normal(size=(35, 4)).astype(np.float32)
        states = real[:5, None, :] + .1
        table = [dict(snapshot_index=0, reverse_step=1000, phase="diffusion", diffusion_t=0,
                      ode_step=None, integration_time=None, ode_conditioning_t=None)]
        model = m.build_model(cfg, self.genes).eval()
        with patch.object(diag.legacy, "timestep_grids", return_value=([0, 500, 999], [0])):
            diag.diagnostics(model, c.build_diffusion(cfg), real[:5], self.path,
                             batch_size=5, seed=1234, division_epsilon=1e-12)
        dist.trajectory_metrics(states, real, table, self.path, cfg, seed=1234)
        plotting.plot_numeric(self.path, c.new_dir(self.path / "plots"))
        self.assertTrue((self.path / "plots/true_x0_metrics_pearson_full.png").is_file())
        import pandas as pd
        collapse = pd.read_csv(self.path / "collapse_coverage.csv").iloc[0]
        self.assertAlmostEqual(collapse.aggregate_gene_variance, float(states[:, 0].var(0).mean()), places=6)
        self.assertTrue(0 <= collapse.added_real_knn_radius_coverage <= 1)
        reference = ad.AnnData(real)
        reference.var_names = self.genes
        reference.obs["celltype"] = "toy"
        embeddings = c.new_dir(self.path / "embeddings")
        umaps.compute_embeddings(reference, states, states, table, embeddings, seed=1234)
        plotting.plot_embeddings(embeddings, c.new_dir(embeddings / "plots"))
        self.assertTrue((embeddings / "plots/independent_1000.png").is_file())


if __name__ == "__main__":
    unittest.main()
