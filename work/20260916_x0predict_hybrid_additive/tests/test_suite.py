"""Small CPU tests, with one real 32768-target rectangular solve; no full experiment."""
import copy
import hashlib
import importlib
import json
import os
import signal
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
    def test_envelope_loss_gradient_and_no_iteration_graph(self):
        config = dict(c.effective_config("softplus_ot")["pca_ot"], tolerance=1e-11)
        pred = (torch.randn(3, 2, dtype=torch.float64) * .3).requires_grad_()
        target = torch.randn(5, 2, dtype=torch.float64) * .3
        with patch.object(obj, "checkpoint", side_effect=AssertionError("envelope must not checkpoint iterations")):
            value, info = obj.rectangular_sinkhorn(pred, target, config)
        gradient = torch.autograd.grad(value, pred)[0]
        reference, ref_info = obj.rectangular_sinkhorn(pred, target, dict(config, gradient_mode="full_autograd"))
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
        self.assertEqual(info["cross"]["iterations"], ref_info["cross"]["iterations"])
        torch.testing.assert_close(gradient, torch.autograd.grad(reference, pred)[0], rtol=1e-6, atol=1e-8)
        # Finite differences exercise both arguments of the prediction self term.
        numeric = torch.empty_like(pred)
        for row in range(len(pred)):
            for col in range(pred.shape[1]):
                plus, minus = pred.detach().clone(), pred.detach().clone()
                plus[row, col] += 1e-5
                minus[row, col] -= 1e-5
                numeric[row, col] = (obj.rectangular_sinkhorn(plus, target, config)[0]
                                     - obj.rectangular_sinkhorn(minus, target, config)[0]) / 2e-5
        torch.testing.assert_close(gradient, numeric, rtol=1e-5, atol=1e-7)
        self.assertIsNone(target.grad)
        self.assertEqual(info["cross"]["gradient_mode"], "envelope")
        with self.assertRaises(ValueError):
            obj.converged_entropic_ot(pred, target, dict(config, gradient_mode="invalid"))

    def test_8192_targets_remain_disjoint_and_differentiable(self):
        sampler = d.DisjointTargets(9000, 8192)
        source = np.arange(128)
        ids, info = sampler.sample(source, 0)
        source = ids[:128].copy()
        ids, info = sampler.sample(source, 1)
        self.assertEqual(len(np.unique(ids)), 8192)
        self.assertEqual(info["repaired"], 128)
        self.assertEqual(np.intersect1d(source, ids).size, 0)
        pred = (torch.rand(4, 50) * .1).requires_grad_()
        target = torch.rand(8192, 50) * .1
        with patch.object(obj, "checkpoint", side_effect=AssertionError("iteration graph")):
            value, info = obj.rectangular_sinkhorn(pred, target, c.effective_config("softplus_ot")["pca_ot"])
        value.backward()
        self.assertGreater(float(pred.grad.norm()), 0)
        self.assertEqual(info["cross"]["target_count"], 8192)

    def test_stop_campaign_selects_own_tree_and_rejects_reused_pid(self):
        stop = importlib.import_module(PKG + ".scripts.stop_campaign")
        campaign = "additive_test"
        processes = {
            100: ([PKG + ".scripts.run_all", "--resume-campaign", campaign], (1, "a", "S")),
            101: ([PKG + ".cli", "train", "--campaign", campaign], (100, "b", "R")),
            102: (["helper"], (101, "c", "S")),
            200: ([PKG + ".scripts.run_all", "--resume-campaign", "additive_other"], (1, "d", "R")),
            201: (["grep", campaign], (1, "e", "S")),
            300: ([PKG + ".cli", "analyze", "/results/additive_test/path"], (1, "f", "S")),
        }
        selected = stop.descendants(processes, stop.select(processes, campaign))
        self.assertEqual(selected, {100, 101, 102, 300})
        with patch.object(stop, "identity", return_value=(1, "reused", "R")), patch.object(stop.os, "kill") as kill:
            stop.send(100, signal.SIGTERM, processes)
            kill.assert_not_called()
        signals = []
        with patch.object(stop, "snapshot", return_value=processes), \
             patch.object(stop, "send", side_effect=lambda pid, sig, ps: signals.append((pid, sig))), \
             patch.object(stop, "alive", return_value=False), patch.object(stop.Path, "is_dir", return_value=True):
            self.assertEqual(stop.stop(campaign), 0)
        self.assertEqual(signals[0], (100, signal.SIGSTOP))
        self.assertIn((102, signal.SIGTERM), signals)
        self.assertEqual(signals[-2:], [(100, signal.SIGTERM), (100, signal.SIGCONT)])

    def test_resume_source_accepts_only_pinned_retry_changes(self):
        current = c.source_provenance()
        policy = c.read_json(c.SUITE / "audit/resume_compatibility.json")
        original = dict(current)
        for path, rule in policy["files"].items():
            self.assertEqual(current[path], rule["current_sha256"])
            original[path] = rule["previous_sha256"][0]
        self.assertIsNone(runner.validate_resume_source(current, current))
        migration = runner.validate_resume_source(original, current)
        self.assertEqual(set(migration["changed"]), set(policy["files"]))
        for path in [next(iter(policy["files"])), "work/20260916_x0predict_hybrid_additive/models.py"]:
            altered = dict(current, **{path: "unreviewed"})
            with self.assertRaises(ValueError):
                runner.validate_resume_source(original, altered)
        with self.assertRaises(ValueError):
            runner.validate_resume_source(dict(original, unknown="added"), current)
        with self.assertRaises(ValueError):
            runner.validate_resume_source(original, {k: v for k, v in current.items() if k not in policy["files"]})

    def test_sinkhorn_continuation_preserves_iterations_loss_and_gradient(self):
        torch.manual_seed(1234)
        config = dict(c.effective_config("softplus_ot")["pca_ot"], gradient_mode="full_autograd")
        pred = (torch.randn(4, 3, dtype=torch.float64) * 5).requires_grad_()
        target = torch.randn(7, 3, dtype=torch.float64) * 5
        with patch.object(obj.solver, "cost_matrix", wraps=obj.solver.cost_matrix) as cost, \
             patch.object(obj, "checkpoint", wraps=obj.checkpoint) as blocks:
            loss, info = obj.converged_entropic_ot(pred, target, config)
        self.assertEqual(cost.call_count, 1)
        self.assertEqual(sum(call.args[4] for call in blocks.call_args_list), info["iterations"])
        self.assertEqual([x["iterations"] for x in info["budget_extensions"]], [200, 400])
        self.assertEqual([x["next_limit"] for x in info["budget_extensions"]], [400, 600])
        self.assertEqual(info["restarts"], 0)
        gradient = torch.autograd.grad(loss, pred)[0]
        reference, ref_info = obj.solver.entropic_ot(pred, target, epsilon=config["epsilon"],
                                tolerance=config["tolerance"], max_iterations=16000)
        self.assertEqual(info["iterations"], ref_info["iterations"])
        torch.testing.assert_close(loss, reference, rtol=0, atol=0)
        torch.testing.assert_close(gradient, torch.autograd.grad(reference, pred)[0], rtol=0, atol=0)
        _, legacy_info = obj.converged_entropic_ot(pred, target, dict(config, max_iterations=2000))
        self.assertEqual(legacy_info["budget_extensions"][0]["iterations"], 200)
        with torch.no_grad():
            no_grad_loss, _ = obj.converged_entropic_ot(pred, target, config)
        torch.testing.assert_close(loss, no_grad_loss, rtol=0, atol=0)
        with self.assertRaisesRegex(obj.solver.SinkhornConvergenceError, "iterations=200"):
            obj.converged_entropic_ot(pred, target, dict(config, retry_max_iterations=200))
        with self.assertRaises(FloatingPointError):
            obj.converged_entropic_ot(pred * float("nan"), target, config)

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
            self.assertEqual((cfg["batch_size"], cfg["total_steps"], cfg["seed"]), (128, 10000, 1234))
            self.assertEqual((cfg["target_size"], cfg["target_refresh_interval"], cfg["pca"]["dimension"]), (8192, 10, 50))
            self.assertEqual(cfg["pca_ot"]["gradient_mode"], "envelope")

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
        sampler = d.DisjointTargets(40000, 32768)
        other = d.DisjointTargets(40000, 32768)
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
            d.DisjointTargets(32800, 32768).sample(np.arange(128), 0)

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
        cfg = dict(self.config("softplus_ot")["pca_ot"], gradient_mode="full_autograd")
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
            cfg["pca_ot"]["gradient_mode"] = "full_autograd"
            return cfg
        fake_campaign = self.path / "additive_toy"
        with patch.object(launcher, "effective_config", side_effect=small_config), \
             patch.object(launcher, "campaign_path", return_value=fake_campaign):
            campaign = launcher.prepare(args)
        # Simulate an existing campaign's source snapshot, before retry fixes.
        snapshot = campaign / "source_sha256.json"
        saved_source = c.read_json(snapshot)
        for filename, rule in c.read_json(c.SUITE / "audit/resume_compatibility.json")["files"].items():
            saved_source[filename] = rule["previous_sha256"][0]
        snapshot.write_text(json.dumps(saved_source))
        snapshot_hash = c.file_hash(snapshot)
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
            self.assertIsNotNone(meta["source_migration"])
            self.assertEqual(c.file_hash(snapshot), snapshot_hash)
            self.assertEqual(meta["frozen_cellunet_hash_before"], meta["frozen_cellunet_hash_after"])
            self.assertIsNotNone(meta["pca_provenance"])
            cache_hash = c.file_hash(Path(meta["pca_provenance"]["path"]) / "real_pca.npy")
            # Switch a legacy full-autograd/20-target run to envelope/8 targets.
            # Its final step=2 must not be relabelled: resume from step=1 and update.
            legacy_hash = c.file_hash(final)
            train_args.target_size = 8
            train_args.sinkhorn_gradient_mode = "envelope"
            expected_bundle = runner.latest_bundle_at_or_before(campaign, "softplus_ot", 1)
            expected_state = c.state_hash(cp.read_checkpoint(expected_bundle["raw"])["state_dict"])
            observations = []
            def switched_loss(model, *a, **kw):
                observations.append((c.state_hash(model), len(kw["target"])))
                return original_loss(model, *a, **kw)
            with patch.object(runner, "training_loss", side_effect=switched_loss):
                fast = runner.train(train_args)
            fast_meta = cp.read_checkpoint(fast)["metadata"]
            self.assertEqual(observations, [(expected_state, 8)])
            self.assertEqual(fast_meta["ot_transitions"][0]["first_new_update"], 2)
            self.assertEqual(fast_meta["ot_transitions"][0]["before"], dict(target_size=20, gradient_mode="full_autograd"))
            self.assertEqual(fast_meta["ot_transitions"][0]["after"], dict(target_size=8, gradient_mode="envelope"))
            self.assertEqual(c.file_hash(final), legacy_hash)
            self.assertEqual(runner.train(train_args), fast)
            resumed_config = fast_meta["effective_config"]
            fast_bundle = runner.latest_bundle_at_or_before(campaign, "softplus_ot", 2, config=resumed_config,
                saved_config=c.read_json(campaign / "configs/softplus_ot.json"))
            self.assertEqual(fast_bundle["ema"], str(fast))
            fast_opt = torch.load(fast_bundle["optimizer"], weights_only=True)
            self.assertTrue(all(int(s["step"]) == 2 for s in fast_opt["state"].values()))
            del train_args.target_size, train_args.sinkhorn_gradient_mode
            train_args.condition = "centered_hill_ot"
            another = cp.read_checkpoint(runner.train(train_args))["metadata"]
            self.assertEqual(another["pca_provenance"]["real_pca_sha256"], cache_hash)
            # Completed reconstruction must use its intermediate EMA for a shorter horizon.
            train_args.condition = "softplus_reconst"
            complete_reconst = runner.train(train_args)
            old_hash = c.file_hash(complete_reconst)
            earlier = runner.latest_bundle_at_or_before(campaign, train_args.condition, 1)
            earlier_ema_hash = c.state_hash(cp.read_checkpoint(earlier["ema"])["state_dict"])
            config_hash = c.file_hash(campaign / "configs/softplus_reconst.json")
            train_args.training_steps = 1
            with patch.object(runner, "training_loss", side_effect=AssertionError("must not train past existing target")):
                shortened = runner.train(train_args)
                self.assertEqual(runner.train(train_args), shortened)
            shortened_payload = cp.read_checkpoint(shortened)
            self.assertEqual(shortened_payload["metadata"]["step"], 1)
            self.assertEqual(shortened_payload["metadata"]["effective_config"]["total_steps"], 1)
            self.assertEqual(c.state_hash(shortened_payload["state_dict"]), earlier_ema_hash)
            self.assertEqual(c.file_hash(complete_reconst), old_hash)
            self.assertEqual(c.file_hash(campaign / "configs/softplus_reconst.json"), config_hash)
            timing_rows = [json.loads(line) for line in sorted((campaign / "centered_hill_ot").glob("*/timing.jsonl"))[-1].read_text().splitlines()]
            self.assertEqual(len(timing_rows), 2)
            self.assertGreaterEqual(timing_rows[-1]["backward_seconds"], 0)
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
        (self.path / "configs").mkdir()
        for condition in c.CONDITIONS:
            (self.path / "configs" / f"{condition}.json").write_text(json.dumps(c.effective_config(condition)))
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
        train_order = [name for name in calls if name.endswith("_train")]
        self.assertEqual(train_order, [condition + "_train" for condition in c.CONDITIONS])
        self.assertTrue(all(name.endswith("_reconst_train") for name in train_order[:4]))
        for previous, following in zip(c.CONDITIONS[1:], c.CONDITIONS[2:]):
            self.assertLess(calls.index(previous + "_embed_plot"), calls.index(following + "_train"))

    def test_shortened_horizon_uses_separate_step_markers(self):
        (self.path / "configs").mkdir()
        for condition in c.CONDITIONS:
            cfg = dict(c.effective_config(condition), total_steps=30000)
            (self.path / "configs" / f"{condition}.json").write_text(json.dumps(cfg))
        cfg = c.read_json(self.path / "configs/softplus_ot.json")
        reduced = c.training_config(cfg)
        self.assertEqual(reduced["total_steps"], 10000)
        self.assertEqual(reduced["lr_anneal_steps"], cfg["lr_anneal_steps"])
        self.assertEqual(c.condition_step_prefix(self.path, "softplus_ot"), "softplus_ot_s10000")
        with self.assertRaises(ValueError):
            c.training_config(cfg, 0)
        calls = []
        args = launcher.parser().parse_args(["--all-eight", "--training-steps", "10000", "--device", "cpu"])
        comparison = importlib.import_module(PKG + ".analysis.comparison")
        def fake(campaign, name, argv, key):
            calls.append(name)
            return self.path / name
        with patch.object(launcher, "canonical_stage1", return_value=({}, {"checkpoint": "stage1.pt"})), \
             patch.object(comparison, "compare") as compare:
            self.assertEqual(launcher.execute(self.path, args, step_fn=fake), [])
        self.assertIn("centered_hill_reconst_s10000_sample", calls)
        self.assertNotIn("centered_hill_reconst_sample", calls)
        self.assertEqual(calls.count("cellunet_only_sample"), 1)
        compare.assert_called_once_with(self.path, training_steps=10000, target_size=None, gradient_mode=None)

    def test_ot_variant_markers_and_per_model_reconst_first_pipeline(self):
        (self.path / "configs").mkdir()
        for condition in c.CONDITIONS:
            cfg = dict(c.effective_config(condition), total_steps=30000, target_size=32768)
            cfg["pca_ot"].pop("gradient_mode")  # original campaign version
            (self.path / "configs" / f"{condition}.json").write_text(json.dumps(cfg))
        opts = dict(target_size=8192, gradient_mode="envelope")
        self.assertEqual(c.condition_step_prefix(self.path, "softplus_ot", 10000, **opts),
                         "softplus_ot_s10000_n8192_envelope")
        self.assertEqual(c.condition_step_prefix(self.path, "softplus_reconst", 10000, **opts),
                         "softplus_reconst_s10000")
        calls = []
        def fake(campaign, name, argv, key):
            calls.append((name, argv))
            return self.path / name
        args = launcher.parser().parse_args(["--resume-campaign", "additive_test", "--all-eight",
            "--training-steps", "10000", "--target-size", "8192", "--sinkhorn-gradient-mode", "envelope"])
        comparison = importlib.import_module(PKG + ".analysis.comparison")
        with patch.object(launcher, "canonical_stage1", return_value=({}, {"checkpoint": "stage1.pt"})), \
             patch.object(comparison, "compare") as compare:
            self.assertEqual(launcher.execute(self.path, args, step_fn=fake), [])
        expected = ["cellunet_only_" + s for s in ("sample", "analyze", "analyze_plot", "embed", "embed_plot")]
        for condition in c.CONDITIONS:
            prefix = c.condition_step_prefix(self.path, condition, 10000, **opts)
            expected += [prefix + "_" + s for s in ("train", "sample", "analyze", "analyze_plot", "embed", "embed_plot")]
        self.assertEqual([name for name, argv in calls], expected)
        for name, argv in calls:
            if name.endswith("_train"):
                self.assertIn("--target-size", argv)
                self.assertIn("envelope", argv)
        compare.assert_called_once_with(self.path, training_steps=10000, **opts)

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
