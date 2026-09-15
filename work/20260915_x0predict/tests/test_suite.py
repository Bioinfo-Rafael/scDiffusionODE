"""CPU synthetic tests; no canonical dataset or expensive GPU experiment."""
import copy
import importlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

PKG = "work.20260915_x0predict"
common = importlib.import_module(PKG + ".common")
models = importlib.import_module(PKG + ".models")
obj = importlib.import_module(PKG + ".training.objectives")
checkpoints = importlib.import_module(PKG + ".training.checkpoints")
runner = importlib.import_module(PKG + ".training.runner")
sampling = importlib.import_module(PKG + ".sampling.trajectory")
diagnostics = importlib.import_module(PKG + ".analysis.diagnostics")
launcher = importlib.import_module(PKG + ".scripts.run_all")
legacy = importlib.import_module("work.20260913_2step.common")
from guided_diffusion.gaussian_diffusion import ModelMeanType

torch.set_num_threads(1)
TEMP = common.SUITE / ".test_tmp"
TEMP.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TEMP / "matplotlib"))


class Constant(torch.nn.Module):
    def __init__(self, value=2.0):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value))

    def forward(self, x, t):
        self.last = torch.ones_like(x) * self.value
        return self.last


class SuiteTests(unittest.TestCase):
    def setUp(self):
        common.seed_all(1234)
        tmp = tempfile.TemporaryDirectory(dir=TEMP)
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name)
        self.edge = self.path / "edges.tsv"
        self.edge.write_text("from\tto\na\tb\nc\ta\n")
        self.genes = ["a", "b", "c", "d"]
        self.x = torch.rand(5, 4) + 0.5
        self.noise = torch.randn(5, 4, dtype=torch.float64)
        self.diffusion = common.build_diffusion(common.effective_config(common.STAGE1))

    def config(self, name):
        config = common.effective_config(name)
        config.update(cell_unet_hidden_num=[8, 8, 4, 4], time_dim=8, field_hidden=8,
                      edge_tsv_path=str(self.edge))
        return config

    def test_mode_and_matrix_and_canonical_defaults(self):
        self.assertEqual(len(common.CONDITIONS), 8)
        self.assertEqual(len(set(common.CONDITIONS)), 8)
        for condition in [common.STAGE1, *common.CONDITIONS]:
            c = common.effective_config(condition)
            self.assertIs(common.build_diffusion(c).model_mean_type, ModelMeanType.START_X)
            for key, expected in dict(total_steps=30000, batch_size=128, seed=1234,
                                      lr=1e-4, weight_decay=1e-4, ema_rate="0.9999",
                                      cell_unet_hidden_num=[2000, 1000, 500, 500]).items():
                self.assertEqual(c[key], expected)
            self.assertNotIn("trajectory_ot", c)

    def test_controlled_stage1_target_is_x0(self):
        model = Constant()
        t = torch.tensor([0, 49, 500, 750, 999])
        total, info = obj.training_loss(model, self.diffusion, self.x, t, torch.ones(5),
                                        self.config(common.STAGE1), self.noise)
        torch.testing.assert_close(total, ((self.x.double() - 2)**2).mean())
        self.assertEqual(info["soft"], 0)
        self.assertFalse(torch.isclose(total, ((self.noise - 2)**2).mean()))
        total.backward()
        self.assertIsNotNone(model.value.grad)

    def test_ot_consumes_raw_output_identity(self):
        config = self.config("simple_softplus_ot_soft")
        model = Constant()
        captured = []
        def sinkhorn(pred, target, settings, **kwargs):
            self.assertIs(pred, model.last)
            self.assertIs(target, self.x)
            captured.append(pred)
            return (pred - target).square().mean(), {}
        with patch.object(self.diffusion, "_predict_xstart_from_eps", side_effect=AssertionError("epsilon conversion")), \
                patch.object(obj, "soft_constraint", return_value=torch.tensor(0.)), \
                patch.object(obj, "sinkhorn_divergence", side_effect=sinkhorn):
            loss, _ = obj.training_loss(model, self.diffusion, self.x, torch.tensor([0, 1, 10, 30, 49]),
                                        torch.ones(5), config, self.noise)
            loss.backward()
        self.assertEqual(len(captured), 1)
        self.assertNotIn("_predict_xstart_from_eps", Path(obj.__file__).read_text())

    def test_timestep_support(self):
        for condition in [common.STAGE1, *common.CONDITIONS]:
            config = self.config(condition)
            sampler = obj.timestep_sampler(config, self.diffusion)
            size = 50 if config["objective"] == "ot" else 1000
            np.testing.assert_array_equal(np.flatnonzero(sampler.weights()), np.arange(size))
            t, weights = sampler.sample(20000, "cpu")
            self.assertEqual((int(t.min()), int(t.max())), (0, size - 1))
            torch.testing.assert_close(weights, torch.ones_like(weights))

    def test_schedule_all_four_and_no_adapter(self):
        t = torch.tensor([999, 750, 500, 250, 0])
        for family in common.FAMILIES:
            model = models.build_model(self.config(family + "_start_x_soft"), self.genes).eval()
            branches = model.branch_outputs(self.x, t)
            expected = torch.tensor([0., 0., 0., .5, 1.])[:, None]
            torch.testing.assert_close(branches["ode_weight"], expected)
            torch.testing.assert_close(branches["ml_weight"], 1 - expected)
            torch.testing.assert_close(model(self.x, t), expected * branches["ode_raw"] +
                                       (1 - expected) * branches["ml_raw"])

    def test_exact_historical_classes_and_masks(self):
        for family, (suite, class_name) in common.FAMILIES.items():
            c = self.config(family + "_start_x_soft")
            model = models.build_model(c, self.genes)
            ode = model.ode_model
            self.assertEqual(type(ode).__name__, class_name)
            self.assertTrue(type(ode).__module__.startswith("work." + suite))
            self.assertEqual(ode.num_components, 1)
            self.assertEqual(float(ode.mask[1, 0]), 1.)
            self.assertEqual(float(ode.mask[0, 1]), 0.)
            reference = common.source_factory(c).build_ode_from_config(c, self.genes)
            reference.load_state_dict(ode.state_dict(), strict=True)
            reference.eval(); ode.eval()
            torch.testing.assert_close(ode(self.x, torch.zeros(5, 1)), reference(self.x, torch.zeros(5, 1)))
            parameter = ode.penalty_parameter()
            expected = 5 * (parameter * (1 - ode.mask)).abs().mean()
            torch.testing.assert_close(ode.off_mask_penalty("l1"), expected)

    def test_simple_equation_initialization(self):
        c = self.config("simple_softplus_start_x_soft")
        ode = models.build_model(c, self.genes).ode_model
        torch.testing.assert_close(ode.input_scale, torch.tensor(.5))
        torch.testing.assert_close(ode.gamma, torch.full((4,), .1))
        expected = torch.nn.functional.softplus(.5 * (self.x @ ode.W.T + ode.b)) - \
            torch.nn.functional.softplus(ode.gamma) * self.x
        torch.testing.assert_close(ode(self.x), expected)
        self.assertEqual(ode.W.ndim, 2)
        self.assertFalse(hasattr(ode, "coeff_net"))

    def test_strict_single_ode_provenance_shapes_and_no_nn(self):
        expected_classes = {
            "simple_softplus": "SimpleSoftplus20260830",
            "hill_after_linear": "HillAfterLinear20260830",
            "centered_signed_hill": "CenteredSignedHill20260830",
            "shifted_hill_rho": "ShiftedHillRho20260830",
        }
        for family, cls in expected_classes.items():
            c = self.config(family + "_start_x_soft")
            ode = models.build_model(c, self.genes).ode_model
            self.assertEqual(type(ode).__module__, "work.20260830.models.ode_fields_20260830")
            self.assertEqual(type(ode).__name__, cls)
            self.assertEqual(c["ode_class"], cls)
            self.assertEqual(c["ode_source_suite"], "work/20260830")
            self.assertEqual(c["ode_source_file"], "work/20260830/models/ode_fields_20260830.py")
            self.assertEqual(c["ode_components"], 1)
            self.assertIs(c["expert_gating"], False)
            self.assertIs(ode.is_lincomb, False)
            self.assertEqual((ode.num_components, ode.num_experts), (1, 1))
            self.assertEqual([name for name, _ in ode.named_modules()], [""])
            for name, parameter in ode.named_parameters():
                self.assertFalse(any(token in name for token in ("coeff_net", "gate", "time_emb")))
                self.assertLessEqual(parameter.ndim, 2)
            if family in ("centered_signed_hill", "shifted_hill_rho"):
                for name in ("A", "theta", "alpha" if family == "centered_signed_hill" else "rho"):
                    self.assertEqual(tuple(getattr(ode, name).shape), (4, 4))
                self.assertEqual(tuple(ode.delta.shape), (4,))
            else:
                self.assertEqual(tuple(ode.W.shape), (4, 4))
            models.assert_single_ode(ode, family)

    def test_reject_old_k8_and_injected_gating(self):
        for family in ("centered_signed_hill", "shifted_hill_rho"):
            old = legacy.effective_config(family + "_recon_soft")
            old.update(edge_tsv_path=str(self.edge), time_dim=8, field_hidden=8)
            old_field = legacy.source_factory(old).build_ode_from_config(old, self.genes)
            with self.assertRaises(AssertionError):
                models.assert_single_ode(old_field, family)
            invalid = self.config(family + "_start_x_soft")
            invalid.update(source_suite="20260816", K=8, gate_mode="softmax")
            with self.assertRaises(AssertionError):
                models.build_model(invalid, self.genes)
        for family in common.FAMILIES:
            ode = models.build_model(self.config(family + "_start_x_soft"), self.genes).ode_model
            ode.coeff_net = torch.nn.Linear(4, 8)
            with self.assertRaises(AssertionError):
                models.assert_single_ode(ode, family)
            del ode.coeff_net
            ode.register_parameter("gate_parameter", torch.nn.Parameter(torch.zeros(4)))
            with self.assertRaises(AssertionError):
                models.assert_single_ode(ode, family)

    def test_all_four_deterministic_forward_formulas(self):
        F = torch.nn.functional
        x = torch.tensor([[-2., 0., .5, 2.], [.25, 1., 4., 8.]])
        for family in common.FAMILIES:
            with self.subTest(family=family):
                ode = models.build_model(self.config(family + "_start_x_soft"), self.genes).ode_model.eval()
                with torch.no_grad():
                    for index, parameter in enumerate(ode.parameters()):
                        parameter.copy_(torch.linspace(-.4 + index * .03, .5 + index * .03,
                                                       parameter.numel()).reshape_as(parameter))
                if family == "simple_softplus":
                    expected = F.softplus(ode.input_scale * (x @ ode.W.T + ode.b)) - F.softplus(ode.gamma) * x
                elif family == "hill_after_linear":
                    z = F.softplus(x @ ode.W.T + ode.b)
                    K, V = F.softplus(ode.raw_K) + ode.positive_epsilon, F.softplus(ode.raw_V) + ode.positive_epsilon
                    expected = V * z.square() / (K.square() + z.square()) - (F.softplus(ode.raw_delta) + ode.positive_epsilon) * x
                else:
                    A = F.softplus(ode.raw_A) + ode.positive_epsilon
                    theta = F.softplus(ode.raw_theta) + ode.positive_epsilon
                    positive_x = x.clamp_min(ode.positive_epsilon)
                    expected = torch.empty_like(x)
                    # Explicit target i / regulator j calculation: no expert axis.
                    for i in range(4):
                        production = ode.b[i].expand(len(x))
                        for j in range(4):
                            log_ratio = torch.log(positive_x[:, j]) - torch.log(theta[i, j])
                            response = (torch.tanh(ode.alpha[i, j] * log_ratio)
                                        if family == "centered_signed_hill" else
                                        torch.expm1(ode.rho[i, j]) * torch.sigmoid(2 * log_ratio))
                            production = production + A[i, j] * response
                        expected[:, i] = production - (F.softplus(ode.raw_delta[i]) + ode.positive_epsilon) * x[:, i]
                torch.testing.assert_close(ode(x, torch.zeros(2, 1)), expected, rtol=2e-6, atol=2e-6)
                torch.testing.assert_close(ode(x, torch.full((2, 1), 999)), expected, rtol=2e-6, atol=2e-6)

    def test_all_eight_backward_freeze_optimizer_ema_roundtrip(self):
        stage1 = models.build_model(self.config(common.STAGE1), self.genes)
        for condition in common.CONDITIONS:
            with self.subTest(condition=condition):
                c = self.config(condition)
                model = models.build_model(c, self.genes)
                before = models.freeze_from_stage1(model, stage1.state_dict())
                model.train(); model.ml_model.train(True)
                opt = models.optimizer_for(model, c)
                models.assert_frozen(model, opt, before)
                ema = copy.deepcopy(model.state_dict())
                t = torch.tensor([0, 1, 10, 30, 49]) if c["objective"] == "ot" else torch.tensor([0, 49, 250, 500, 999])
                loss, info = obj.training_loss(model, self.diffusion, self.x, t, torch.ones(5), c, self.noise)
                loss.backward()
                self.assertTrue(all(p.grad is None for p in model.ml_model.parameters()))
                self.assertTrue(any(p.grad is not None for p in model.ode_model.parameters()))
                opt.step(); models.update_ema(ema, model, .9999)
                models.assert_frozen(model, opt, before)
                self.assertEqual(common.state_hash({k.removeprefix('ml_model.'):v for k,v in ema.items()
                                                   if k.startswith('ml_model.')}), before)
                path = checkpoints.save_checkpoint(self.path / (condition + ".pt"), ema,
                          {"effective_config": c, "gene_names": self.genes, "frozen_cellunet_hash_before": before})
                restored, _ = checkpoints.restore(path)
                models.assert_frozen(restored, expected=before)
                self.assertEqual(common.state_hash(restored), common.state_hash(ema))
                self.assertGreaterEqual(info["soft"], 0)

    def test_high_t_data_gradient_inactive_but_soft_penalty_backward(self):
        c = self.config("simple_softplus_start_x_soft")
        model = models.build_model(c, self.genes).train()
        t = torch.full((5,), 750)
        data_loss = self.diffusion.training_losses(model, self.x, t, noise=self.noise)["loss"].mean()
        self.assertFalse(data_loss.requires_grad)
        total, _ = obj.training_loss(model, self.diffusion, self.x, t, torch.ones(5), c, self.noise)
        total.backward()
        self.assertIsNotNone(model.ode_model.W.grad)

    def test_native_sampling_and_saved_semantics(self):
        model = Constant(.25).eval()
        config = self.config(common.STAGE1)
        with patch.object(self.diffusion, "_predict_xstart_from_eps", side_effect=AssertionError("epsilon conversion")):
            for t in (0, 250, 999):
                out = self.diffusion.p_sample(model, self.x, torch.full((5,), t), clip_denoised=False)
                torch.testing.assert_close(out["pred_xstart"], torch.full_like(self.x, .25))
            output = sampling.sample_to_disk(model, self.diffusion, self.path / "sample", config,
                         count=2, batch_size=2, device="cpu", provenance={"gene_names":self.genes})
        self.assertTrue((output / "model_x0.npy").exists())
        self.assertFalse((output / "epsilon.npy").exists())
        np.testing.assert_allclose(np.load(output / "model_x0.npy"), .25)
        np.testing.assert_allclose(np.load(output / "sample_state.npy")[:, -1], .25)

    def test_diagnostics_target_and_norms(self):
        model = Constant(.25)
        real = self.x.numpy()
        with patch.object(diagnostics, "timestep_grids", return_value=([0, 250, 500, 750, 999], [0])):
            diagnostics.diagnostics(model, self.diffusion, real, self.path,
                                     batch_size=3, seed=1234, division_epsilon=1e-12)
        import pandas as pd
        table = pd.read_csv(self.path / "true_x0_metrics.csv")
        self.assertEqual(set(table.metric), {"pearson", "mse", "cosine", "l2", "norm_ratio"})
        mse = table[table.metric == "mse"]["mean"].to_numpy()
        np.testing.assert_allclose(mse, ((real - .25)**2).mean(), rtol=1e-6)
        self.assertFalse((self.path / "true_noise_metrics.csv").exists())

    def test_epsilon_rejected_and_old_defaults_unchanged(self):
        old = legacy.effective_config(common.STAGE1)
        old_diffusion = legacy.build_diffusion(old)
        self.assertIs(old_diffusion.model_mean_type, ModelMeanType.EPSILON)
        with self.assertRaises(AssertionError):
            obj.training_loss(Constant(), old_diffusion, self.x, torch.zeros(5, dtype=torch.long),
                              torch.ones(5), self.config(common.STAGE1))
        for condition in common.CONDITIONS:
            invalid = self.config(condition); invalid["predict_xstart"] = False
            with self.assertRaises(ValueError):
                common.build_diffusion(invalid)

    def test_distribution_analysis_and_plot_files(self):
        distributions = importlib.import_module(PKG + ".analysis.distributions")
        plotting = importlib.import_module(PKG + ".analysis.plotting")
        config = self.config(common.STAGE1)
        config.update(analysis_ot_cells=5, analysis_pairs=20)
        config["evaluation"].update(sliced_wasserstein_projections=4, sliced_wasserstein_points=5)
        real = self.x.numpy()
        with patch.object(diagnostics, "timestep_grids", return_value=([0, 500, 999], [0])):
            diagnostics.diagnostics(Constant(.25), self.diffusion, real, self.path,
                                     batch_size=5, seed=1234, division_epsilon=1e-12)
        table = sampling.snapshot_table(False)[:2]
        states = np.stack([real, real + .1], axis=1)
        distributions.trajectory_metrics(states, real, table, self.path, config, seed=1234)
        figures = common.new_dir(self.path / "figures")
        plotting.plot_numeric(self.path, figures)
        from PIL import Image
        for metric in ("pearson", "mse", "cosine", "l2", "norm_ratio"):
            path = figures / f"true_x0_metrics_{metric}_full.png"
            self.assertTrue(path.exists())
            with Image.open(path) as im:
                im.verify()
        self.assertTrue((figures / "sinkhorn_to_real.png").exists())

    def test_protected_repository_unchanged(self):
        baseline = common.read_json(common.SUITE / "audit/baseline.json")
        changed = [p for p, expected in baseline["sha256"].items()
                   if common.file_hash(common.ROOT / p) != expected]
        self.assertEqual(changed, [])

    def test_exclusive_outputs_and_path_confinement(self):
        common.write_json(self.path / "once.json", {})
        with self.assertRaises(FileExistsError):
            common.write_json(self.path / "once.json", {})
        for name in ("../old", "/tmp/old", "old_campaign"):
            with self.assertRaises(ValueError):
                common.campaign_path(name)
        with self.assertRaises(ValueError):
            common.new_dir(common.ROOT / "work/20260913_2step/new_run")

    def test_launcher_failure_continues_and_stage1_shared(self):
        calls = []
        def fake(campaign, name, argv, key):
            calls.append(name)
            if name == common.CONDITIONS[0] + "_train":
                raise RuntimeError("synthetic condition failure")
            return self.path / name
        args = launcher.parser().parse_args(["--all-eight", "--device", "cpu"])
        errors = launcher.execute(self.path, args, step_fn=fake)
        self.assertEqual(len(errors), 1)
        self.assertEqual(calls.count(common.STAGE1 + "_train"), 1)
        self.assertIn(common.CONDITIONS[-1] + "_embed_plot", calls)
        self.assertIn(common.STAGE1 + "_sample", calls)
        self.assertEqual(len([x for x in calls if x.endswith('_train')]), 9)

    def test_dry_run_creates_nothing(self):
        before = sorted(str(p) for p in common.SUITE.glob("runs/*"))
        self.assertEqual(launcher.main(["--all-eight", "--dry-run"]), 0)
        self.assertEqual(before, sorted(str(p) for p in common.SUITE.glob("runs/*")))

    def test_training_loop_resume_and_canonical_sharing(self):
        """2 Stage1 + 1 update per family/objective, synthetic four-gene CPU only."""
        campaign = self.path / "x0predict_toy"
        campaign.mkdir(); (campaign / "configs").mkdir()
        data_path = self.path / "synthetic.txt"; data_path.write_text("synthetic test data")
        for condition in [common.STAGE1, *common.CONDITIONS]:
            c = self.config(condition)
            c.update(total_steps=2 if condition == common.STAGE1 else 1, batch_size=5,
                     save_interval=1, data_dir=str(data_path), output_campaign=campaign.name)
            c["ot"]["batch_size"] = 5
            common.write_json(campaign / "configs" / (condition + ".json"), c)
        common.write_json(campaign / "source_sha256.json", common.source_provenance())
        def batches(**kwargs):
            while True:
                yield self.x.clone(), {}
        fake_data = SimpleNamespace(n_obs=5)
        with patch.object(runner, "campaign_path", return_value=campaign), \
             patch.object(runner, "load_real", return_value=(fake_data, self.genes, {})), \
             patch("guided_diffusion.cell_datasets_loader.load_data", side_effect=batches):
            real_loss = runner.training_loss
            counter = []
            def interrupted(*args, **kwargs):
                counter.append(1)
                if len(counter) == 2:
                    raise RuntimeError("synthetic interruption after checkpoint 1")
                return real_loss(*args, **kwargs)
            args = SimpleNamespace(campaign=campaign.name, condition=common.STAGE1, device="cpu")
            with patch.object(runner, "training_loss", side_effect=interrupted), self.assertRaises(RuntimeError):
                runner.train(args)
            self.assertEqual(runner.latest_bundle(campaign, common.STAGE1)["step"], 1)
            runner.train(args)
            _, origin = checkpoints.canonical_stage1(campaign)
            before = common.file_hash(origin["checkpoint"])
            # Idempotent completion: no extra run or optimization.
            stage1_attempts = list((campaign / common.STAGE1).iterdir())
            runner.train(args)
            self.assertEqual(stage1_attempts, list((campaign / common.STAGE1).iterdir()))
            for condition in common.CONDITIONS:
                args.condition = condition
                path = runner.train(args)
                meta = checkpoints.read_checkpoint(path)["metadata"]
                self.assertEqual(meta["originating_stage1"], origin)
                self.assertEqual(meta["frozen_cellunet_hash_before"], meta["frozen_cellunet_hash_after"])
            self.assertEqual(common.file_hash(origin["checkpoint"]), before)


if __name__ == "__main__":
    unittest.main()
