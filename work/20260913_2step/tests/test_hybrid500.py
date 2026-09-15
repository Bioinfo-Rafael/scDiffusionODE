"""Synthetic CPU checks for opt-in interpolation and unchanged objectives."""

import copy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

PKG = "work.20260913_2step"
common = importlib.import_module(PKG + ".common")
models = importlib.import_module(PKG + ".models")
hybrid = importlib.import_module(PKG + ".models.hybrid500")
objectives = importlib.import_module(PKG + ".training.objectives")
trajectory = importlib.import_module(PKG + ".training.trajectory")
checkpoints = importlib.import_module(PKG + ".training.checkpoints")
torch.set_num_threads(1)


class Hybrid500Tests(unittest.TestCase):
    def setUp(self):
        parent = common.SUITE / ".test_tmp"
        parent.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.edge = self.path / "edge.tsv"
        self.edge.write_text("from\tto\na\tb\n")
        self.genes = ["a", "b", "c", "d"]

    def config(self, condition, mode="hybrid500_ot_v1"):
        c = common.effective_config(condition, mode)
        c.update(
            cell_unet_hidden_num=[8, 8, 4, 4],
            field_hidden=8,
            time_dim=8,
            edge_tsv_path=str(self.edge),
        )
        return c

    def test_weights_boundaries_and_invalid_time(self):
        t = torch.tensor([999, 750, 500, 250, 0])
        torch.testing.assert_close(
            hybrid.ode_weight(t), torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0])
        )
        for t in (-1, 1000, float("nan")):
            with self.assertRaises(ValueError):
                hybrid.ode_weight(t)

    def test_all_architectures_mixed_times_gradients_and_freezing(self):
        for condition in common.LEGACY_CONDITIONS:
            c = self.config(condition)
            model = models.build_model(c, self.genes).train()
            opt = models.optimizer_for(model, c)
            frozen = common.state_hash(model.ml_model)
            x = torch.rand(5, 4)
            t = torch.tensor([999, 750, 500, 250, 0])[:, None]
            model.eval()
            with patch.object(
                model.ode_model, "forward", wraps=model.ode_model.forward
            ) as field:
                model(x, t)
                self.assertEqual(field.call_args.args[0].shape[0], 2)
            branches = model.branch_outputs(x, t)
            expected = model.ml_model(x, t) * (1 - hybrid.ode_weight(t))
            expected[3:] += model.ode_model(x[3:], t[3:]) * hybrid.ode_weight(t[3:])
            torch.testing.assert_close(model(x, t), expected)
            torch.testing.assert_close(branches["output"], expected)
            with patch.object(
                model.ode_model, "forward", side_effect=AssertionError("inactive ODE")
            ):
                torch.testing.assert_close(
                    model(x[:3], t[:3]), model.ml_model(x[:3], t[:3])
                )
            model.train()
            opt.zero_grad()
            model(x, t).square().mean().backward()
            self.assertTrue(
                any(p.grad is not None for p in model.ode_model.parameters())
            )
            opt.step()
            models.assert_frozen(model, opt, frozen)

    def test_old_restore_retains_old_schedule_new_restore_retains_new(self):
        condition = common.LEGACY_CONDITIONS[0]
        for mode, expected in ((None, 1 - 750 / 999), ("hybrid500_ot_v1", 0)):
            c = self.config(condition, mode)
            model = models.build_model(c, self.genes)
            path = self.path / (str(mode) + ".pt")
            checkpoints.save_checkpoint(
                path,
                model.state_dict(),
                dict(
                    effective_config=c,
                    gene_names=self.genes,
                    frozen_cellunet_hash_before=common.state_hash(model.ml_model),
                ),
            )
            restored, _ = checkpoints.restore(path)
            weight = restored.ode_branch_weight(torch.ones(1, 4), torch.tensor([[750]]))
            self.assertAlmostEqual(float(weight), expected, places=6)

    def test_numerical_settings_unchanged_and_no_mse(self):
        for condition in common.LEGACY_CONDITIONS + common.CONDITIONS[4:]:
            old = common.effective_config(condition)
            new = common.effective_config(condition, "hybrid500_ot_v1")
            self.assertEqual(old["ot"], new["ot"])
            self.assertEqual(old.get("trajectory_ot"), new.get("trajectory_ot"))
        c = self.config(common.LEGACY_CONDITIONS[0])
        model = models.build_model(c, self.genes)
        diffusion = common.build_diffusion(c)
        x, t = torch.rand(3, 4), torch.tensor([0, 20, 49])
        with (
            patch.object(
                diffusion, "training_losses", side_effect=AssertionError("MSE")
            ),
            patch.object(
                objectives, "sinkhorn_divergence", return_value=(x.new_tensor(2.0), {})
            ),
        ):
            total, values = objectives.training_loss(
                model, diffusion, x, t, torch.ones(3), c
            )
        self.assertAlmostEqual(float(total), 2 + values["soft"], places=5)
        c = self.config(common.CONDITIONS[4])
        c["trajectory_ot"].update(
            trajectory_batch_size=2,
            real_target_points=2,
            ode_steps=4,
            checkpoint_block=0,
        )
        model = models.build_model(c, self.genes)
        with (
            patch.object(model, "forward", side_effect=AssertionError("Hybrid/MSE")),
            patch.object(
                trajectory, "sinkhorn_divergence", return_value=(x.new_tensor(3.0), {})
            ),
        ):
            total, values, _, _ = trajectory.trajectory_loss(
                model, x[:2], x[:2], c, step=2
            )
        self.assertAlmostEqual(float(total), 3 + values["soft"], places=5)

    def test_reject_rescaled_diffusion_and_wrong_schedule(self):
        c = self.config(common.LEGACY_CONDITIONS[0])
        for key, value in (("rescale_timesteps", True), ("diffusion_steps", 500)):
            changed = copy.deepcopy(c)
            changed[key] = value
            with self.assertRaises(ValueError):
                models.build_model(changed, self.genes)
