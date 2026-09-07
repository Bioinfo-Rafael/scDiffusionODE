#!/usr/bin/env python3
"""Focused unit/smoke checks for all three ODEs and four model families."""

from __future__ import annotations

import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


SUITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUITE_ROOT.parents[1]
for value in (REPO_ROOT, SUITE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from guided_diffusion.cell_model import Cell_Unet  # noqa: E402
from guided_diffusion.script_util import create_gaussian_diffusion  # noqa: E402
from models import factory  # noqa: E402


FAMILIES = factory.MODEL_FAMILIES
ODE_TYPES = ("hill_after_linear", "racipe", "exp")


class TinyCellUnet(nn.Module):
    """Cheap shape-compatible stand-in; one separate test covers real Cell_Unet."""

    def __init__(self, input_dim=2, **_kwargs):
        super().__init__()
        self.input_dim = int(input_dim)
        self.proj = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, x, t, y=None):
        del t, y
        return self.proj(x.float())


class ODEModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads((SUITE_ROOT / "configs" / "base.json").read_text())
        cls.genes = ["source", "target", "g2", "g3", "g4"]
        cls.mask = torch.zeros(5, 5)
        cls.mask[1, 0] = 1.0  # source -> target in [target, source] convention.
        cls.temp = tempfile.TemporaryDirectory(prefix="ode_hill_unit_")
        cls.edge_path = Path(cls.temp.name) / "edges.tsv"
        cls.edge_path.write_text("from\tto\nsource\ttarget\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def config(self, family, ode_type):
        config = dict(self.base)
        config.update({
            "experiment": f"{family}__{ode_type}",
            "model_family": family,
            "ode_type": ode_type,
            "edge_tsv_path": str(self.edge_path),
            "field_hidden": 16,
            "time_dim": 8,
        })
        if family == "ts_soft_tau80_hybrid_lincomb":
            config.update({
                "regime_gate_mode": "Ts_I_vs_II_III",
                "regime_gate_type": "sigmoid",
                "t_s": 10,
                "gate_tau": 80.0,
            })
        else:
            config.update({"regime_gate_mode": "none", "t_s": None})
        return config

    def test_all_twelve_construct_forward_backward_and_checkpoint(self):
        x = torch.randn(3, len(self.genes))
        t = torch.tensor([[0.0], [9.0], [19.0]])
        with mock.patch.object(factory, "Cell_Unet", TinyCellUnet):
            for family in FAMILIES:
                for ode_type in ODE_TYPES:
                    with self.subTest(family=family, ode_type=ode_type):
                        config = self.config(family, ode_type)
                        model = factory.build_model_from_config(config, self.genes, 20, "cpu")
                        model.train()
                        output = model(x, t)
                        self.assertEqual(tuple(output.shape), tuple(x.shape))
                        self.assertTrue(torch.isfinite(output).all())
                        loss = output.square().mean() + model.ode_model.off_mask_penalty("l1")
                        loss.backward()
                        for name, parameter in model.named_parameters():
                            self.assertTrue(torch.isfinite(parameter).all(), name)
                            if parameter.grad is not None:
                                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

                        model.eval()
                        reference = model(x, t).detach()
                        with tempfile.TemporaryDirectory(prefix="checkpoint_") as directory:
                            path = Path(directory) / "model000002.pt"
                            torch.save(model.state_dict(), path)
                            restored = factory.build_model_from_config(
                                config, self.genes, 20, "cpu"
                            ).eval()
                            restored.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
                            actual = restored(x, t).detach()
                            self.assertTrue(torch.allclose(reference, actual, rtol=1e-6, atol=1e-7))
                        del model, restored, output, loss
                        gc.collect()

    def test_penalty_targets_only_W_or_r_and_uses_full_mean(self):
        for ode_type, expected_name in (
            ("hill_after_linear", "W"), ("racipe", "r"), ("exp", "W")
        ):
            for family in ("ode_only_lincomb", "standard_hybrid_single"):
                with self.subTest(ode_type=ode_type, family=family):
                    config = self.config(family, ode_type)
                    ode = factory.build_ode_from_config(
                        config, self.genes, "cpu", mask=self.mask
                    )
                    edge_parameter = ode.penalty_parameter()
                    with torch.no_grad():
                        edge_parameter.copy_(torch.linspace(
                            -0.7, 0.9, edge_parameter.numel()
                        ).reshape_as(edge_parameter))
                    expected = config["off_mask_lambda"] * (
                        edge_parameter * (1.0 - self.mask)
                    ).abs().mean()
                    penalty = ode.off_mask_penalty("l1")
                    self.assertEqual(ode.penalty_parameter_name, expected_name)
                    self.assertTrue(torch.allclose(penalty, expected))
                    grads = torch.autograd.grad(
                        penalty, tuple(ode.parameters()), allow_unused=True
                    )
                    grad_names = {
                        name for (name, _), grad in zip(ode.named_parameters(), grads)
                        if grad is not None and bool((grad != 0).any())
                    }
                    expected_parameter_name = "r" if ode_type == "racipe" else "W"
                    self.assertEqual(grad_names, {expected_parameter_name})

    def test_racipe_neutral_factor_is_exact_one_with_live_gradient(self):
        config = self.config("ode_only_lincomb", "racipe")
        ode = factory.build_ode_from_config(config, self.genes, "cpu", mask=self.mask)
        with torch.no_grad():
            ode.r.zero_()
        x = torch.randn(2, len(self.genes))
        factors = ode.shifted_hill_factors(x)
        self.assertTrue(torch.equal(factors, torch.ones_like(factors)))
        factors.sum().backward()
        self.assertIsNotNone(ode.r.grad)
        self.assertTrue(torch.isfinite(ode.r.grad).all())
        self.assertGreater(float(ode.r.grad.abs().sum()), 0.0)
        self.assertEqual(float(ode.off_mask_penalty()), 0.0)
        self.assertTrue(torch.equal(ode.lambdas, torch.ones_like(ode.lambdas)))

    def test_toy_edge_direction_is_source_to_target(self):
        config = self.config("standard_hybrid_single", "exp")
        ode = factory.build_ode_from_config(config, self.genes, "cpu", mask=self.mask)
        with torch.no_grad():
            ode.W.zero_()
            ode.b.zero_()
            ode.raw_delta.fill_(-30.0)
            ode.W[1, 0] = 2.0  # allowed source -> target
            ode.W[0, 1] = 3.0  # deliberately off-mask reverse edge
        expected = config["off_mask_lambda"] * 3.0 / (len(self.genes) ** 2)
        self.assertAlmostEqual(float(ode.off_mask_penalty()), expected, places=6)
        x0 = torch.zeros(1, len(self.genes))
        x1 = x0.clone(); x1[0, 0] = 1.0
        delta = ode(x1) - ode(x0)
        self.assertGreater(float(delta[0, 1]), 0.0)

    def test_single_has_no_lincomb_modules_or_eight_coefficients(self):
        for ode_type in ODE_TYPES:
            lincomb = factory.build_ode_from_config(
                self.config("ode_only_lincomb", ode_type), self.genes, "cpu", mask=self.mask
            )
            single = factory.build_ode_from_config(
                self.config("standard_hybrid_single", ode_type), self.genes, "cpu", mask=self.mask
            )
            self.assertTrue(hasattr(lincomb, "coeff_net"))
            self.assertTrue(hasattr(lincomb, "time_emb"))
            self.assertFalse(hasattr(single, "coeff_net"))
            self.assertFalse(hasattr(single, "time_emb"))
            self.assertEqual(lincomb.num_components, 8)
            self.assertEqual(single.num_components, 1)
            self.assertGreater(
                sum(p.numel() for p in lincomb.parameters()),
                sum(p.numel() for p in single.parameters()),
            )

    def test_existing_standard_and_tau80_coefficients_are_unchanged(self):
        x = torch.zeros(3, len(self.genes))
        t = torch.tensor([[0.0], [10.0], [19.0]])
        with mock.patch.object(factory, "Cell_Unet", TinyCellUnet):
            standard = factory.build_model_from_config(
                self.config("standard_hybrid_single", "exp"), self.genes, 20, "cpu"
            )
            actual = standard.ode_branch_weight(x, t)
            expected = 1.0 - t / 19.0
            self.assertTrue(torch.equal(actual, expected))

            tau80 = factory.build_model_from_config(
                self.config("ts_soft_tau80_hybrid_lincomb", "exp"),
                self.genes, 20, "cpu",
            )
            actual = tau80.ode_branch_weight(x, t)
            expected = torch.sigmoid((10.0 - t) / 80.0)
            self.assertTrue(torch.equal(actual, expected))

    def test_weighted_branch_outputs_are_exact_forward_contributions(self):
        x = torch.randn(3, len(self.genes))
        t = torch.tensor([[0.0], [10.0], [19.0]])
        with mock.patch.object(factory, "Cell_Unet", TinyCellUnet):
            for family in ("standard_hybrid_single", "standard_hybrid_lincomb"):
                with self.subTest(family=family):
                    model = factory.build_model_from_config(
                        self.config(family, "hill_after_linear"),
                        self.genes,
                        20,
                        "cpu",
                    ).eval()
                    branches = model.branch_outputs(x, t)
                    expected_ode = branches["ode_weight"] * branches["ode_raw"]
                    expected_ml = branches["ml_weight"] * branches["ml_raw"]
                    self.assertTrue(torch.equal(branches["ode_contribution"], expected_ode))
                    self.assertTrue(torch.equal(branches["ml_contribution"], expected_ml))
                    self.assertTrue(torch.allclose(
                        model(x, t),
                        branches["ode_contribution"] + branches["ml_contribution"],
                        rtol=1e-6,
                        atol=1e-7,
                    ))

    def test_minimal_checkpoint_sampling_all_twelve(self):
        diffusion = create_gaussian_diffusion(steps=20, noise_schedule="cosine")
        with mock.patch.object(factory, "Cell_Unet", TinyCellUnet):
            for family in FAMILIES:
                for ode_type in ODE_TYPES:
                    with self.subTest(family=family, ode_type=ode_type):
                        model = factory.build_model_from_config(
                            self.config(family, ode_type), self.genes, 20, "cpu"
                        ).eval()
                        sample, _ = diffusion.p_sample_loop(
                            model,
                            (1, len(self.genes)),
                            clip_denoised=False,
                            start_time=diffusion.num_timesteps,
                        )
                        self.assertEqual(tuple(sample.shape), (1, len(self.genes)))
                        self.assertTrue(torch.isfinite(sample).all())

    def test_real_cell_unet_pipeline_shape(self):
        model = Cell_Unet(input_dim=len(self.genes)).eval()
        x = torch.randn(2, len(self.genes))
        t = torch.tensor([[0.0], [19.0]])
        with torch.no_grad():
            output = model(x, t)
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
