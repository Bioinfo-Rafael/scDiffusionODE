#!/usr/bin/env python3
"""Focused tests for the four direct-Hill single-ODE experiments."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn


SUITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUITE_ROOT.parents[1]
for value in (REPO_ROOT, SUITE_ROOT, SUITE_ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from models import factory  # noqa: E402
from scripts import common  # noqa: E402
from scripts.analyze_corr_norm_grid import _write_csv, timestep_grid  # noqa: E402


class TinyCellUnet(nn.Module):
    def __init__(self, input_dim=2, **_kwargs):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), int(input_dim))

    def forward(self, x, t, y=None):
        del t, y
        return self.proj(x.float())


class SingleODEDirectHillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads((SUITE_ROOT / "configs/base.json").read_text())
        cls.genes = ["source", "target", "g2", "g3", "g4"]
        cls.mask = torch.zeros(5, 5)
        cls.mask[1, 0] = 1.0
        cls.temp = tempfile.TemporaryDirectory(prefix="single_ode_hill_unit_")
        cls.edge_path = Path(cls.temp.name) / "edges.tsv"
        cls.edge_path.write_text("from\tto\nsource\ttarget\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def config(self, family: str, ode_type: str) -> dict:
        prefix = (
            "linear_singleODE"
            if family == "linear_hybrid_single_ode"
            else "ts_soft_tau80_singleODE"
        )
        config = dict(self.base)
        config.update({
            "experiment": f"{prefix}_{ode_type}",
            "model_family": family,
            "ode_type": ode_type,
            "edge_tsv_path": str(self.edge_path),
            "cell_unet_hidden_num": [16, 12, 8, 8],
            "target_chunk_size": 2,
        })
        if family == "ts_soft_tau80_hybrid_single_ode":
            config.update({
                "regime_gate_mode": "Ts_I_vs_II_III",
                "regime_gate_type": "sigmoid",
                "t_s": 10,
                "gate_tau": 80.0,
            })
        else:
            config.update({"regime_gate_mode": "none", "t_s": None})
        return config

    def field(self, ode_type: str):
        return factory.build_ode_from_config(
            self.config("linear_hybrid_single_ode", ode_type),
            self.genes,
            "cpu",
            mask=self.mask,
        )

    def test_centered_signed_hill_response_contract(self):
        field = self.field("centered_signed_hill")
        x = torch.tensor([0.0, -2.0, 0.5, 1.0, 2.0])
        x_pos = field.positive_regulators(x)
        theta = torch.ones_like(x)
        positive = field.edge_response(x_pos, theta, torch.ones_like(x))
        negative = field.edge_response(x_pos, theta, -torch.ones_like(x))
        neutral = field.edge_response(x_pos, theta, torch.zeros_like(x))
        self.assertTrue(torch.isfinite(positive).all())
        self.assertTrue(torch.isfinite(negative).all())
        self.assertTrue(torch.all(positive[1:] >= positive[:-1]))
        self.assertTrue(torch.all(negative[1:] <= negative[:-1]))
        self.assertTrue(torch.equal(neutral, torch.zeros_like(neutral)))
        self.assertEqual(float(positive[3]), 0.0)

    def test_shifted_hill_response_contract_and_rho_gradient(self):
        field = self.field("shifted_hill_rho")
        x = torch.tensor([0.0, -2.0, 0.5, 1.0, 2.0])
        x_pos = field.positive_regulators(x)
        theta = torch.ones_like(x)
        positive = field.edge_response(x_pos, theta, torch.ones_like(x))
        negative = field.edge_response(x_pos, theta, -torch.ones_like(x))
        neutral = field.edge_response(x_pos, theta, torch.zeros_like(x))
        self.assertTrue(torch.isfinite(positive).all())
        self.assertTrue(torch.isfinite(negative).all())
        self.assertTrue(torch.all(positive[1:] >= positive[:-1]))
        self.assertTrue(torch.all(negative[1:] <= negative[:-1]))
        self.assertTrue(torch.equal(neutral, torch.zeros_like(neutral)))
        rho = torch.zeros_like(x, requires_grad=True)
        field.edge_response(x_pos, theta, rho).sum().backward()
        self.assertTrue(torch.isfinite(rho.grad).all())
        self.assertGreater(float(rho.grad.abs().sum()), 0.0)

    def test_all_four_construct_forward_backward_and_are_finite(self):
        x = torch.tensor([
            [-1.0, 0.0, 0.5, 1.0, 2.0],
            [0.2, -0.3, 1.5, 0.0, 3.0],
        ])
        t = torch.tensor([[0.0], [19.0]])
        with mock.patch.object(factory, "Cell_Unet", TinyCellUnet):
            for family in factory.MODEL_FAMILIES:
                for ode_type in ("centered_signed_hill", "shifted_hill_rho"):
                    with self.subTest(family=family, ode_type=ode_type):
                        model = factory.build_model_from_config(
                            self.config(family, ode_type), self.genes, 20, "cpu"
                        )
                        output = model(x, t)
                        self.assertEqual(output.shape, x.shape)
                        self.assertTrue(torch.isfinite(output).all())
                        loss = output.square().mean() + model.ode_model.off_mask_penalty()
                        loss.backward()
                        for name, parameter in model.named_parameters():
                            if parameter.grad is not None:
                                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_exactly_one_ode_and_no_lincomb_gate(self):
        for ode_type in ("centered_signed_hill", "shifted_hill_rho"):
            field = self.field(ode_type)
            self.assertFalse(field.is_lincomb)
            self.assertEqual(field.num_components, 1)
            self.assertFalse(hasattr(field, "coeff_net"))
            self.assertFalse(hasattr(field, "time_emb"))
            self.assertEqual(field.A.shape, (len(self.genes), len(self.genes)))
            x = torch.randn(3, len(self.genes))
            components = field.component_outputs(x, torch.zeros(3, 1))
            self.assertEqual(components.shape, (3, 1, len(self.genes)))
            self.assertTrue(torch.equal(field(x), components[:, 0, :]))
            coefficients = field.effective_coefficients(x, None)
            self.assertTrue(torch.equal(coefficients, torch.ones(3, 1)))

    def test_nonnegative_A_and_signed_off_mask_parameter(self):
        for ode_type, signed_name in (
            ("centered_signed_hill", "alpha"),
            ("shifted_hill_rho", "rho"),
        ):
            field = self.field(ode_type)
            signed = getattr(field, signed_name)
            self.assertTrue(torch.all(field.A >= 0))
            self.assertEqual(field.penalty_parameter_name, signed_name)
            self.assertFalse(hasattr(field, "W"))
            with torch.no_grad():
                signed.copy_(torch.linspace(-0.7, 0.9, signed.numel()).reshape_as(signed))
            expected = self.base["off_mask_lambda"] * (
                signed * (1.0 - self.mask)
            ).abs().mean()
            self.assertTrue(torch.allclose(field.off_mask_penalty(), expected))
            field.zero_grad(set_to_none=True)
            field.off_mask_penalty().backward()
            self.assertIsNotNone(signed.grad)
            self.assertIsNone(field.raw_A.grad)

    def test_hybrid_weights_and_branch_identity(self):
        x = torch.randn(3, len(self.genes))
        t = torch.tensor([[0.0], [10.0], [19.0]])
        with mock.patch.object(factory, "Cell_Unet", TinyCellUnet):
            linear = factory.build_model_from_config(
                self.config("linear_hybrid_single_ode", "centered_signed_hill"),
                self.genes, 20, "cpu",
            ).eval()
            tau80 = factory.build_model_from_config(
                self.config("ts_soft_tau80_hybrid_single_ode", "centered_signed_hill"),
                self.genes, 20, "cpu",
            ).eval()
            self.assertTrue(torch.equal(linear.ode_branch_weight(x, t), 1.0 - t / 19.0))
            self.assertTrue(torch.equal(
                tau80.ode_branch_weight(x, t), torch.sigmoid((10.0 - t) / 80.0)
            ))
            branches = linear.branch_outputs(x, t)
            self.assertTrue(torch.allclose(
                branches["output"],
                branches["ode_contribution"] + branches["ml_contribution"],
            ))
            self.assertTrue(torch.allclose(linear(x, t), branches["output"]))

    def test_linear_vs_ts_only_changes_hybrid_policy(self):
        with mock.patch.object(factory, "Cell_Unet", TinyCellUnet):
            for ode_type in ("centered_signed_hill", "shifted_hill_rho"):
                linear = factory.build_model_from_config(
                    self.config("linear_hybrid_single_ode", ode_type), self.genes, 20, "cpu"
                )
                tau80 = factory.build_model_from_config(
                    self.config("ts_soft_tau80_hybrid_single_ode", ode_type),
                    self.genes, 20, "cpu",
                )
                self.assertEqual(
                    {name: tuple(p.shape) for name, p in linear.named_parameters()},
                    {name: tuple(p.shape) for name, p in tau80.named_parameters()},
                )
                self.assertEqual(linear.hybrid_norm_mode, tau80.hybrid_norm_mode)

    def test_timestep_grid_and_machine_readable_csv(self):
        self.assertEqual(timestep_grid(1000, 20), [*range(0, 1000, 20), 999])
        self.assertEqual(timestep_grid(20, 20), [0, 19])
        rows = [{"timestep": 0, "metric_mean": 1.0, "metric_std": 0.0}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timestep_metrics.csv"
            _write_csv(path, rows)
            with path.open(encoding="utf-8") as handle:
                restored = list(csv.DictReader(handle))
            self.assertEqual(restored[0]["timestep"], "0")
            self.assertTrue(np.isfinite(float(restored[0]["metric_mean"])))

    def test_config_order_and_training_steps(self):
        self.assertEqual(common.EXPERIMENT_ORDER, (
            "linear_singleODE_centered_signed_hill",
            "linear_singleODE_shifted_hill_rho",
            "ts_soft_tau80_singleODE_centered_signed_hill",
            "ts_soft_tau80_singleODE_shifted_hill_rho",
        ))
        self.assertEqual(self.base["K"], 1)
        self.assertEqual(self.base["gate_mode"], "none")
        self.assertEqual(self.base["total_steps"], 30000)
        self.assertEqual(self.base["lr_anneal_steps"], 30000)
        lincomb_base = json.loads(
            (REPO_ROOT / "work/20260816/configs/base.json").read_text()
        )
        intentional = {
            "schema_version", "K", "gate_mode", "ts_cache_path", "reference_sources"
        }
        for key in (set(self.base) & set(lincomb_base)) - intentional:
            self.assertEqual(self.base[key], lincomb_base[key], key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
