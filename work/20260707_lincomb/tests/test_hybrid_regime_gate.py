import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ODE.ode_20260609_hybrid5x3 import UnifiedODEMLHybrid

WORK_0707 = os.path.join(REPO, "work", "20260707_lincomb")
if WORK_0707 not in sys.path:
    sys.path.insert(0, WORK_0707)
from scripts.common import read_json, update_manifest


class DummyODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.soft = True
        self.ratio_reg_eps = 1e-8
        self.ratio_reg_target = 1.0
        self._cached_ratio_reg = None

    def forward(self, x, t=None):
        value = torch.tensor([1.0, 0.0], device=x.device) * self.weight
        return value.expand(x.shape[0], -1)

    def off_mask_penalty(self, norm="l1"):
        return self.weight.new_zeros(())


class DummyML(nn.Module):
    hidden_num = [2]

    def forward(self, x, t, y=None):
        return torch.tensor([0.0, 2.0], device=x.device).expand(x.shape[0], -1)

    def forward_with_features(self, x, t, y=None):
        return self.forward(x, t, y), {"ml_emb": x}


class DummyScale(nn.Module):
    def forward(self, x, t):
        return torch.full((x.shape[0], 1), 3.0, device=x.device)


def build(mode, reverse=False, regime="none", gate_type="sigmoid", t_s=None, tau=20.0):
    kwargs = {}
    if mode == "scale_model":
        kwargs.update(
            scale_model=DummyScale(), scale_input_source="x", ode_input_source="none"
        )
    return UnifiedODEMLHybrid(
        DummyODE(), DummyML(), timesteps=1000, hybrid_norm_mode=mode,
        reverse_coef=reverse, regime_gate_mode=regime,
        regime_gate_type=gate_type, t_s=t_s, gate_tau=tau, **kwargs,
    )


class HybridBlendTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.zeros(3, 2)
        self.t = torch.tensor([[0.0], [500.0], [999.0]])

    def _terms(self, mode):
        ode = torch.tensor([[1.0, 0.0]]).expand(3, -1)
        ml = torch.tensor([[0.0, 2.0]]).expand(3, -1)
        if mode in ("normed_learned_scale", "scale_model"):
            ode = ode / ode.norm(dim=-1, keepdim=True)
            ml = ml / ml.norm(dim=-1, keepdim=True)
        return ode, ml

    def test_reverse_false_matches_legacy_formula(self):
        model = build("none", reverse=False).eval()
        output = model(self.x, self.t)
        ode, ml = self._terms("none")
        r = 1.0 - self.t / 999.0
        self.assertTrue(torch.equal(output, r * ode + (1.0 - r) * ml))

    def test_reverse_all_modes(self):
        for mode in ("none", "ratio_reg", "normed_learned_scale", "scale_model"):
            with self.subTest(mode=mode):
                model = build(mode, reverse=True).eval()
                output = model(self.x, self.t)
                ode, ml = self._terms(mode)
                r = 1.0 - self.t / 999.0
                expected = (1.0 - r) * ode + r * ml
                if mode == "scale_model":
                    expected = 3.0 * expected
                self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_regime_sigmoid_endpoints(self):
        model = build(
            "none", regime="Ts_I_vs_II_III", gate_type="sigmoid", t_s=600, tau=20
        )
        weights = model._regime_ode_weight(
            torch.tensor([999.0, 600.0, 0.0]), torch.device("cpu"), torch.float32
        ).squeeze(1)
        self.assertLess(float(weights[0]), 0.1)
        self.assertAlmostEqual(float(weights[1]), 0.5, places=6)
        self.assertGreater(float(weights[2]), 0.9)

    def test_regime_hard(self):
        model = build("none", regime="Ts_I_vs_II_III", gate_type="hard", t_s=5)
        weights = model._regime_ode_weight(
            torch.tensor([[4.0], [5.0], [6.0]]), torch.device("cpu"), torch.float32
        )
        self.assertTrue(torch.equal(weights, torch.tensor([[1.0], [1.0], [0.0]])))

    def test_regime_applies_to_all_modes(self):
        for mode in ("none", "ratio_reg", "normed_learned_scale", "scale_model"):
            with self.subTest(mode=mode):
                model = build(mode, regime="Ts_I_vs_II_III", t_s=500, tau=20).eval()
                output = model(self.x, self.t)
                ode, ml = self._terms(mode)
                w = torch.sigmoid((500.0 - self.t) / 20.0)
                expected = w * ode + (1.0 - w) * ml
                if mode == "scale_model":
                    expected *= 3.0
                self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_regime_validation(self):
        with self.assertRaises(ValueError):
            build("none", regime="Ts_I_vs_II_III", t_s=None)
        with self.assertRaises(ValueError):
            build("none", regime="Ts_I_vs_II_III", t_s=10, tau=0)
        with self.assertRaises(ValueError):
            build("none", regime="Ts_I_vs_II_III", t_s=1000)
        with self.assertRaises(ValueError):
            build("none", reverse=True, regime="Ts_I_vs_II_III", t_s=10)

    def test_checkpoint_keys_unchanged_by_flags(self):
        current = build("none", reverse=False)
        reverse = build("none", reverse=True)
        self.assertEqual(set(current.state_dict()), set(reverse.state_dict()))
        reverse.load_state_dict(current.state_dict(), strict=True)
        info = reverse.get_model_info()
        self.assertTrue(info["reverse_coef"])
        self.assertEqual(info["regime_gate_mode"], "none")

    def test_checkpoint_metadata_and_tiny_training_smoke(self):
        model = build("none", regime="Ts_I_vs_II_III", t_s=500, tau=20).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        target = torch.randn_like(self.x)
        loss = torch.nn.functional.mse_loss(model(self.x, self.t), target)
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "model.pt")
            torch.save(model.state_dict(), checkpoint)
            restored = build("none", regime="Ts_I_vs_II_III", t_s=500, tau=20)
            restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
            info = restored.get_model_info()
            self.assertEqual(info["regime_gate_mode"], "Ts_I_vs_II_III")
            self.assertEqual(info["t_s"], 500.0)
            update_manifest(
                directory, reverse_coef=info["reverse_coef"],
                regime_gate_mode=info["regime_gate_mode"], t_s=info["t_s"],
                gate_tau=info["gate_tau"],
            )
            manifest = read_json(os.path.join(directory, "manifest.json"))
            self.assertIn("reverse_coef", manifest)
            self.assertEqual(manifest["t_s"], 500.0)


if __name__ == "__main__":
    unittest.main()
