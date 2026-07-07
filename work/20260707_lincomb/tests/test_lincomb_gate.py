import os
import sys
import unittest

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ODE.ode_20260707_lincomb import ConfigurableLinCombField, LinCombOnlyDenoiser
from ODE.ode_20260609_mathmlp import LinCombField


class LinCombGateTest(unittest.TestCase):
    def test_raw_matches_legacy_lincomb(self):
        torch.manual_seed(11)
        legacy = LinCombField(4, K=3, mask=None, soft=True).eval()
        configurable = ConfigurableLinCombField(
            4, K=3, mask=None, soft=True, gate_mode="raw"
        ).eval()
        configurable.load_state_dict(legacy.state_dict(), strict=True)
        x = torch.randn(5, 4)
        t = torch.arange(5).float().unsqueeze(1)
        self.assertTrue(torch.equal(legacy(x, t), configurable(x, t)))

    def test_softmax_and_entropy(self):
        field = ConfigurableLinCombField(
            4, K=3, mask=None, soft=True, gate_mode="softmax", entropy_lambda=0.2
        ).train()
        x = torch.randn(5, 4)
        values = field.get_gate_values(x, torch.zeros(5, 1))
        self.assertTrue(bool((values["coefficients"] >= 0).all()))
        self.assertTrue(torch.allclose(values["coefficients"].sum(1), torch.ones(5)))
        field(x, torch.zeros(5, 1))
        self.assertIsNotNone(field._cached_gate_reg)
        field.off_mask_penalty().backward()
        self.assertTrue(any(p.grad is not None for p in field.coeff_net.parameters()))

    def test_sparse_is_raw_l1(self):
        field = ConfigurableLinCombField(
            4, K=3, mask=None, soft=True, gate_mode="raw", sparse_lambda=0.5
        ).train()
        x = torch.randn(5, 4)
        expected = field.get_gate_values(x, torch.zeros(5, 1))["coefficients"].abs().mean()
        field(x, torch.zeros(5, 1))
        self.assertTrue(torch.allclose(field._cached_gate_reg, expected))

    def test_wrapper_interface(self):
        field = ConfigurableLinCombField(4, K=2, gate_mode="raw")
        model = LinCombOnlyDenoiser(field)
        output = model(torch.randn(3, 4), torch.zeros(3, 1), y=torch.zeros(3))
        self.assertEqual(tuple(output.shape), (3, 4))


if __name__ == "__main__":
    unittest.main()
