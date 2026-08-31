from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


SUITE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for path in (REPO_ROOT, SUITE_ROOT, SUITE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import EXPERIMENT_ORDER, load_experiment_config  # noqa: E402
from models import build_ode_from_config  # noqa: E402
from models.cellunet_ode_regularized_20260830 import CellUNetODERegularized20260830  # noqa: E402
from training import TrainLoop20260830, loss_components_20260830  # noqa: E402
from guided_diffusion.cell_model import Cell_Unet  # noqa: E402
from guided_diffusion.train_util import TrainLoop  # noqa: E402


BASE = {
    "K": 1,
    "gate_mode": "none",
    "SoftReg": True,
    "use_mask_reg": True,
    "off_mask_lambda": 5.0,
    "positive_epsilon": 1e-6,
    "raw_delta_init": 0.1,
    "hill_n": 2.0,
}
ODE_TYPES = (
    "centered_signed_hill",
    "shifted_hill_rho",
    "hill_after_linear",
    "simple_softplus",
)


class DummyCell(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x, t, y=None):
        del t, y
        return self.linear(x)


def ode(ode_type, d=4, mask=None):
    config = {**BASE, "ode_type": ode_type}
    return build_ode_from_config(
        config,
        [f"g{i}" for i in range(d)],
        mask=torch.zeros(d, d) if mask is None else mask,
    )


class ExperimentTests(unittest.TestCase):
    def test_shared_files_are_byte_identical_to_preimplementation_baseline(self):
        expected = {
            "guided_diffusion/gaussian_diffusion.py": "eeb83640dc140f91e3519976fa5cf031076d7713d049d2a1b9debc8b0390b9ab",
            "guided_diffusion/train_util.py": "8aba336eee5240a788d5c2de46092a91d1c5f21a0e215e46106e5b28fe710c19",
            "guided_diffusion/cell_model.py": "939a153b5641eb929f9cb8a5d9ca772de1469ef876c9f42d436880c6c4b0b425",
        }
        for relative, digest in expected.items():
            actual = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, relative)

    def test_legacy_cellunet_import_and_forward(self):
        model = Cell_Unet(input_dim=4, hidden_num=[16, 12, 8, 8], dropout=0.0)
        output = model(torch.randn(3, 4), torch.tensor([0, 1, 2]))
        self.assertEqual(output.shape, (3, 4))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(issubclass(TrainLoop20260830, TrainLoop))

    def test_all_odes_are_single_and_shape_preserving(self):
        x = torch.rand(3, 4) + 0.2
        for ode_type in ODE_TYPES:
            with self.subTest(ode_type=ode_type):
                model = ode(ode_type)
                self.assertEqual(model(x, torch.tensor([1, 2, 3])).shape, x.shape)
                self.assertFalse(model.is_lincomb)
                self.assertEqual(model.num_components, 1)
                self.assertIsNone(model.gate_mode)
                self.assertFalse(hasattr(model, "coeff_net"))
                self.assertFalse(hasattr(model, "time_emb"))
                self.assertFalse(any(parameter.ndim == 3 for parameter in model.parameters()))

    def test_penalty_parameter_per_ode(self):
        expected = {
            "centered_signed_hill": "alpha",
            "shifted_hill_rho": "rho",
            "hill_after_linear": "W",
            "simple_softplus": "W",
        }
        for ode_type, name in expected.items():
            model = ode(ode_type)
            self.assertEqual(model.penalty_parameter_name, name)
            self.assertIs(model.penalty_parameter(), getattr(model, name))

    def test_off_mask_internal_weight_is_applied_exactly_once(self):
        model = ode("simple_softplus", d=2)
        with torch.no_grad():
            model.W.fill_(1.0)
        self.assertEqual(float(model.off_mask_penalty("l1")), 5.0)

    def test_matrix_entries_are_target_source(self):
        x1 = torch.tensor([[1.0, 1.0]])
        x2 = torch.tensor([[2.0, 1.0]])
        for ode_type in ODE_TYPES:
            with self.subTest(ode_type=ode_type):
                model = ode(ode_type, d=2)
                with torch.no_grad():
                    parameter = model.penalty_parameter()
                    parameter.zero_()
                    parameter[1, 0] = 1.0
                y1, y2 = model(x1), model(x2)
                self.assertNotEqual(float(y1[0, 1]), float(y2[0, 1]))

    def test_tsv_mask_is_transposed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            edge = Path(directory) / "edges.tsv"
            edge.write_text("from\tto\ng0\tg1\n", encoding="utf-8")
            config = {**BASE, "ode_type": "simple_softplus", "edge_tsv_path": str(edge)}
            model = build_ode_from_config(config, ["g0", "g1"])
        self.assertEqual(float(model.mask[1, 0]), 1.0)
        self.assertEqual(float(model.mask[0, 1]), 0.0)

    def test_wrapper_returns_exact_cell_output_and_ode_value_does_not_mix(self):
        torch.manual_seed(1)
        cell = DummyCell(4)
        field = ode("simple_softplus")
        wrapper = CellUNetODERegularized20260830(cell, field).train()
        x, t = torch.rand(3, 4), torch.arange(3)
        expected = cell(x, t)
        actual1 = wrapper(x, t)
        with torch.no_grad():
            field.W.add_(100.0)
        actual2 = wrapper(x, t)
        self.assertTrue(torch.equal(actual1, expected))
        self.assertTrue(torch.equal(actual2, expected))

    def test_consistency_gradient_reaches_cell_and_ode(self):
        wrapper = CellUNetODERegularized20260830(
            DummyCell(4), ode("simple_softplus")
        ).train()
        wrapper(torch.rand(3, 4), torch.arange(3))
        wrapper.consistency_penalty_20260830().mean().backward()
        self.assertGreater(float(wrapper.ml_model.linear.weight.grad.abs().sum()), 0)
        self.assertGreater(float(wrapper.ode_model.W.grad.abs().sum()), 0)

    def test_eval_does_not_call_ode(self):
        field = ode("centered_signed_hill")
        calls = []
        handle = field.register_forward_hook(lambda *args: calls.append(1))
        wrapper = CellUNetODERegularized20260830(DummyCell(4), field)
        x, t = torch.rand(2, 4), torch.arange(2)
        wrapper.train()(x, t)
        self.assertEqual(len(calls), 1)
        wrapper.eval()(x, t)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(wrapper._cached_cell_ode_reg_20260830)
        handle.remove()

    def test_schedule_weights_and_lambda_zero(self):
        diffusion = torch.tensor(2.0)
        soft = torch.tensor(3.0)
        consistency = torch.tensor([1.0, 4.0])
        weights = torch.tensor([2.0, 0.5])
        zero = loss_components_20260830(diffusion, soft, 1.0, consistency, weights, 0.0)
        self.assertEqual(float(zero["total_loss"]), 5.0)
        self.assertEqual(float(zero["cell_ode_consistency_20260830"]), 2.0)
        no_regularizers = loss_components_20260830(
            diffusion, torch.tensor(0.0), 0.0, consistency, weights, 0.0
        )
        self.assertTrue(torch.equal(no_regularizers["total_loss"], diffusion))

    def test_component_logging_names_are_independent(self):
        values = loss_components_20260830(
            torch.tensor(1.0), torch.tensor(2.0), 1.0,
            torch.tensor([3.0]), torch.tensor([1.0]), 0.1,
        )
        self.assertTrue({
            "diffusion_loss",
            "ode_soft_constraint",
            "cell_ode_consistency_20260830",
            "total_loss",
        }.issubset(values))
        self.assertAlmostEqual(float(values["total_loss"]), 3.3, places=6)

    def test_twelve_configs_in_canonical_order(self):
        self.assertEqual(len(EXPERIMENT_ORDER), 12)
        observed = []
        for name in EXPERIMENT_ORDER:
            config = load_experiment_config(name)
            observed.append((config["ode_type"], config["cell_ode_reg_lambda_20260830"]))
        expected = [
            (ode_type, value)
            for ode_type in ODE_TYPES
            for value in (0.1, 1.0, 10.0)
        ]
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
