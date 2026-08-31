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
from training import (  # noqa: E402
    DETAILED_LOSS_COLUMNS_20260830,
    TrainLoop20260830,
    loss_components_20260830,
)
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
        self.assertEqual(float(model.off_mask_penalty_base("l1")), 1.0)
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
        self.assertEqual(float(zero["cell_ode_consistency_raw_20260830"]), 2.5)
        self.assertEqual(
            float(zero["cell_ode_consistency_sampler_weighted_20260830"]), 2.0
        )
        no_regularizers = loss_components_20260830(
            diffusion, torch.tensor(0.0), 0.0, consistency, weights, 0.0
        )
        self.assertTrue(torch.equal(no_regularizers["total_loss"], diffusion))

    def test_component_logging_names_are_independent(self):
        values = loss_components_20260830(
            torch.tensor(1.0), torch.tensor(2.0), 1.0,
            torch.tensor([3.0]), torch.tensor([1.0]), 0.1,
            ode_offmask_base_raw=torch.tensor(0.4),
        )
        self.assertTrue({
            "diffusion_loss",
            "ode_offmask_base_raw",
            "ode_offmask_after_internal_lambda",
            "ode_regularization_final_weighted",
            "cell_ode_consistency_raw_20260830",
            "cell_ode_consistency_sampler_weighted_20260830",
            "cell_ode_consistency_final_weighted_20260830",
            "total_loss",
        }.issubset(values))
        self.assertAlmostEqual(float(values["total_loss"]), 3.3, places=6)
        self.assertAlmostEqual(float(values["ode_offmask_base_raw"]), 0.4, places=6)
        reconstructed = (
            values["diffusion_loss"]
            + values["ode_regularization_final_weighted"]
            + values["cell_ode_consistency_final_weighted_20260830"]
        )
        self.assertTrue(torch.allclose(values["total_loss"], reconstructed))

    def test_detailed_loss_schema_contains_required_columns(self):
        required = {
            "training_step", "diffusion_loss", "ode_offmask_base_raw",
            "ode_offmask_after_internal_lambda", "ode_regularization_final_weighted",
            "cell_ode_consistency_raw_20260830",
            "cell_ode_consistency_sampler_weighted_20260830",
            "cell_ode_consistency_final_weighted_20260830", "total_loss",
            "off_mask_lambda", "ode_reg_lambda", "cell_ode_reg_lambda_20260830",
            "learning_rate",
        }
        self.assertTrue(required.issubset(DETAILED_LOSS_COLUMNS_20260830))

    def test_local_lr_annealing_reaches_zero_at_training_end(self):
        loop = object.__new__(TrainLoop20260830)
        loop.lr = 1e-4
        loop.lr_anneal_steps = 100000
        loop.step = 99999
        loop.resume_step = 0
        loop.opt = type("Opt", (), {"param_groups": [{"lr": 1e-4}]})()
        loop._anneal_lr()
        self.assertEqual(loop.opt.param_groups[0]["lr"], 0.0)

    def test_microbatch_logging_aggregates_the_whole_optimizer_step(self):
        class DeterministicDiffusion:
            num_timesteps = 1000

            @staticmethod
            def training_losses(model, micro, timesteps, model_kwargs=None):
                model(micro, timesteps, **(model_kwargs or {}))
                return {"loss": micro[:, 0]}

        class FixedSampler:
            @staticmethod
            def sample(size, device):
                return (
                    torch.zeros(size, dtype=torch.long, device=device),
                    torch.ones(size, device=device),
                )

        wrapper = CellUNetODERegularized20260830(
            DummyCell(4), ode("simple_softplus")
        )
        with tempfile.TemporaryDirectory() as directory:
            loop = TrainLoop20260830(
                model=wrapper,
                diffusion=DeterministicDiffusion(),
                data=iter(()),
                batch_size=3,
                microbatch=2,
                lr=1e-4,
                ema_rate="0.9999",
                log_interval=1000,
                save_interval=5000,
                resume_checkpoint="",
                schedule_sampler=FixedSampler(),
                weight_decay=1e-4,
                lr_anneal_steps=100000,
                model_name="model",
                save_dir=directory,
                ode_reg_lambda=1.0,
                ode_reg_norm="l1",
                save_loss_details=True,
                cell_ode_reg_lambda_20260830=0.1,
            )
            batch = torch.tensor([
                [1.0, 0.2, 0.3, 0.4],
                [3.0, 0.2, 0.3, 0.4],
                [9.0, 0.2, 0.3, 0.4],
            ])
            loop.forward_backward(batch, {})
        # Microbatch means are 2 and 9; full-step sample-weighted mean is 13/3.
        self.assertAlmostEqual(
            loop._current_loss_components_20260830["diffusion_loss"], 13.0 / 3.0
        )
        self.assertNotEqual(
            loop._current_loss_components_20260830["diffusion_loss"], 9.0
        )

    def test_twelve_configs_in_canonical_order(self):
        self.assertEqual(len(EXPERIMENT_ORDER), 12)
        observed = []
        for name in EXPERIMENT_ORDER:
            config = load_experiment_config(name)
            observed.append((config["ode_type"], config["cell_ode_reg_lambda_20260830"]))
            self.assertEqual(config["total_steps"], 100000)
            self.assertEqual(config["lr_anneal_steps"], 100000)
            self.assertEqual(config["total_steps"], config["lr_anneal_steps"])
            expected_fixed = {
                "cell_unet_hidden_num": [2000, 1000, 500, 500],
                "diffusion_steps": 1000,
                "noise_schedule": "linear",
                "schedule_sampler": "uniform",
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "batch_size": 128,
                "microbatch": -1,
                "ema_rate": "0.9999",
                "save_interval": 5000,
                "seed": 1234,
                "num_samples": 3000,
                "sample_batch_size": 50,
                "use_ddim": False,
                "clip_denoised": False,
            }
            for key, expected_value in expected_fixed.items():
                self.assertEqual(config[key], expected_value, f"{name}: {key}")
        expected = [
            (ode_type, value)
            for ode_type in ODE_TYPES
            for value in (0.1, 1.0, 10.0)
        ]
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
