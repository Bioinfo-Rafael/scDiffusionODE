#!/usr/bin/env python3
"""Small-dimensional integration tests for the unchanged TrainLoop contract."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from guided_diffusion import logger  # noqa: E402
from guided_diffusion.fp16_util import MixedPrecisionTrainer  # noqa: E402
from guided_diffusion.nn import update_ema  # noqa: E402
from guided_diffusion.train_util import TrainLoop  # noqa: E402
from models.factory import build_experiment_components  # noqa: E402

SCRIPTS_ROOT = SUITE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
import common as script_common  # noqa: E402


class TinyDenoiser(nn.Module):
    """Minimal epsilon predictor with the Cell_Unet call signature."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        initial_weight = torch.tensor(
            [
                [0.17, -0.05, 0.03],
                [0.02, 0.13, -0.07],
                [-0.04, 0.06, 0.11],
            ],
            dtype=torch.float32,
        )[:dim, :dim]
        self.weight = nn.Parameter(initial_weight.clone())
        self.bias = nn.Parameter(torch.linspace(-0.04, 0.05, dim))
        self.time_gain = nn.Parameter(torch.linspace(0.03, 0.07, dim))

    def forward(self, noisy, timesteps, **model_kwargs):
        del model_kwargs
        if timesteps.shape != (noisy.shape[0], 1):
            raise ValueError("TinyDenoiser expects [batch,1] timesteps")
        time_feature = torch.sin(timesteps.to(noisy.dtype) / 7.0)
        return torch.tanh(
            noisy @ self.weight.T
            + self.bias
            + time_feature * self.time_gain
        )


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260903)
        self.genes = ("g0", "g1", "g2")
        self.device = torch.device("cpu")

    def _config(self, forward_model: str, loss_mode: str) -> dict:
        return {
            "input_dim": len(self.genes),
            "forward_model": forward_model,
            "forward_dtype": "float64",
            "loss_mode": loss_mode,
            "normalize_elbo_by_dimension": True,
            "surrogate_terminal_kl_weight": 0.7,
            "noise_schedule": "cosine",
            "diffusion_steps": 16,
            "microbatch": -1,
            "schedule_sampler": "batch_shared_physical_uniform",
            "use_fp16": False,
            "timestep_respacing": "",
            "weight_decay": 0.0,
            "covariance_jitter": 0.0,
            "aux_dim": 2,
            "d_diagonal_floor": 0.0,
            "initial_d_diagonal": 0.5,
            "use_grn_mask": False,
            "grn_penalty_weight": 0.0,
        }

    def _components(self, forward_model: str, loss_mode: str):
        components = build_experiment_components(
            self._config(forward_model, loss_mode),
            self.genes,
            self.device,
            denoiser=TinyDenoiser(len(self.genes)),
        )
        process = components.model.forward_process
        with torch.no_grad():
            if forward_model == "stationary_qd":
                process.raw_q_k.copy_(torch.tensor([[0., .06], [-.035, 0.]], dtype=torch.float64))
                process.b.copy_(torch.tensor([[.24, .04], [-.03, .21]], dtype=torch.float64))
            else:
                process.raw_w.copy_(
                    torch.tensor(
                        [
                            [0.04, 0.035, -0.015],
                            [-0.025, -0.02, 0.045],
                            [0.01, -0.03, 0.025],
                        ],
                        dtype=torch.float64,
                    )
                )
                process.raw_b.copy_(
                    torch.tensor([0.05, -0.035, 0.02], dtype=torch.float64)
                )
        return components

    def _float32_components(self, forward_model: str):
        config = self._config(forward_model, "paper_elbo")
        config["forward_dtype"] = "float32"
        return build_experiment_components(
            config,
            self.genes,
            self.device,
            denoiser=TinyDenoiser(len(self.genes)),
        )

    @staticmethod
    def _batch():
        x = torch.tensor(
            [
                [0.38, -0.21, 0.47],
                [-0.32, 0.26, 0.14],
                [0.09, 0.41, -0.28],
                [0.23, -0.17, -0.36],
            ],
            dtype=torch.float32,
        )
        path_noise = torch.tensor(
            [
                [0.27, -0.31, 0.08],
                [-0.12, 0.19, 0.36],
                [0.05, 0.24, -0.29],
                [0.33, 0.07, -0.16],
            ],
            dtype=torch.float64,
        )
        boundary_noise = torch.tensor(
            [
                [-0.11, 0.22, 0.14],
                [0.29, -0.08, 0.04],
                [-0.17, 0.31, -0.06],
                [0.13, -0.25, 0.21],
            ],
            dtype=torch.float64,
        )
        return x, path_noise, boundary_noise

    def _losses(self, components):
        x, path_noise, boundary_noise = self._batch()
        # One fractional legacy index is shared by the whole local batch.
        timesteps = torch.full((x.shape[0],), 6.35, dtype=torch.float32)
        return components.diffusion.training_losses(
            components.model,
            x,
            timesteps,
            noise=path_noise,
            boundary_noise=boundary_noise,
        )

    def test_paper_and_surrogate_losses_have_batch_shape_and_all_gradients(self):
        expected_keys = {
            "loss",
            "path_loss",
            "terminal_kl",
            "boundary_nll",
            "forward_regularization",
            "plain_epsilon_mse",
            "total_loss",
            "path_loss_raw",
            "terminal_kl_raw",
            "boundary_nll_raw",
            "path_after_duration",
            "path_final_per_dim",
            "terminal_final_per_dim",
            "boundary_final_per_dim",
            "paper_elbo_per_dim",
            "grn_penalty_raw",
            "grn_penalty_weight",
            "grn_penalty_final_weighted",
            "sampled_physical_time",
            "fractional_diffusion_timestep",
            "dimension",
        }
        for forward_model in ("stationary_qd", "free_affine"):
            for loss_mode in ("paper_elbo", "epsilon_surrogate"):
                with self.subTest(
                    forward_model=forward_model, loss_mode=loss_mode
                ):
                    components = self._components(forward_model, loss_mode)
                    losses = self._losses(components)
                    self.assertEqual(set(losses), expected_keys)
                    for name, value in losses.items():
                        self.assertEqual(value.shape, (4,), name)
                        self.assertTrue(torch.isfinite(value).all(), name)

                    if loss_mode == "paper_elbo":
                        self.assertGreater(float(losses["boundary_nll"].abs().sum()), 0.0)
                    else:
                        torch.testing.assert_close(
                            losses["boundary_nll"],
                            torch.zeros_like(losses["boundary_nll"]),
                            atol=0.0,
                            rtol=0.0,
                        )

                    losses["loss"].mean().backward()
                    named_parameters = dict(components.model.named_parameters())
                    expected_forward_names = (
                        {
                            "forward_process.raw_embedding",
                            "forward_process.raw_q_k",
                            "forward_process.b",
                            "forward_process.raw_isotropic_d",
                        }
                        if forward_model == "stationary_qd"
                        else {
                            "forward_process.raw_w",
                            "forward_process.raw_b",
                        }
                    )
                    self.assertTrue(expected_forward_names <= set(named_parameters))
                    self.assertTrue(
                        {"denoiser.weight", "denoiser.bias", "denoiser.time_gain"}
                        <= set(named_parameters)
                    )
                    for name, parameter in named_parameters.items():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertGreater(
                            float(parameter.grad.abs().sum()),
                            0.0,
                            f"zero gradient for {name} in {forward_model}/{loss_mode}",
                        )

    def test_named_parameters_feed_optimizer_ema_and_raw_state_dict_paths(self):
        for forward_model in ("stationary_qd", "free_affine"):
            with self.subTest(forward_model=forward_model):
                components = self._components(forward_model, "paper_elbo")
                model = components.model
                named_parameters = list(model.named_parameters())
                trainer = MixedPrecisionTrainer(model=model, use_fp16=False)

                self.assertEqual(len(trainer.master_params), len(named_parameters))
                for master, (_, parameter) in zip(
                    trainer.master_params, named_parameters
                ):
                    self.assertIs(master, parameter)

                losses = self._losses(components)
                trainer.backward(losses["loss"].mean())
                optimizer = torch.optim.AdamW(
                    trainer.master_params, lr=1e-3, weight_decay=0.0
                )
                before_step = {
                    name: parameter.detach().clone()
                    for name, parameter in named_parameters
                }
                optimizer.step()
                for name, parameter in named_parameters:
                    self.assertFalse(
                        torch.equal(before_step[name], parameter.detach()),
                        f"optimizer did not update {name}",
                    )

                # TrainLoop creates one EMA tensor per master parameter.  The
                # positional contract therefore includes both denoiser and
                # forward-process parameters.
                ema_parameters = [
                    torch.zeros_like(parameter) for parameter in trainer.master_params
                ]
                update_ema(ema_parameters, trainer.master_params, rate=0.25)
                for ema, source in zip(ema_parameters, trainer.master_params):
                    torch.testing.assert_close(ema, 0.75 * source.detach())

                raw_state = copy.deepcopy(
                    trainer.master_params_to_state_dict(trainer.master_params)
                )
                parameter_names = {name for name, _ in named_parameters}
                self.assertTrue(parameter_names <= set(raw_state))
                self.assertTrue(
                    any(name.startswith("denoiser.") for name in parameter_names)
                )
                self.assertTrue(
                    any(
                        name.startswith("forward_process.")
                        for name in parameter_names
                    )
                )

                restored = self._components(forward_model, "paper_elbo")
                restored.model.load_state_dict(raw_state, strict=True)
                for (name, expected), (restored_name, actual) in zip(
                    model.named_parameters(), restored.model.named_parameters()
                ):
                    self.assertEqual(name, restored_name)
                    torch.testing.assert_close(actual, expected)

                restored_master = trainer.state_dict_to_master_params(raw_state)
                self.assertEqual(len(restored_master), len(named_parameters))
                for tensor, (name, expected) in zip(
                    restored_master, named_parameters
                ):
                    torch.testing.assert_close(tensor, expected, msg=name)

                optimizer_state = copy.deepcopy(optimizer.state_dict())
                restored_optimizer = torch.optim.AdamW(
                    restored.model.parameters(), lr=1e-3, weight_decay=0.0
                )
                restored_optimizer.load_state_dict(optimizer_state)
                self.assertEqual(
                    len(restored_optimizer.state), len(named_parameters)
                )

    def test_actual_core_trainloop_checkpoint_and_resume_include_forward_parameters(self):
        """Exercise raw/EMA/optimizer save and resume through the core loop."""

        batch = self._batch()[0]
        condition = {"y": torch.zeros(batch.shape[0], dtype=torch.long)}
        for forward_model in ("stationary_qd", "free_affine"):
            with self.subTest(forward_model=forward_model), tempfile.TemporaryDirectory() as tmp:
                temporary_root = Path(tmp)
                run_root = temporary_root / "runs"
                run_path = run_root / forward_model / "smoke"
                segment = run_path / "segments" / "segment_000"
                segment.mkdir(parents=True)
                script_common.write_json(
                    run_path / "requested_config.json",
                    {"config": {"ema_rate": "0.9,0.99"}, "provenance": {}},
                )

                components = self._float32_components(forward_model)
                logger.configure(dir=str(segment / "logs"), format_strs=[])
                loop = TrainLoop(
                    model=components.model,
                    diffusion=components.diffusion,
                    data=iter(()),
                    batch_size=batch.shape[0],
                    microbatch=-1,
                    lr=1e-4,
                    ema_rate="0.9,0.99",
                    log_interval=100,
                    save_interval=100,
                    resume_checkpoint="",
                    use_fp16=False,
                    schedule_sampler=components.schedule_sampler,
                    weight_decay=0.0,
                    lr_anneal_steps=2,
                    model_name="model",
                    save_dir=str(segment),
                    ode_reg_lambda=0.0,
                    save_loss_details=False,
                )
                loop.run_step(batch, condition)
                loop.step = 1
                loop.save()

                raw = segment / "model" / "model000001.pt"
                optimizer_peer = raw.parent / "opt000001.pt"
                ema_peers = (
                    raw.parent / "ema_0.9_000001.pt",
                    raw.parent / "ema_0.99_000001.pt",
                )
                self.assertTrue(raw.is_file())
                self.assertTrue(optimizer_peer.is_file())
                self.assertTrue(all(path.is_file() for path in ema_peers))
                saved = torch.load(raw, map_location="cpu", weights_only=False)
                self.assertTrue(
                    any(name.startswith("forward_process.") for name in saved)
                )

                old_runs_root = script_common.RUNS_ROOT
                script_common.RUNS_ROOT = run_root
                try:
                    self.assertEqual(
                        script_common.latest_raw_checkpoint(run_path), raw.resolve()
                    )
                    # A bundle is incomplete if even one configured EMA rate is
                    # missing; accepting merely any EMA file would be unsafe.
                    ema_peers[1].unlink()
                    self.assertIsNone(
                        script_common.latest_raw_checkpoint(run_path)
                    )
                    torch.save(
                        loop.mp_trainer.master_params_to_state_dict(
                            loop.ema_params[1]
                        ),
                        ema_peers[1],
                    )
                    self.assertEqual(
                        script_common.latest_raw_checkpoint(run_path), raw.resolve()
                    )
                finally:
                    script_common.RUNS_ROOT = old_runs_root

                restored = self._float32_components(forward_model)
                restored_loop = TrainLoop(
                    model=restored.model,
                    diffusion=restored.diffusion,
                    data=iter(()),
                    batch_size=batch.shape[0],
                    microbatch=-1,
                    lr=1e-4,
                    ema_rate="0.9,0.99",
                    log_interval=100,
                    save_interval=100,
                    resume_checkpoint=str(raw),
                    use_fp16=False,
                    schedule_sampler=restored.schedule_sampler,
                    weight_decay=0.0,
                    lr_anneal_steps=2,
                    model_name="model",
                    save_dir=str(segment),
                    ode_reg_lambda=0.0,
                    save_loss_details=False,
                )
                self.assertEqual(restored_loop.resume_step, 1)
                self.assertTrue(restored_loop.opt.state)
                self.assertEqual(len(restored_loop.ema_params), 2)
                for expected_rate, restored_rate in zip(
                    loop.ema_params, restored_loop.ema_params
                ):
                    for expected, actual in zip(expected_rate, restored_rate):
                        torch.testing.assert_close(actual, expected)
                for key, value in components.model.state_dict().items():
                    torch.testing.assert_close(
                        restored.model.state_dict()[key], value, msg=key
                    )


if __name__ == "__main__":
    unittest.main()
