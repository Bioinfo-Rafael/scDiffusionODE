#!/usr/bin/env python3
"""Exact auxiliary Model A against independently assembled dense references."""
from __future__ import annotations

from contextlib import ExitStack
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from diffusion.stationary_qd import StationaryQDForward
from diffusion.objectives import boundary_gaussian_nll, standard_normal_kl
from diffusion.time_mapping import PhysicalTimeMap
from models.wrapper import LearnableForwardModel
from sampling.reverse_sde import sample_reverse_sde


class TinyDenoiser(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gain = nn.Parameter(torch.linspace(.1, .2, dim, dtype=torch.float64))

    def forward(self, values, timesteps):
        return torch.tanh(self.gain * values + .001 * timesteps)


class StationaryQDForwardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260905)

    def model(self, d=7, k=3, *, random=True, **kwargs):
        model = StationaryQDForward(d, aux_dim=k, **kwargs)
        if random:
            with torch.no_grad():
                model.raw_q_k.normal_(std=.12)
                model.b.normal_(std=.2)
        return model

    def reference(self, model, x, time):
        # Full matrices are test-only. Full exp is independent of the reduced exp.
        z = torch.linalg.qr(model.raw_embedding, mode='reduced')[0]
        qk = model.raw_q_k - model.raw_q_k.T
        bb = model.b @ model.b.T
        eye = torch.eye(model.dim, dtype=x.dtype)
        q = z @ qk @ z.T
        d = model.isotropic_d() * eye + z @ bb @ z.T
        phi = torch.matrix_exp(-time * (q + d))
        covariance = eye - phi @ phi.T
        covariance = (covariance + covariance.T) / 2
        # A noise coordinate is tied to a root. The specified root generally
        # differs from dense chol, so compare using the specified coordinate.
        lk = torch.linalg.cholesky(z.T @ covariance @ z)
        vp = -torch.expm1(-2 * model.isotropic_d() * time)
        root = vp.sqrt() * (eye - z @ z.T) + z @ lk @ z.T
        return dict(z=z, q=q, d=d, phi=phi, covariance=covariance, root=root,
                    mean=x @ phi.T)

    def assert_value_gradients(self, actual, expected, model, *, atol=3e-9, rtol=3e-8):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        parameters = tuple(model.parameters())
        probe = torch.linspace(.3, 1., actual.numel(), dtype=actual.dtype).reshape(actual.shape)
        ga = torch.autograd.grad((actual * probe).sum(), parameters, retain_graph=True)
        ge = torch.autograd.grad((expected * probe).sum(), parameters, retain_graph=True)
        for (name, _), a, e in zip(model.named_parameters(), ga, ge):
            self.assertTrue(torch.isfinite(a).all(), name)
            torch.testing.assert_close(a, e, atol=atol, rtol=rtol, msg=name)

    def test_parameter_constraints_lyapunov_and_inner_products(self):
        for d, k in ((3, 1), (5, 2), (8, 3), (4, 3)):
            with self.subTest(d=d, k=k):
                m = self.model(d, k)
                z, q, diffusion = m.basis(), m.q_matrix(), m.d_matrix()
                torch.testing.assert_close(z.T @ z, torch.eye(k, dtype=z.dtype))
                torch.testing.assert_close(q.T, -q, atol=1e-14, rtol=0)
                self.assertGreater(float(torch.linalg.eigvalsh(diffusion).min()), 0)
                torch.testing.assert_close(m.stationarity_residual(), torch.zeros_like(q), atol=2e-14, rtol=0)
                h = z @ m.b
                torch.testing.assert_close(diffusion[0, 1], h[0] @ h[1])
                torch.testing.assert_close(q[0, 1], z[0] @ m.q_auxiliary() @ z[1])
                self.assertEqual(sum(p.numel() for p in m.parameters()), d*k + 2*k*k + 1)

    def test_standard_vp_limit_independent_of_embedding_and_physical_clock(self):
        for d, k in ((3, 1), (7, 2), (5, 4)):
            m = self.model(d, k, random=False)
            x, noise = torch.randn(4, d, dtype=torch.float64), torch.randn(4, d, dtype=torch.float64)
            for _ in range(2):
                with torch.no_grad():
                    m.raw_embedding.normal_()
                for alpha in (.9999, .37, .001):
                    s = -math.log(alpha)
                    sample, _, stats = m.q_sample(x, s, noise, return_stats=True)
                    dense = stats.materialize_for_analysis()
                    torch.testing.assert_close(m.q_matrix(), torch.zeros(d, d, dtype=x.dtype))
                    torch.testing.assert_close(m.d_matrix(), .5 * torch.eye(d, dtype=x.dtype))
                    torch.testing.assert_close(stats.mean, math.sqrt(alpha) * x)
                    torch.testing.assert_close(dense['covariance'], (1-alpha)*torch.eye(d, dtype=x.dtype))
                    torch.testing.assert_close(sample, math.sqrt(alpha)*x + math.sqrt(1-alpha)*noise)

    def test_dense_reference_values_and_all_parameter_gradients(self):
        for d, k in ((3, 1), (5, 2), (7, 3), (4, 3)):
            with self.subTest(d=d, k=k):
                m = self.model(d, k)
                x = torch.randn(4, d, dtype=torch.float64)
                epsilon = torch.randn_like(x)
                prediction = torch.randn_like(x)
                s = .43
                stats = m.transition_stats(x, s)
                ref = self.reference(m, x, s)
                dense = stats.materialize_for_analysis()
                self.assert_value_gradients(stats.mean, ref['mean'], m)
                self.assert_value_gradients(dense['covariance'], ref['covariance'], m)
                sample, _ = m.sample_from_stats(stats, epsilon)
                sample_ref = ref['mean'] + epsilon @ ref['root'].T
                self.assert_value_gradients(sample, sample_ref, m)
                score = m.conditional_score(stats, prediction)
                score_ref = -torch.linalg.solve(ref['root'].T, prediction.T).T
                self.assert_value_gradients(score, score_ref, m)
                true_score = m.conditional_score(stats, epsilon)
                torch.testing.assert_close(true_score, -torch.linalg.solve(ref['covariance'], (sample - stats.mean).T).T)
                residual = prediction - epsilon
                u = torch.linalg.solve(ref['root'].T, residual.T).T
                dsm_ref = (u @ ref['d'] * u).sum(-1)
                self.assert_value_gradients(m.noise_metric_quadratic(stats, residual), dsm_ref, m)
                kl_ref = standard_normal_kl(ref['mean'], ref['covariance'], torch.linalg.cholesky(ref['covariance']))
                self.assert_value_gradients(m.terminal_kl(stats), kl_ref, m)
                # A nonlinear predictor retains all reparameterization gradients.
                score = m.conditional_score(stats, torch.tanh(sample * .17 + .2))
                score_ref = -torch.linalg.solve(ref['root'].T, torch.tanh(sample_ref * .17 + .2).T).T
                nll = m.boundary_nll(stats, x, sample, score)
                nll_ref = boundary_gaussian_nll(x_start=x, y_boundary=sample_ref, model_score=score_ref,
                    transition_matrix=ref['phi'], affine_shift=None, covariance=ref['covariance'],
                    cholesky=torch.linalg.cholesky(ref['covariance']))
                self.assert_value_gradients(nll, nll_ref, m)
                decoder_mean = torch.linalg.solve(ref['phi'], (sample_ref + score_ref @ ref['covariance'].T).T).T
                self.assert_value_gradients(m.boundary_mean(stats, sample, score), decoder_mean, m)
                root_decoder = torch.linalg.solve(ref['phi'], ref['root'])
                self.assert_value_gradients(m.boundary_noise(stats, epsilon), epsilon @ root_decoder.T, m)

    def test_sampling_covariance_analytically_and_empirically(self):
        m = self.model(6, 2)
        stats = m.transition_stats(torch.zeros(1, 6, dtype=torch.float64), .62)
        dense = self.reference(m, stats.mean, .62)
        actual_root = stats.noise_transform(torch.eye(6, dtype=torch.float64)).T
        torch.testing.assert_close(actual_root @ actual_root.T, dense['covariance'])
        with torch.no_grad():
            samples = stats.noise_transform(torch.randn(80000, 6, dtype=torch.float64))
        torch.testing.assert_close(torch.cov(samples.T), dense['covariance'], atol=.012, rtol=0)
        diffusion_root = m.diffusion_noise(torch.eye(6, dtype=torch.float64)).T
        torch.testing.assert_close(diffusion_root @ diffusion_root.T, 2 * dense['d'])

    def test_grn_orientation_effective_interaction_only_no_embedding_mask(self):
        mask = torch.zeros(5, 5, dtype=torch.float64)
        mask[0, 1] = 1
        m = self.model(5, 2, grn_mask_target_source=mask, grn_penalty_weight=3.)
        interaction = m.interaction_matrix()
        mask = mask + torch.eye(5, dtype=torch.float64)
        torch.testing.assert_close(m.grn_mask_target_source, mask)
        torch.testing.assert_close(m.grn_penalty_base(), ((1-mask)*interaction).abs().mean())
        torch.testing.assert_close(m.additional_regularization(), 3*((1-mask)*interaction).abs().mean())
        self.assertEqual(torch.count_nonzero(m.basis()).item(), 10)
        self.assertGreater(float(interaction[1, 0].abs()), 0)
        # A mask update cannot alter Z, transitions or declared diffusion.
        before = m.transition_stats(torch.ones(2, 5, dtype=torch.float64), .4).mean
        z = m.basis().detach().clone()
        m.grn_mask_target_source.fill_(1)
        torch.testing.assert_close(z, m.basis(), atol=0, rtol=0)
        torch.testing.assert_close(before, m.transition_stats(torch.ones(2, 5, dtype=torch.float64), .4).mean, atol=0, rtol=0)
        self.assertEqual(float(m.grn_penalty_base()), 0.)

    def test_full_production_elbo_backward_and_reverse_never_use_dense_linalg(self):
        d, k = 17, 3
        m = self.model(d, k)
        model = LearnableForwardModel(TinyDenoiser(d), m)
        x = torch.randn(4, d, dtype=torch.float64)
        def checked(original):
            def call(matrix, *args, **kwargs):
                self.assertEqual(tuple(matrix.shape[-2:]), (k, k))
                return original(matrix, *args, **kwargs)
            return call
        with ExitStack() as stack:
            for module, name in ((torch, 'matrix_exp'), (torch.linalg, 'cholesky_ex'),
                                 (torch.linalg, 'cholesky'), (torch.linalg, 'solve'),
                                 (torch.linalg, 'solve_triangular'), (torch.linalg, 'slogdet')):
                stack.enter_context(patch.object(module, name, checked(getattr(module, name))))
            for name in ('q_matrix', 'd_matrix', 'diffusion_covariance', 'stationary_operator', 'interaction_matrix'):
                stack.enter_context(patch.object(m, name, side_effect=AssertionError('dense materialization: '+name)))
            terms = model(x, torch.ones(4), physical_time=.4, boundary_time=.0001, terminal_time=2.)
            loss = (terms['weighted_mismatch'] + terms['boundary_nll'] + terms['terminal_kl']).mean()
            loss.backward()
            for name, param in model.named_parameters():
                self.assertTrue(torch.isfinite(param.grad).all(), name)
            tm = PhysicalTimeMap(torch.tensor([.04, .2, .4], dtype=torch.float64))
            a = sample_reverse_sde(model, tm, batch_size=4, seed=31)
            b = sample_reverse_sde(model, tm, batch_size=4, seed=31)
            torch.testing.assert_close(a.samples, b.samples, atol=0, rtol=0)

    def test_no_gene_space_square_tensor_in_elbo_forward_or_backward(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        class NoGeneSquare(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                def check(value):
                    if isinstance(value, torch.Tensor):
                        if value.ndim >= 2 and value.shape[-2:] == (19, 19):
                            raise AssertionError(f"gene-space matrix in {func}")
                    elif isinstance(value, (tuple, list)):
                        for item in value:
                            check(item)
                check(result)
                return result

        model = LearnableForwardModel(TinyDenoiser(19), self.model(19, 3))
        x = torch.randn(4, 19, dtype=torch.float64)
        with NoGeneSquare():
            terms = model(x, torch.ones(4), physical_time=.5,
                          boundary_time=.001, terminal_time=2.)
            (terms['weighted_mismatch'] + terms['boundary_nll'] +
             terms['terminal_kl']).mean().backward()

    def test_raw_and_ema_checkpoint_metadata_survives_serialization(self):
        import io
        from guided_diffusion.fp16_util import MixedPrecisionTrainer
        model = LearnableForwardModel(TinyDenoiser(7), self.model())
        trainer = MixedPrecisionTrainer(model=model, use_fp16=False)
        state = trainer.master_params_to_state_dict(trainer.master_params)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        metadata = restored._metadata['forward_process']
        self.assertEqual(metadata['model_schema_version'], 2)
        self.assertEqual(metadata['model_a_parameterization'], 'auxiliary_shared_subspace')
        self.assertEqual(metadata['aux_dim'], 3)
        self.assertFalse(metadata['full_d_cholesky'])
        self.assertFalse(metadata['full_d_matrix_exponential'])
        self.assertFalse(metadata['approximation'])
        model.load_state_dict(restored)

    def test_explicit_floor_fixed_sigma_and_initialization_gradient_caveat(self):
        m = self.model(random=False, isotropic_d_floor=.01)
        with torch.no_grad():
            m.raw_isotropic_d.fill_(-1000)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(m.d_matrix()).min()), .01 - 1e-14)
        stats = m.transition_stats(torch.ones(2, 7, dtype=torch.float64), .0001)
        self.assertGreater(float(stats.variance_perp), 0.)
        loss = m.noise_metric_quadratic(stats, torch.ones_like(stats.mean)).mean()
        loss.backward()
        torch.testing.assert_close(m.b.grad, torch.zeros_like(m.b))
        fixed = self.model(5, 2, learn_isotropic_d=False)
        self.assertNotIn('raw_isotropic_d', dict(fixed.named_parameters()))
        with self.assertRaisesRegex(ValueError, 'jitter=0'):
            self.model(covariance_jitter=1e-5)

    def test_checkpoint_schema_rejects_old_dense_and_wrong_k_even_nonstrict(self):
        m = self.model()
        copy = self.model()
        copy.load_state_dict(m.state_dict())
        for name, value in m.state_dict().items():
            torch.testing.assert_close(value, copy.state_dict()[name], atol=0, rtol=0)
        for state in ({'raw_q_upper': torch.zeros(21), 'raw_d_lower': torch.zeros(28)},
                      self.model(7, 2).state_dict()):
            with self.assertRaisesRegex(RuntimeError, 'Incompatible Model A checkpoint'):
                m.load_state_dict(state, strict=False)
        wrapper = LearnableForwardModel(TinyDenoiser(7), m)
        with self.assertRaisesRegex(RuntimeError, 'Incompatible Model A checkpoint'):
            wrapper.load_state_dict({'forward_process.raw_q_upper': torch.zeros(21)}, strict=False)
        self.assertEqual(m.provenance()['aux_dim'], 3)

    def test_float32_boundary_series_and_fail_fast_diagnostics(self):
        m = self.model(1024, 8, dtype=torch.float32)
        stats = m.transition_stats(torch.zeros(2, 1024), .000001)
        self.assertEqual(stats.covariance_evaluation, 'adaptive_integral_series')
        self.assertEqual(stats.cholesky_k.shape, (8, 8))
        self.assertGreater(float(stats.variance_perp), 0.)
        invalid = -torch.eye(8)
        with self.assertRaisesRegex(RuntimeError, r'min_eig=.*condition_number=.*sigma²=.*physical_time='):
            m._factor_covariance(invalid, m.isotropic_d(), torch.tensor(.01))
        for time in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                m.transition_stats(torch.zeros(2, 1024), time)
        with self.assertRaisesRegex(ValueError, 'batch-shared'):
            m.transition_stats(torch.zeros(2, 1024), torch.tensor([.1, .2]))
        with self.assertRaisesRegex(ValueError, 'forward dtype'):
            self.model(7, 2, dtype=torch.float32, isotropic_d_floor=1e-50)
        with self.assertRaisesRegex(ValueError, 'aux_dim'):
            self.model(3, 4)

    def test_integral_series_matches_identity_values_and_gradients(self):
        m = self.model(6, 3)
        s = torch.tensor(.17, dtype=torch.float64)
        a = m.q_auxiliary() + m.d_auxiliary()
        phi = torch.matrix_exp(-s * a)
        identity = torch.eye(3, dtype=torch.float64) - phi @ phi.T
        series, terms = m._integral_series_covariance(-a, 2*m.d_auxiliary(), s)
        self.assertGreater(terms, 1)
        torch.testing.assert_close(series, identity, atol=2e-13, rtol=2e-13)
        parameters = (m.raw_q_k, m.b, m.raw_isotropic_d)
        ga = torch.autograd.grad(series.square().sum(), parameters, retain_graph=True)
        ge = torch.autograd.grad(identity.square().sum(), parameters)
        for actual, expected in zip(ga, ge):
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


if __name__ == '__main__':
    unittest.main()
