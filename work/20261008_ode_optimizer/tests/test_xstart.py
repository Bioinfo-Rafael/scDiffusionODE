"""X_START target, sampler semantics, and two-optimizer resume regression."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import CONDITIONS, SUITE, config_for, diffusion_for, legacy, prediction_target
import test_suite as base_tests
from test_suite import tiny_config, toy_model, toy_data
import torch


class XStartTests(unittest.TestCase):
    def test_all_four_configs_and_boolean_validation(self):
        from guided_diffusion.gaussian_diffusion import ModelMeanType
        for name in CONDITIONS:
            c = config_for(name, ['predict_xstart=true'])
            self.assertEqual(prediction_target(c), 'x_start')
            self.assertEqual(diffusion_for(c).model_mean_type, ModelMeanType.START_X)
        self.assertEqual(prediction_target(config_for('lambda0')), 'epsilon')
        with self.assertRaises(ValueError):
            config_for('lambda0', ['predict_xstart="false"'])

    def test_loss_target_and_sampler_output_are_xstart(self):
        class ConstantModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.25))
            def forward(self, x, t, **kwargs):
                return self.value.expand_as(x)
        c = config_for('lambda0', ['predict_xstart=true'])
        diffusion = diffusion_for(c)
        model = ConstantModel()
        clean = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]])
        noise = torch.full_like(clean, -2.)
        t = torch.tensor([100, 999])
        losses = diffusion.training_losses(model, clean, t, noise=noise)
        torch.testing.assert_close(losses['loss'], (clean - .25).square().mean(1))
        noisy = diffusion.q_sample(clean, t, noise=noise)
        target, label = legacy('analysis.metrics').diffusion_training_target(diffusion, clean, noisy, t, noise)
        self.assertEqual(label, 'x_start')
        torch.testing.assert_close(target, clean)
        out = diffusion.p_mean_variance(model, noisy, t, clip_denoised=False)
        torch.testing.assert_close(out['pred_xstart'], torch.full_like(clean, .25))

    def test_lambda_zero_xstart_cell_gradient(self):
        c = tiny_config(); c['predict_xstart'] = True
        model = toy_model(c).train(); diffusion = diffusion_for(c)
        batch, _ = next(toy_data(c))
        losses = diffusion.training_losses(model, batch, torch.tensor([0, 50, 500, 999]))
        diff = losses['loss'].mean()
        soft = model.ode_model.off_mask_penalty('l1')
        total = diff + c['ode_reg_lambda']*soft + 0.*model.consistency_penalty_20260830().mean()
        params = list(model.ml_model.parameters())
        left = torch.autograd.grad(diff, params, retain_graph=True)
        right = torch.autograd.grad(total, params)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_xstart_resume_and_epsilon_resume_rejected(self):
        # Reuse the tiny CPU loop factory without running the unrelated legacy tests.
        helper = base_tests.CoreTests()
        helper.setUp()
        try:
            c = tiny_config('lambda0p1'); c['predict_xstart'] = True
            full = helper.loop(helper.root/'full', c); full.run_loop()
            part = helper.loop(helper.root/'split', c); part.run_loop(stop_after=2)
            resumed = helper.loop(helper.root/'split', c, resume=True); resumed.run_loop()
            for a,b in zip(full.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            for a,b in zip(full.ema_params[0], resumed.ema_params[0]):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            epsilon = dict(c, predict_xstart=False)
            with self.assertRaisesRegex(ValueError, 'matching dual-optimizer'):
                helper.loop(helper.root/'split', epsilon, resume=True)
        finally:
            helper.tearDown()


if __name__ == '__main__':
    unittest.main()
