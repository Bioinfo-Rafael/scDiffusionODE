"""Numerical and plotting regressions; run with unittest (no real data needed)."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

PREFIX = 'work.20261010_model_comparison'
B = importlib.import_module(PREFIX+'.analysis.branches')
L = importlib.import_module(PREFIX+'.analysis.losses')
P = importlib.import_module(PREFIX+'.analysis.plotting')
C = importlib.import_module(PREFIX+'.common')
S = importlib.import_module(PREFIX+'.scripts.diagnose_branches')


class Field(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x, t):
        return x*self.factor


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ml_model = Field(2.)
        self.ode_model = Field(-1.)


class ToyDiffusion:
    from guided_diffusion.gaussian_diffusion import ModelMeanType
    model_mean_type = ModelMeanType.START_X
    num_timesteps = 1000

    def q_sample(self, x, t, noise):
        return x+noise*t[:, None]/1000

    def _scale_timesteps(self, t):
        return t


def loss_row(step, rec=4., reg=.5, consistency=0.):
    return dict(step=step, diffusion_loss=rec, ode_offmask_base_raw=reg/5,
                cell_ode_consistency_raw_20260830=20.,
                ode_regularization_final_weighted=reg,
                cell_ode_consistency_final_weighted_20260830=consistency,
                total_loss=rec+reg+consistency)


def write_logs(run, rows):
    folder = Path(run)/'logs'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'loss_000000_test.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))


class DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = torch.tensor([[1., 2., 4.], [2., 5., 9.], [4., 1., 3.]])
        self.meta = dict(campaign='synthetic', condition='D03', checkpoint_step=30000, weights='ema')

    def evaluate(self, model=None, batch=2):
        return B.evaluate(model or ToyModel(), ToyDiffusion(), self.x, [0, 49, 999],
                          batch_size=batch, seed=12, device='cpu', metadata=self.meta)

    def test_three_pairs_target_norms_and_noise_batch_invariance(self):
        pairs, norms = self.evaluate()
        self.assertEqual(set(pairs.pair), set(B.PAIRS))
        zero = pairs[pairs.diffusion_timestep == 0].set_index('pair')
        self.assertAlmostEqual(zero.loc['cell_target', 'norm_ratio_mean'], 2.)
        self.assertAlmostEqual(zero.loc['ode_cell', 'norm_ratio_mean'], .5)
        self.assertAlmostEqual(zero.loc['ode_target', 'pearson_mean'], -1.)
        self.assertAlmostEqual(zero.loc['cell_target', 'mse_mean'], float(self.x.double().square().mean()))
        self.assertAlmostEqual(zero.loc['ode_target', 'mse_mean'], float(4*self.x.double().square().mean()))
        target = norms[norms.branch == 'target']['mean'].to_numpy()
        np.testing.assert_allclose(target, self.x.double().norm(dim=1).mean())
        other_pairs, other_norms = self.evaluate(batch=3)
        pd.testing.assert_frame_equal(pairs, other_pairs, check_exact=False, rtol=1e-10, atol=1e-12)
        pd.testing.assert_frame_equal(norms, other_norms, check_exact=False)

    def test_undefined_metrics_and_mode_restore(self):
        result = B.pair_values(torch.ones(2, 3), torch.zeros(2, 3))
        for key in ('cosine', 'pearson', 'norm_ratio'):
            self.assertTrue(torch.isnan(result[key]).all())
        np.testing.assert_allclose(result['mse'], 1.)
        self.assertEqual(B.stats(result['pearson'])['undefined_count'], 2)
        model = ToyModel().train()
        model.ode_model.eval()
        self.evaluate(model)
        self.assertTrue(model.training)
        self.assertTrue(model.ml_model.training)
        self.assertFalse(model.ode_model.training)
        wrong = ToyDiffusion()
        wrong.model_mean_type = wrong.ModelMeanType.EPSILON
        with self.assertRaisesRegex(ValueError, 'START_X'):
            B.evaluate(model, wrong, self.x, [0], batch_size=2, seed=1, device='cpu', metadata={})

    def test_real_diffusion_agrees_with_august_metrics(self):
        # Check real q_sample/timestep handling against the imported historical evaluator.
        c = C.config('D03_consistency_lambda0p1')
        c['cell_unet_hidden_num'] = [16, 12, 8, 8]
        M = importlib.import_module(PREFIX+'.adapters.model')
        torch.manual_seed(7)
        model = M.build(c, ['a', 'b', 'c'], mask=torch.ones(3, 3))
        diffusion = C.diffusion(c)
        pairs, _ = B.evaluate(model, diffusion, self.x, [49], batch_size=3, seed=7, device='cpu', metadata=self.meta)
        old_target, old_ode = B.legacy.evaluate_diffusion_timesteps(model, diffusion, self.x, [49],
            batch_size=3, seed=7, device='cpu', metadata={})
        indexed = pairs.set_index('pair')
        for metric in ('cosine_mean', 'pearson_mean', 'mse_mean', 'pearson_global'):
            self.assertAlmostEqual(indexed.loc['cell_target', metric], old_target['cell_target_'+metric].iloc[0], places=9)
            self.assertAlmostEqual(indexed.loc['ode_cell', metric], old_ode['cell_ode_'+metric].iloc[0], places=9)

    def test_loss_weights_fractions_and_sparse_optimizer_window(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [loss_row(1, rec=2.), loss_row(2, rec=6.), loss_row(100, rec=10.)]
            write_logs(folder, rows)
            (Path(folder)/'logs/loss_000000_test.rolled_back.jsonl').write_text(json.dumps(loss_row(200)))
            history, mean, std, fraction = L.load(folder, window=3)
            np.testing.assert_allclose(mean.diffusion_loss, [2., 4., 10.])
            np.testing.assert_allclose(std.diffusion_loss, [0., 2., 0.])
            np.testing.assert_allclose(fraction.consistency, 0.)
            np.testing.assert_allclose(fraction[['reconstruction', 'grn', 'consistency']].sum(axis=1), 1.)
            self.assertAlmostEqual(fraction.reconstruction_rolling.iloc[1], 4/4.5)
            self.assertNotAlmostEqual(fraction.reconstruction_rolling.iloc[1], fraction.reconstruction.iloc[:2].mean())
            self.assertEqual(history.step.tolist(), [1, 2, 100])
            bad = dict(rows[0], total_loss=999.)
            write_logs(folder, [bad])
            with self.assertRaisesRegex(ValueError, 'total_loss'):
                L.load(folder)

    def test_conflicting_live_logs_and_zero_total(self):
        with tempfile.TemporaryDirectory() as folder:
            write_logs(folder, [loss_row(1), loss_row(1, rec=999.)])
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                L.load(folder)
            write_logs(folder, [loss_row(1, rec=0., reg=0.)])
            _, _, _, fraction = L.load(folder)
            self.assertTrue(fraction[['reconstruction', 'grn', 'consistency']].isna().all().all())

    def test_plots_and_loss_only_cli(self):
        # End-to-end real PNG rendering, including lambda=0 and log axes.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            condition = 'D03_consistency_lambda0p1'
            run = root/'campaign'/condition
            rows = [loss_row(i, rec=4/i) for i in range(1, 6)]
            write_logs(run, rows)
            (run/'effective_config.json').write_text(json.dumps(C.config(condition)))
            output = root/'out'
            with patch.object(C, 'HERE', root):
                S.main(['--run-roots', str(run.parent), '--conditions', condition,
                        '--output-dir', str(output), '--loss-only', '--rolling-window', '3'])
            self.assertEqual(json.loads((output/'manifest.json').read_text())['status'], 'complete')
            figures = output/'campaign'/condition/'figures'
            self.assertEqual(len(list(figures.glob('loss*.png'))), 10)
            pairs, norms = self.evaluate()
            P.plot_branches(pairs, norms, figures, 'Synthetic diagnostic fixture')
            P.plot_comparison(pairs, norms, output/'comparison')
            self.assertTrue((figures/'branches_mse_mean_log.png').stat().st_size > 1000)
            self.assertEqual(len(list((output/'comparison').glob('comparison_ema_*.png'))), 6)
            self.assertEqual(len(list((output/'comparison').glob('comparison_loss*.png'))), 6)


if __name__ == '__main__':
    unittest.main()
