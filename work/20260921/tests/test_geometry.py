"""Synthetic unit/integration checks; never train or update a model."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from geometry import LocalGeometry, project
from diagnostics import directions, exposure, mixing, mode_occupancy
import model_io


class GeometryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.points = np.column_stack([rng.normal(size=(100, 2)), np.zeros(100)])
        self.geo = LocalGeometry(self.points, np.zeros(3), k=30, dimensions=(2,))
        self.basis = self.geo.local(0)[1]

    def test_plane(self):
        t, n = project([[1, 2, 0], [0, 0, 1]], self.basis)
        np.testing.assert_allclose(t, [[1, 2, 0], [0, 0, 0]], atol=1e-12)
        np.testing.assert_allclose(n, [[0, 0, 0], [0, 0, 1]], atol=1e-12)

    def test_rotation_invariance(self):
        rotation, _ = np.linalg.qr(np.random.default_rng(19).normal(size=(3, 3)))
        geo = LocalGeometry(self.points @ rotation, np.zeros(3), 30, (2,))
        v = np.array([[1, 2, 3], [-2, .3, -1]])
        for before, after in zip(project(v, self.basis), project(v @ rotation, geo.local(0)[1])):
            np.testing.assert_allclose(np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1), atol=1e-12)

    def test_projector(self):
        v = np.random.default_rng(19).normal(size=(12, 3))
        tangent, normal = project(v, self.basis)
        np.testing.assert_allclose(v, tangent+normal, atol=1e-12)
        np.testing.assert_allclose(np.sum(tangent*normal, axis=1), 0, atol=1e-12)

    def test_manifold_distance_and_cosine(self):
        query = np.array([[0, 0, 0], [0, 0, 2.5]])
        rows = self.geo.decompose(query, {'model_drift': np.array([[0, 0, 0], [0, 0, -1]])}, 1, 0, [0, 1])
        by_id = {r['trajectory_id']: r for r in rows}
        self.assertAlmostEqual(by_id[0]['manifold_distance'], 0)
        self.assertAlmostEqual(by_id[1]['manifold_distance'], 2.5)
        self.assertAlmostEqual(by_id[1]['cos_normal'], 1)
        self.assertTrue(np.isnan(by_id[0]['cos_normal']))

    def test_scaled_manifold_with_nonzero_pca_mean(self):
        mean = np.array([3., -2., 7.])
        geo = LocalGeometry(self.points, mean, 30, (2,))
        a = .37
        scaled = geo.scaled(self.points, a)
        np.testing.assert_allclose(scaled, a*(self.points+mean)-mean)
        indices, dist = geo.anchors(scaled, a)
        np.testing.assert_array_equal(indices, np.arange(len(self.points)))
        np.testing.assert_allclose(dist, 0, atol=1e-7)
        q = scaled[:1] + [0, 0, 2]
        rows = geo.decompose(q, {'model_drift': np.array([[0, 0, -1.]])}, a, 100, [0])
        self.assertAlmostEqual(rows[0]['manifold_distance'], 2)

    def test_cache_and_invalid_dimension(self):
        self.assertIs(self.geo.local(0), self.geo.local(0))
        with self.assertRaises(ValueError):
            LocalGeometry(self.points, np.zeros(3), 3, (3,))


class DiagnosticTests(unittest.TestCase):
    def test_gaussian_oracle_start_x_score(self):
        # For clean N(0,I), E[x_clean|x_t] = sqrt(alpha_bar_t)*x_t;
        # the true VP marginal score is -x_t. Exercise the saved GENE->PCA score.
        import torch
        from types import SimpleNamespace
        from guided_diffusion.script_util import create_gaussian_diffusion
        import analyze
        diffusion = create_gaussian_diffusion(predict_xstart=True)
        class GaussianOracle(torch.nn.Module):
            def forward(self, x, t):
                a = torch.tensor(np.sqrt(diffusion.alphas_cumprod), dtype=x.dtype)
                return a[t.reshape(-1)].unsqueeze(1)*x
        args = SimpleNamespace(sample_count=4, sampling_batch=4, pca_components=3, device='cpu')
        pca = SimpleNamespace(components_=np.eye(3), mean_=np.zeros(3))
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            _, _, _, vectors = analyze.sample(GaussianOracle().eval(), diffusion,
                np.zeros((4, 3)), pca, args, Path(folder), [999, 20, 0])
            np.testing.assert_allclose(vectors['score'], -vectors['state'], atol=.002, rtol=.002)

    def test_swd_identity_permutation_and_known_shift(self):
        x = np.arange(10.).reshape(-1, 1)
        v = directions(1, 16, 3)
        self.assertAlmostEqual(exposure(x, x[::-1], v)['swd'], 0)
        self.assertAlmostEqual(exposure(x, x+2, v)['swd'], 2)
        self.assertAlmostEqual(exposure(x, x+2, v)['covariance_trace_ratio'], 1)
        with self.assertRaises(ValueError):
            exposure(x, x[:-1], v)

    def test_mode_vote_and_divergences(self):
        real = np.array([[0.], [.1], [10.], [10.1]])
        labels = {'celltype': np.array(['a', 'a', 'b', 'b'])}
        rows, summary, assignments = mode_occupancy(real, real[:2], labels, 5, k=1)
        self.assertEqual([x['label'] for x in assignments], ['a', 'a'])
        self.assertAlmostEqual(summary[0]['total_variation_distance'], .5)
        self.assertGreater(summary[0]['jensen_shannon_divergence'], 0)
        self.assertAlmostEqual(sum(x['generated_fraction'] for x in rows), 1)

    def test_mixing_separated_island_and_self_exclusion(self):
        real = np.arange(8.).reshape(-1, 1)
        result, _, fraction = mixing(real, real+100, 3, 7)
        np.testing.assert_array_equal(fraction, 0)
        self.assertEqual(result['fraction_near_zero'], 1)
        # Duplicate coordinates at matching real/generated pairs: self exclusion
        # must remove generated identity, not whichever tied point sorts first.
        _, _, fraction = mixing(real, real, 1, 7)
        np.testing.assert_array_equal(fraction, 1)


def fixture(folder):
    import anndata
    import pandas as pd
    import torch
    from guided_diffusion.cell_model import Cell_Unet
    folder = Path(folder)
    campaign = folder / 'runs' / 'synthetic'
    ckptdir = campaign / 'stage1_cellunet' / 'fixture' / 'checkpoints'
    ckptdir.mkdir(parents=True)
    (campaign / 'configs').mkdir()
    genes = ['g0', 'g1', 'g2', 'g3']
    x = np.random.default_rng(8).normal(size=(24, 4)).astype('float32')
    adata = anndata.AnnData(x, obs=pd.DataFrame({'Superclass': ['A']*12+['B']*12,
                      'celltype': ['a']*8+['b']*8+['c']*8}, index=[f'c{i}' for i in range(24)]),
                      var=pd.DataFrame({'gene_name': genes}, index=genes))
    data_path = folder / 'synthetic.h5ad'
    adata.write_h5ad(data_path)
    c = dict(suite_version='20260915_x0predict_v1', condition='stage1_cellunet', objective='stage1',
        predict_xstart=True, diffusion_steps=1000, noise_schedule='linear', timestep_respacing='',
        learn_sigma=False, use_kl=False, rescale_timesteps=False, rescale_learned_sigmas=False,
        class_cond=False, ts_layer=None, use_ddim=False, clip_denoised=False, use_fp16=False,
        cell_unet_hidden_num=[8, 4], total_steps=1, seed=42, data_dir=str(data_path))
    # Synthetic format fixture only: random weights, no training, no final-EMA scientific claim.
    torch.manual_seed(11)
    state = Cell_Unet(input_dim=4, hidden_num=c['cell_unet_hidden_num']).state_dict()
    meta = dict(effective_config=c, checkpoint_kind='ema', step=1, gene_names=genes,
        gene_order_hash=model_io.gene_hash(genes), data_sha256=model_io.file_hash(data_path),
        model_mean_type='START_X', predict_xstart=True,
        preprocessing='load_data(train_vae=True, preprocess=False, layer=None); unchanged X/gene order')
    checkpoint = ckptdir / 'ema_fixture.pt'
    torch.save(dict(state_dict=state, metadata=meta), checkpoint)
    record = dict(checkpoint=str(checkpoint), checkpoint_sha256=model_io.file_hash(checkpoint),
        cellunet_hash=model_io.state_hash(state), gene_order_hash=meta['gene_order_hash'],
        data_sha256=meta['data_sha256'], step=1, status='completed')
    model_io.write_json(campaign / 'canonical_stage1.json', record)
    model_io.write_json(campaign / 'configs/stage1_cellunet.json', c)
    return campaign, checkpoint, meta


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.campaign, self.checkpoint, self.meta = fixture(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_restore(self):
        payload, provenance = model_io.validate_campaign(self.campaign)
        model, diffusion = model_io.restore(payload, 'cpu')
        self.assertEqual(model_io.state_hash(model), provenance['model_hash_before'])
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertEqual(diffusion.model_mean_type.name, 'START_X')

    def test_tampered_checkpoint_rejected(self):
        with self.checkpoint.open('ab') as f:
            f.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            model_io.validate_campaign(self.campaign)

    def test_changed_dataset_rejected(self):
        path = Path(self.meta['effective_config']['data_dir'])
        with path.open('ab') as f:
            f.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'Dataset SHA256'):
            model_io.load_real(self.meta)

    def test_invalid_candidate_does_not_hide_valid_one(self):
        invalid = self.campaign.parent / 'broken'
        invalid.mkdir()
        (invalid / 'canonical_stage1.json').write_text('{}')
        with contextlib.redirect_stdout(io.StringIO()):
            _, report = model_io.discover(stage1_root=self.temp.name)
        self.assertEqual(Path(report['campaign']), self.campaign.resolve())

    def test_nonfinal_ema_rejected(self):
        import torch
        payload = torch.load(self.checkpoint, weights_only=True)
        payload['metadata']['step'] = 0
        torch.save(payload, self.checkpoint)
        record = model_io.read_json(self.campaign / 'canonical_stage1.json')
        record['checkpoint_sha256'] = model_io.file_hash(self.checkpoint)
        model_io.write_json(self.campaign / 'canonical_stage1.json', record)
        with self.assertRaisesRegex(ValueError, 'final Stage1 EMA'):
            model_io.validate_campaign(self.campaign)

    def test_config_and_gene_order_rejected(self):
        config = dict(self.meta['effective_config'], predict_xstart=False)
        with self.assertRaises(ValueError):
            model_io.validate_config(config)
        changed = dict(self.meta, gene_names=self.meta['gene_names'][::-1])
        with self.assertRaisesRegex(ValueError, 'gene ordering'):
            model_io.load_real(changed)

    def test_ambiguous_campaigns_fail(self):
        other = self.campaign.parent / 'second'
        shutil.copytree(self.campaign, other)
        record = model_io.read_json(other / 'canonical_stage1.json')
        record['checkpoint'] = str(other / self.checkpoint.relative_to(self.campaign))
        model_io.write_json(other / 'canonical_stage1.json', record)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, '2 valid'):
            model_io.discover(stage1_root=self.temp.name)

    def test_dry_run_no_dataset_or_inference(self):
        import analyze
        with patch.object(model_io, 'load_real', side_effect=AssertionError('dataset accessed')), \
             patch.object(model_io, 'restore', side_effect=AssertionError('inference restored')), \
             contextlib.redirect_stdout(io.StringIO()):
            analyze.main(['--campaign', str(self.campaign), '--dry-run'])

    def test_full_synthetic_pipeline_and_native_sampling_parity(self):
        import analyze
        import torch
        import pandas as pd
        torch.set_num_threads(1)
        output = Path(self.temp.name) / 'analysis'
        analyze.main(['--campaign', str(self.campaign), '--device', 'cpu', '--output', str(output),
            '--sample-count', '8', '--sampling-batch', '4', '--pca-components', '3',
            '--tangent-dimensions', '1,2', '--local-knn', '5', '--mode-knn', '3', '--mixing-knn', '4',
            '--snapshot-timesteps', '999,0', '--geometry-timesteps', '0,1,300', '--cpu-threads', '1'])
        report = model_io.read_json(output / 'checkpoint_provenance.json')
        self.assertTrue(report['unchanged_assertion_passed'])
        self.assertEqual(model_io.read_json(output / 'metadata.json')['status'], 'completed')
        self.assertEqual(len(list((output / 'figures').glob('*.png'))), 11)
        geometry = pd.read_csv(output / 'tangent_normal.csv')
        high = pd.read_csv(output / 'tangent_normal_high_noise_reference.csv')
        self.assertEqual(set(high.t), {300})
        self.assertEqual(set(high.regime), {'reference_high_noise'})
        np.testing.assert_allclose(geometry[geometry.t == 0].noise_norm, 0)
        np.testing.assert_allclose(geometry.model_drift_normal_fraction_sq+geometry.model_drift_tangent_fraction_sq, 1, atol=1e-10)
        # Match RNG ordering: analyze seeds before constructing/restoring model.
        torch.manual_seed(42)
        payload, _ = model_io.validate_campaign(self.campaign)
        model, diffusion = model_io.restore(payload, 'cpu')
        final, initial = [], []
        with torch.no_grad():
            for _ in range(2):
                x = torch.randn((4, 4))
                initial.append(x.numpy().copy())
                for t in reversed(range(1000)):
                    x = diffusion.p_sample(model, x, torch.full((4,), t), clip_denoised=False)['sample']
                final.append(x.numpy())
        np.testing.assert_array_equal(np.load(output / 'final_generated.npy'), np.concatenate(final))
        np.testing.assert_array_equal(np.load(output / 'reverse_state.npy')[0], np.concatenate(initial))
        np.testing.assert_array_equal(np.load(output / 'pred_xstart.npy')[-1], np.load(output / 'final_generated.npy'))


if __name__ == '__main__':
    unittest.main()
