"""Synthetic integration/negative tests; no real data or long training required."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from prepare_data import prepare, select_hvg
from train import train, Batches
from sample import sample
from export_velocity import export
from visualize import visualize
from evaluate import evaluate


def fixture(directory):
    c = config(); c.update(runs=str(directory / 'run'), input=str(directory / 'input.h5ad'),
        expected_cells=80, expected_genes=64, expected_erythroid=60, n_hvg=32,
        inference_batch_size=16, pca_components=10, neighbor_pcs=10, neighbors=10)
    c['training'].update(cell_unet_hidden_num=[16, 12, 8, 8], batch_size=8,
                         sample_batch_size=4, save_interval=2, log_interval=2)
    rng = np.random.default_rng(12)
    x = rng.gamma(2, 2, (80, 64)).astype('float32')
    x *= 1e4 / x.sum(axis=1, keepdims=True)
    obs = pd.DataFrame({'celltype': [ERYTHROID[i//12] if i < 60 else 'Other' for i in range(80)],
                        'stage': ['E7.0' if i < 40 else 'E8.0' for i in range(80)]},
                       index=[f'cell_{i}' for i in range(80)])
    var = pd.DataFrame({'gene_name': [f'symbol_{i}' for i in range(64)]},
                       index=[f'ENSMUSG{i:06d}' for i in range(64)])
    a = ad.AnnData(X=sparse.csr_matrix(x), obs=obs, var=var)
    a.layers['spliced'] = a.X.copy(); a.layers['unspliced'] = a.X.copy()
    a.uns['mouse_gastrulation_preprocessing'] = dict(target_sum=1e4, log1p=False,
                                                   normalization='scanpy.pp.normalize_total')
    a.write_h5ad(c['input'])
    return c, a


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory(dir=paths(config()) / 'logs', prefix='synthetic_')
        cls.c, cls.original = fixture(Path(cls.tmp.name))
        cls.original_hash = sha256(cls.c['input'])
        prepare(cls.c)
        cls.ck = train(cls.c, steps=2)

    @classmethod
    def tearDownClass(cls):
        # Keep artifacts for visual inspection; unittest temporary cleanup disabled.
        cls.tmp._finalizer.detach()
        print('SYNTHETIC_ARTIFACTS=' + cls.tmp.name)

    def test_01_hvg_preserves_linear_values_and_ids(self):
        a, meta = read_data(self.c)
        order = a.var['source_gene_index'].to_numpy()
        np.testing.assert_array_equal(a.X.toarray(), self.original.X[:, order].toarray())
        self.assertEqual(ids(a.obs_names), ids(self.original.obs_names))
        self.assertEqual(sha256(self.c['input']), self.original_hash)
        self.assertEqual(a.n_vars, 32)
        self.assertNotEqual(ids(a.var_names), a.var.gene_name.tolist())
        self.assertEqual(prepare(self.c), paths(self.c) / 'data/training.h5ad')
        # No entire training sparse matrix densification in minibatch loader.
        a.X.toarray = lambda: self.fail('full matrix densified')
        self.assertEqual(Batches(a, 8, 1234, preprocess=False).next().shape, (8, 32))
        with self.assertRaises(ValueError):
            Batches(a, 8, 1234, preprocess=True)

    def test_02_source_architecture_loss(self):
        c = config()
        self.assertEqual(c['training']['cell_unet_hidden_num'], [2000, 1000, 500, 500])
        self.assertEqual(c['training']['total_steps'], 30000)
        self.assertEqual(c['n_hvg'], 1024)
        self.assertTrue(c['training']['predict_xstart'])
        model, _, _, _ = load_checkpoint(self.c, self.ck)
        self.assertFalse(hasattr(model, 'ode_model'))
        d = diffusion(self.c)
        x = torch.randn(8, 32); t = torch.arange(8); noise = torch.randn_like(x, dtype=torch.float64)
        output = model(d.q_sample(x, t, noise), t[:, None])
        loss, _ = OBJECTIVES.training_loss(model, d, x, t, torch.ones(8), self.c['training'], noise=noise)
        torch.testing.assert_close(loss, ((output-x)**2).mean())

    def test_03_resume_equals_uninterrupted(self):
        resumed = train(self.c, steps=12, resume=self.ck)
        other = copy.deepcopy(self.c); other['runs'] = str(Path(self.tmp.name) / 'uninterrupted')
        prepare(other); continuous = train(other, steps=12)
        a = torch.load(resumed, weights_only=False); b = torch.load(continuous, weights_only=False)
        for key in ('raw', 'ema'):
            for name in a[key]:
                torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
        self.assertEqual(a['optimizer']['param_groups'], b['optimizer']['param_groups'])
        self.assertEqual(a['step'], 12)

    def test_04_sampling_and_fields(self):
        path = sample(self.c, checkpoint=self.ck, count=4)
        g = ad.read_h5ad(path)
        self.assertEqual(g.shape, (4, 32)); self.assertNotIn('celltype', g.obs)
        self.assertTrue(np.isfinite(g.X).all())
        self.assertEqual(sample(self.c, checkpoint=self.ck, count=4), path)
        path = export(self.c, checkpoint=self.ck)
        f = ad.read_h5ad(path); a, _ = read_data(self.c)
        self.assertNotIn('velocity', f.layers); self.assertNotIn('benchmark_velocity', f.uns)
        np.testing.assert_allclose(f.layers['reconstruction_displacement'],
                                   f.layers['cellunet_x_start'] - a.X.toarray())
        self.assertEqual(ids(f.obs_names), ids(a.obs_names))
        self.assertEqual(export(self.c, checkpoint=self.ck), path)

    def test_05_id_and_checkpoint_rejection(self):
        ck = torch.load(self.ck, weights_only=False)
        bad = paths(self.c) / 'checkpoints/bad.pt'
        ck['gene_ids'] = ck['gene_ids'][::-1]; torch.save(ck, bad)
        with self.assertRaisesRegex(ValueError, 'gene IDs/order'):
            load_checkpoint(self.c, bad)
        ck = torch.load(self.ck, weights_only=False)
        ck['cell_ids'][0] = 'fabricated'; torch.save(ck, bad)
        with self.assertRaisesRegex(ValueError, 'cell IDs/order'):
            load_checkpoint(self.c, bad)
        with self.assertRaises(ValueError):
            ids(['a', 'a'])

    def test_06_benchmark_never_launches_for_cellunet(self):
        with patch('subprocess.run', side_effect=AssertionError('benchmark launched')):
            self.assertEqual(evaluate(self.c), 2)
        report = json.loads((paths(self.c) / 'metrics/benchmark_status.json').read_text())
        self.assertEqual(report['status'], 'not_applicable')
        self.assertTrue(all(v is None for v in report['metrics'].values()))
        self.assertFalse((paths(self.c) / 'metrics/metrics.csv').exists())

    def test_07_visualization_scopes(self):
        sample(self.c, checkpoint=self.ck, count=4); export(self.c, checkpoint=self.ck)
        for scope in ('samples', 'all', 'erythroid'):
            path = visualize(self.c, scope)
            self.assertTrue((path / 'completed.json').exists())
            self.assertEqual(visualize(self.c, scope), path)
        full = ad.read_h5ad(paths(self.c) / 'figures/all/embedding.h5ad')
        ery = ad.read_h5ad(paths(self.c) / 'figures/erythroid/embedding.h5ad')
        self.assertEqual(full.n_obs, 80); self.assertEqual(ery.n_obs, 60)
        self.assertFalse(np.array_equal(full.obsm['X_umap'][:60], ery.obsm['X_umap']))

    def test_08_bad_preprocessing_rejected(self):
        c, a = copy.deepcopy(self.c), self.original.copy()
        c['runs'] = str(Path(self.tmp.name) / 'bad_run')
        c['input'] = str(Path(self.tmp.name) / 'bad_input.h5ad')
        a.uns['mouse_gastrulation_preprocessing']['log1p'] = True
        a.write_h5ad(c['input'])
        with self.assertRaisesRegex(ValueError, 'provenance'):
            prepare(c)
        a.uns['mouse_gastrulation_preprocessing']['log1p'] = False
        a.X[0, 0] += 1
        a.write_h5ad(c['input'])
        with self.assertRaisesRegex(ValueError, 'X != spliced'):
            prepare(c)

if __name__ == '__main__':
    unittest.main(verbosity=2)
