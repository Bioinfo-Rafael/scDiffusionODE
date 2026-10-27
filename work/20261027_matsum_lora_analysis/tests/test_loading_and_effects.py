"""Real checkpoint loader fixtures and explicit reference for batched cell effects."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import numpy as np
import pandas as pd
import torch
import anndata as ad

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from load_model import CONDITIONS, load_trained_model, shared_matrices, extract_coefficients
from artifacts import write_json, sha256
from analyze_cells import batched_gene_rankings

P = 'work.20261010_model_comparison'
C = importlib.import_module(P + '.common')
M = importlib.import_module(P + '.adapters.model')
D = importlib.import_module(P + '.adapters.data')
torch.set_num_threads(1)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = json.loads((HERE / 'config.json').read_text())
        self.cfg.update(run_root=str(self.root / 'runs'), data_dir=str(self.root / 'data'), matrix_batch_size=3)
        data_dir = self.root / 'data'; data_dir.mkdir()
        self.a = ad.AnnData(np.random.default_rng(3).uniform(.1, 3, (10, 6)).astype(np.float32),
            obs=pd.DataFrame({'celltype': ['Erythroid1'] * 5 + ['Erythroid2'] * 5, 'stage': ['E7.5'] * 10,
                              'sample': np.arange(10) % 2}, index=[f'cell{i}' for i in range(10)]),
            var=pd.DataFrame({'gene_name': [f'symbol{i}' for i in range(6)]}, index=[f'g{i}' for i in range(6)]))
        self.a.obsm['X_umap'] = np.arange(20).reshape(10, 2).astype(float)
        source = data_dir / 'source.h5ad'; self.a.write_h5ad(source)
        source_sha = sha256(source)
        training = ad.AnnData(self.a.X, obs=self.a.obs[['celltype', 'stage']].copy(), var=self.a.var.copy())
        training.uns['stage1_data'] = dict(preprocess=False, input_sha256=source_sha, hvg_method='seurat_log1p_copy_exact_rank')
        path = data_dir / 'training.h5ad'; training.write_h5ad(path)
        edge = data_dir / 'edges.tsv'; edge.write_text('from\tto\ng0\tg1\ng2\tg3\n')
        grn = dict(source=str(edge), source_sha256=sha256(edge), mapped_tsv=str(edge), mapped_sha256=sha256(edge), unique_edges=2)
        meta = dict(training_sha256=sha256(path), input_sha256=source_sha, expression_scale=C.SCALE,
                    gene_ids=self.a.var_names.tolist(), cell_ids=self.a.obs_names.tolist(), grn=grn)
        write_json(data_dir / 'manifest.json', dict(training=str(path), metadata=meta, grn=grn))
        self.checkpoints = {}
        sources = C.source_hashes()
        for kind, condition in CONDITIONS.items():
            config = C.config(condition)
            config.update(input=str(source), expected_cells=10, expected_erythroid=10, expected_genes=6, n_hvg=6)
            config['field_kwargs'] = dict(config['field_kwargs'], time_dim=8, hidden=12)
            field = M.build_field(config, meta['gene_ids'], D.mask_for(meta['gene_ids'], meta), meta)
            raw = {'ode_model.' + key: value.clone() for key, value in field.state_dict().items()}
            ema = copy.deepcopy(raw); ema['ode_model.b'] += .4
            ck = dict(schema='model_comparison_v1', config=config, step=30000, raw=raw, ema=ema,
                      gene_ids=meta['gene_ids'], cell_ids=meta['cell_ids'], data_sha256=meta['training_sha256'],
                      expression_scale=C.SCALE, grn=grn, provenance={'source_hashes': sources})
            run = self.root / 'runs' / condition
            write_json(run / 'effective_config.json', config)
            checkpoint = run / 'checkpoints/model030000.pt'; checkpoint.parent.mkdir()
            self.checkpoints[kind] = (checkpoint, ck)
            self.save_checkpoint(kind)

    def save_checkpoint(self, kind):
        path, ck = self.checkpoints[kind]
        torch.save(ck, path)
        write_json(path.parent / 'latest.json', dict(checkpoint=str(path), sha256=sha256(path), step=30000))

    def tearDown(self): self.temp.cleanup()

    def test_both_original_fields_load_ema_and_merge_source_by_id(self):
        for kind in CONDITIONS:
            field, a, stamp = load_trained_model(kind, self.cfg, 'cpu')
            raw, _, _ = load_trained_model(kind, dict(self.cfg, weights='raw'), 'cpu')
            self.assertEqual(field.model_type, kind)
            np.testing.assert_allclose((field.b - raw.b).detach().numpy(), .4)
            np.testing.assert_array_equal(a.obs['sample'], self.a.obs['sample'])
            np.testing.assert_array_equal(a.obsm['X_umap_source'], self.a.obsm['X_umap'])
            self.assertEqual(stamp['step'], 30000)

    def test_preflight_does_not_import_plotting_or_clustering(self):
        script = '''
import importlib.abc
import sys
class NoAnalysisImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('matplotlib', 'sklearn', 'umap', 'plots', 'analyze_coefficients', 'analyze_cells', 'analyze_basis'):
            raise RuntimeError('Preflight imported analysis dependency: ' + fullname)
sys.meta_path.insert(0, NoAnalysisImports())
import run
run.main()
'''
        output = self.root / 'preflight'
        result = subprocess.run([sys.executable, '-c', script, '--run-root', self.cfg['run_root'],
                                 '--data-dir', self.cfg['data_dir'], '--check-only', '--device', 'cpu',
                                 '--output', str(output)], cwd=HERE, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['completed_models'], ['matsum', 'lora'])

    def test_wrong_ids_nonfinal_and_unreviewed_source_rejected(self):
        path, ck = self.checkpoints['matsum']
        original = copy.deepcopy(ck)
        for change in ('ids', 'step', 'source', 'nonfinite'):
            ck = copy.deepcopy(original)
            if change == 'ids': ck['gene_ids'] = ck['gene_ids'][::-1]
            if change == 'step': ck['step'] = 29000
            if change == 'source': ck['provenance']['source_hashes']['ODE/ode_20260609_mathmlp.py'] = 'unreviewed'
            if change == 'nonfinite': ck['ema']['ode_model.b'][0] = float('nan')
            self.checkpoints['matsum'] = (path, ck); self.save_checkpoint('matsum')
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_trained_model('matsum', self.cfg, 'cpu')

    def test_batched_gene_strength_equals_explicit_reference(self):
        field, a, _ = load_trained_model('lora', self.cfg, 'cpu')
        coeff = extract_coefficients(field, a, 49, 3)
        basis, base = shared_matrices(field)
        output = self.root / 'effects'; output.mkdir()
        batched_gene_rankings(coeff, a, basis, base, field, output, self.cfg)
        w = base + np.einsum('bk,kij->bij', coeff, basis)
        absolute = np.abs(w); absolute[:, np.arange(6), np.arange(6)] = 0
        rankings = pd.read_csv(output / 'effective_gene_rankings.csv')
        all_rows = rankings[rankings.group == 'ALL'].set_index('gene_id').loc[a.var_names]
        np.testing.assert_allclose(all_rows.mean_cell_incoming, absolute.sum(axis=2).mean(axis=0))
        np.testing.assert_allclose(all_rows.mean_cell_outgoing, absolute.sum(axis=1).mean(axis=0))



if __name__ == '__main__': unittest.main()
