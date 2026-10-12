"""Regression for retry_20261012 checkpoints created before model expansion."""
import hashlib
import importlib
from pathlib import Path
import subprocess
import tempfile
import types
import unittest

import torch

PACKAGE = 'work.20261010_model_comparison'
C = importlib.import_module(PACKAGE+'.common')
K = importlib.import_module(PACKAGE+'.adapters.checkpoint')
M = importlib.import_module(PACKAGE+'.adapters.model')
O = importlib.import_module(PACKAGE+'.adapters.objectives')
V = importlib.import_module(PACKAGE+'.adapters.source_compatibility')
VERIFIER = 'work/20261010_model_comparison/adapters/source_compatibility.py'


def historical_envelope():
    # Reconstruct from git bytes, independently of the allowlist's saved values.
    current = C.source_hashes()
    saved = dict(current)
    rules = C.read_json(C.HERE/'audit/d_adamw_source_compatibility.json')['files']
    for filename in rules:
        result = subprocess.run(['git', 'show', '4d7187f:'+filename], cwd=C.ROOT, capture_output=True)
        if result.returncode == 0:
            saved[filename] = hashlib.sha256(result.stdout).hexdigest()
        else:
            saved.pop(filename, None)
    return saved, current


class AdamWCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.saved, cls.current = historical_envelope()

    def test_original_adamw_envelope_all_four_conditions(self):
        for path in sorted((C.HERE/'configs/ode_sgd').glob('D*.json')):
            config = C.config(path.stem)  # original pilot AdamW config
            changed = V.verify_sources(self.saved, config, self.current)
            self.assertEqual(len(changed), 48)
        self.assertEqual(self.saved['work/20261010_model_comparison/adapters/training.py'],
                         'e2651650c8003a9a06e9b91e93ab4a6f32096728a9b866a94f5ac1c08cc1be34')
        self.assertEqual(self.saved['work/20261010_model_comparison/common.py'],
                         '17da4957449008a6089e3ea8e2eec2e41dcd95f44f9ad9d20113ddc40d0b5fbc')

    def test_unknown_numeric_changes_and_wrong_scopes_rejected(self):
        config = C.config('D03_consistency_lambda0p1')
        for filename in ('guided_diffusion/cell_model.py',
                         'work/20260830/models/ode_fields_20260830.py',
                         'work/20261010_model_comparison/adapters/objectives.py',
                         'work/20261010_model_comparison/configs/base.json'):
            with self.subTest(filename=filename):
                tampered = dict(self.current, **{filename: 'unreviewed'})
                with self.assertRaisesRegex(ValueError, 'unreviewed'):
                    V.verify_sources(self.saved, config, tampered)
                tampered_saved = dict(self.saved, **{filename: 'unreviewed'})
                with self.assertRaisesRegex(ValueError, 'unreviewed'):
                    V.verify_sources(tampered_saved, config, self.current)
        with self.assertRaisesRegex(ValueError, 'source/config'):
            V.verify_sources(self.saved, C.config('A01_cellunet_startx'), self.current)
        with self.assertRaisesRegex(ValueError, 'unreviewed'):
            V.verify_sources(self.saved, C.config(C.HERE/'configs/ode_sgd/D03_consistency_lambda0p1.json'), self.current)

    def test_previous_verifier_checkpoints_for_all_supported_scopes(self):
        old_bytes = subprocess.check_output(['git', 'show', '8e84b30:'+VERIFIER], cwd=C.ROOT)
        saved = dict(self.current, **{VERIFIER: hashlib.sha256(old_bytes).hexdigest()})
        for config in (C.config('D03_consistency_lambda0p1'),
                       C.config(C.HERE/'configs/ode_sgd/D03_consistency_lambda0p1.json'),
                       C.config('B06_lowrank')):
            self.assertEqual(V.verify_sources(saved, config, self.current), [VERIFIER])

    def test_actual_checkpoint_load_preserves_weights_and_checks_metadata(self):
        c = C.config('D03_consistency_lambda0p1')
        c.update(cell_unet_hidden_num=[16, 12, 8, 8], n_hvg=3, expected_cells=3)
        (C.HERE/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=C.HERE/'runs', prefix='test_adamw_compat_') as folder:
            run = Path(folder)
            edges = run/'edges.tsv'
            edges.write_text('from\tto\ng0\tg1\ng1\tg2\n')
            meta = dict(gene_ids=['g0', 'g1', 'g2'], cell_ids=['c0', 'c1', 'c2'],
                        training_sha256='synthetic-training-data',
                        grn=dict(unique_edges=2, mapped_tsv=str(edges), mapped_sha256=C.sha256(edges)))
            mask = importlib.import_module(PACKAGE+'.adapters.data').mask_for(meta['gene_ids'], meta)
            model = M.build(c, meta['gene_ids'], mask=mask).eval()
            checkpoint = run/'model030000.pt'
            bundle = dict(schema=K.SCHEMA, config=c, gene_ids=meta['gene_ids'], cell_ids=meta['cell_ids'],
                          data_sha256=meta['training_sha256'], expression_scale=C.SCALE, grn=meta['grn'],
                          provenance=dict(source_hashes=self.saved), raw=model.state_dict(),
                          ema=model.state_dict(), step=30000)
            torch.save(bundle, checkpoint)
            before = C.sha256(checkpoint)
            x = torch.tensor([[1., 2., 4.], [3., 2., 1.]])
            t = torch.tensor([[49], [999]])
            for weights in ('raw', 'ema'):
                restored, _, _ = K.load(c, run, meta, value=str(checkpoint), weights=weights)
                for name, expected in M.branches(c, model, x, t).items():
                    torch.testing.assert_close(M.branches(c, restored, x, t)[name], expected, rtol=0, atol=0)
            self.assertEqual(C.sha256(checkpoint), before)
            audit = C.read_json(run/'source_compatibility.json')
            self.assertEqual(audit['training_source_hashes'], self.saved)
            self.assertEqual(audit['current_source_hashes'], self.current)
            with self.assertRaisesRegex(ValueError, 'gene IDs'):
                K.load(c, run, dict(meta, gene_ids=list(reversed(meta['gene_ids']))), value=str(checkpoint))

    def test_old_and_current_model_loss_and_gradients_match(self):
        c = C.config('D03_consistency_lambda0p1')
        c['cell_unet_hidden_num'] = [16, 12, 8, 8]
        old_modules = {}
        for name in ('model', 'objectives'):
            source = subprocess.check_output(['git', 'show', f'4d7187f:work/20261010_model_comparison/adapters/{name}.py'], cwd=C.ROOT)
            module = types.ModuleType(PACKAGE+'.adapters.reviewed_old_'+name)
            module.__package__ = PACKAGE+'.adapters'
            exec(compile(source, f'reviewed_4d7187f_{name}.py', 'exec'), module.__dict__)
            old_modules[name] = module
        torch.manual_seed(123)
        old = old_modules['model'].build(c, ['g0', 'g1', 'g2'], mask=torch.ones(3, 3)).train()
        new = M.build(c, ['g0', 'g1', 'g2'], mask=torch.ones(3, 3), state=old.state_dict()).train()
        x = torch.rand(4, 3)
        noise = torch.randn(4, 3, dtype=torch.float64)
        t = torch.tensor([0, 49, 499, 999])
        weights = torch.ones(4)
        old_loss, old_parts = old_modules['objectives'].loss(c, old, C.diffusion(c), x, t, weights, noise=noise)
        new_loss, new_parts = O.loss(c, new, C.diffusion(c), x, t, weights, noise=noise)
        torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
        self.assertEqual(old_parts, new_parts)
        old_loss.backward(); new_loss.backward()
        for left, right in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
