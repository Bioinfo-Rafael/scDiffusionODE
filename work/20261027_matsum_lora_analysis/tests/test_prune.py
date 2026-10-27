"""Cleanup must preserve requested outputs and reject unknown array schemas."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prune_outputs import plan_run, apply_plan, MOMENT_KEYS

class CleanupTests(unittest.TestCase):
    def fixture(self, root):
        (root/'manifest.json').write_text(json.dumps({'status':'complete'}))
        basis=root/'matsum/basis'; basis.mkdir(parents=True)
        np.savez_compressed(basis/'shared_matrices.npz', B=np.ones((8,2,2)))
        for name in ['basis_heatmaps_full.png','basis_heatmaps_zoom.png','my_manual_plot.png']:
            (basis/name).write_bytes(b'fixture')
        group=root/'matsum/t049/cell_matrices/groups/all';group.mkdir(parents=True)
        path=group/'matrix_moments.npz'
        np.savez_compressed(path, **{k:np.arange(8.) for k in MOMENT_KEYS}, mean_W=np.ones((2,2)))
        return basis,path

    def test_dry_run_apply_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);basis,path=self.fixture(root)
            _,plan=plan_run(root)
            self.assertTrue((basis/'basis_heatmaps_zoom.png').exists())
            self.assertEqual(len(plan['delete']),1)
            apply_plan(root,plan)
            self.assertFalse((basis/'basis_heatmaps_zoom.png').exists())
            self.assertTrue((basis/'basis_heatmaps_full.png').exists())
            self.assertTrue((basis/'my_manual_plot.png').exists())
            with np.load(path) as z:
                self.assertEqual(set(z.files),MOMENT_KEYS)
                for k in z.files: np.testing.assert_array_equal(z[k],np.arange(8.))
            _,plan=plan_run(root);self.assertEqual(plan['delete'],[])
            apply_plan(root,plan)

    def test_unknown_schema_and_running_run_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);basis,path=self.fixture(root)
            np.savez_compressed(path,**{k:np.arange(8.) for k in MOMENT_KEYS}, user_array=np.ones(2))
            _,plan=plan_run(root)
            with self.assertRaises(ValueError): apply_plan(root,plan)
            self.assertTrue((basis/'basis_heatmaps_zoom.png').exists())
            (root/'manifest.json').write_text(json.dumps({'status':'running'}))
            with self.assertRaises(ValueError): plan_run(root)

if __name__=='__main__': unittest.main()
