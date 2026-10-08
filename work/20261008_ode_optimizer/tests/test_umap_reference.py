"""Lightweight reference-selection regression tests; no training or sampling."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import SUITE, write_json, sha256
from plot_umap import select_real_reference, render_all, SELECTION_VERSION
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


class ReferenceSelectionTests(unittest.TestCase):
    def data(self, column='Superclass', sparse_x=False):
        x = np.arange(4005 * 4, dtype=np.float32).reshape(4005, 4)
        obs = pd.DataFrame({column: ['Erythropoietic']*4001+['Immune', 'Other', None, 'Immune'],
                            'celltype': ['ery']*4001+['other']*4},
                           index=[f'cell-{i}' for i in range(4005)])
        var = pd.DataFrame({'gene_name': ['g3', 'g1', 'g2', 'g0']})
        return ad.AnnData(sparse.csr_matrix(x) if sparse_x else x, obs=obs, var=var)

    def test_all_erythropoietic_no_3000_cell_cap(self):
        a = self.data()
        x, genes, indices, names, info = select_real_reference(a)
        self.assertEqual(len(x), 4001)
        np.testing.assert_array_equal(x, a.X[:4001])
        np.testing.assert_array_equal(indices, np.arange(4001))
        self.assertEqual(names.tolist(), a.obs_names[:4001].tolist())
        self.assertEqual(genes, ['g3', 'g1', 'g2', 'g0'])
        self.assertFalse(info['subsampling'])
        self.assertEqual(info['celltype_counts'], {'ery': 4001})

    def test_lowercase_column_and_sparse_x(self):
        a = self.data(column='superclass', sparse_x=True)
        x, _, _, _, info = select_real_reference(a)
        np.testing.assert_array_equal(x, a.X[:4001].toarray())
        self.assertEqual(info['selected_superclass_column'], 'superclass')

    def test_missing_annotation_or_population_fails(self):
        a = self.data()
        del a.obs['Superclass']
        with self.assertRaises(KeyError): select_real_reference(a)
        a = self.data()
        a.obs['Superclass'] = 'Immune'
        with self.assertRaises(ValueError): select_real_reference(a)

    def test_redraw_all_panels_and_comparison_from_saved_coordinates(self):
        import tempfile
        from common import CONDITIONS, STEPS
        from plot_comparison import main as comparison
        with tempfile.TemporaryDirectory(dir=SUITE/'tests') as directory:
            campaign = Path(directory)
            folder = campaign/'umap'
            folder.mkdir()
            rng = np.random.default_rng(10)
            arrays = {f'{name}__{step}':rng.normal(size=(7,2)) for name in CONDITIONS for step in STEPS}
            np.savez_compressed(folder/'coordinates.npz',real=rng.normal(size=(4001,2)),
                                limits=np.array([-5,5,-5,5]),**arrays)
            write_json(folder/'embedding.json',dict(fit_count=1,
                settings={'selection_version':SELECTION_VERSION},
                coordinate_sha256=sha256(folder/'coordinates.npz')))
            render_all(folder)
            comparison(['--campaign',str(campaign)])
            self.assertEqual(len(list(folder.glob('*.png'))),25)
            from PIL import Image
            with Image.open(folder/'umap_comparison_600_1000_all_conditions.png') as im:
                self.assertGreater(im.height,im.width)


if __name__ == '__main__':
    unittest.main()
