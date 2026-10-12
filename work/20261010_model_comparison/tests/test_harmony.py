"""Verify correction uses PCA and preserves all original visualization files."""
import importlib
import numpy as np
import pandas as pd
import anndata as ad
import pytest


def test_harmony_keeps_originals(tmp_path):
    pytest.importorskip('harmonypy')
    module=importlib.import_module('work.20261010_model_comparison.scripts.replot_harmony')
    rng=np.random.default_rng(1234)
    a=ad.AnnData(rng.normal(size=(96,16)).astype('float32'),
        obs=pd.DataFrame({'sampling_origin':['Real hematopoietic reference']*48+
                          ['Generated (unconditional)']*48},index=[f'c{i}' for i in range(96)]))
    a.obsm['X_pca']=rng.normal(size=(96,10))
    a.obsm['X_pca'][48:]+=2
    a.obsm['X_umap']=rng.normal(size=(96,2))
    source=tmp_path/'joint_umap.h5ad';a.write_h5ad(source)
    image=tmp_path/'joint_umap.png';image.write_bytes(b'original plot')
    original=source.read_bytes()
    result=module.replot(source)
    assert source.read_bytes()==original and image.read_bytes()==b'original plot'
    np.testing.assert_array_equal(result.X,a.X)
    np.testing.assert_array_equal(result.obsm['X_pca'],a.obsm['X_pca'])
    assert not np.allclose(result.obsm['X_pca_harmony'],a.obsm['X_pca'])
    assert result.uns['neighbors']['params']['use_rep']=='X_pca_harmony'
    assert np.isfinite(result.obsm['X_umap']).all()
    assert (tmp_path/'joint_umap_harmony.png').is_file()
    restored=ad.read_h5ad(tmp_path/'joint_umap_harmony.h5ad')
    assert restored.uns['harmony_visualization']['batch_key']=='sampling_origin'
