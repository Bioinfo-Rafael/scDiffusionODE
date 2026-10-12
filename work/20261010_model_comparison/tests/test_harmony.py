"""Verify correction uses PCA and preserves all original visualization files."""
import importlib
import numpy as np
import pandas as pd
import anndata as ad
import pytest


def test_expansion_finish_and_montages(tmp_path, monkeypatch):
    from PIL import Image
    finish=importlib.import_module('work.20261010_model_comparison.scripts.finish_comparison_plots')
    montage=importlib.import_module('work.20261010_model_comparison.scripts.combine_harmony_pngs')
    common=importlib.import_module('work.20261010_model_comparison.common')
    for name in common.expansion_conditions():
        folder=tmp_path/name/'figures/generative';folder.mkdir(parents=True)
        (folder/'joint_umap.h5ad').touch()
        for filename in ('joint_umap.png','joint_umap_harmony.png'):
            Image.new('RGB',(32,24),'white').save(folder/filename)
    for normal in (True,False):
        out=montage.combine(tmp_path,'model_expansion',normal)
        with Image.open(out) as image:
            assert image.size==(4*(32+20)+20,8*(24+40+20)+20)
    calls=[]
    monkeypatch.setattr(finish.subprocess,'run',lambda command,**kwargs:calls.append(command))
    finish.main(['--run-root',str(tmp_path),'--harmony-python','harmony-python'])
    assert len(calls)==5  # environment, summarize, five-column metrics, Harmony, normal montage
    assert calls[1][3]=='summarize'
    assert calls[2][1].endswith('replot_comparison.py')
    assert calls[3][0]=='harmony-python' and calls[3][-1]=='model_expansion'
    assert calls[4][-1]=='--normal'


def test_expansion_plot_only_dry_run(tmp_path):
    import subprocess
    import sys
    common=importlib.import_module('work.20261010_model_comparison.common')
    result=subprocess.check_output([sys.executable,'-S',str(common.HERE/'scripts/launch_model_expansion.py'),
                                    '--dry-run','--plots-only','--run-root',str(tmp_path/'plots'),
                                    '--harmony-python','harmony-python'],cwd=common.ROOT,text=True)
    lines=result.splitlines()
    assert '--check-only' in lines[0]
    assert sum('--stages visualize' in line for line in lines)==32
    assert 'finish_comparison_plots.py' in lines[-1]
    assert '--harmony-python harmony-python' in lines[-1]
    assert '--stages train' not in result and '--stage1-checkpoint' not in result
    assert not (tmp_path/'plots').exists()


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
