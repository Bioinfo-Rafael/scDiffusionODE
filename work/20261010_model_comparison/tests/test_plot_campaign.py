import importlib
import sys
from PIL import Image

P='work.20261010_model_comparison.scripts.'


def test_d_plot_only_orchestration(tmp_path,monkeypatch):
    m=importlib.import_module(P+'finish_ode_sgd_plots')
    for cfg in (m.HERE/'configs/ode_sgd').glob('*.json'):
        d=tmp_path/cfg.stem/'figures/generative';d.mkdir(parents=True)
        (d/'joint_umap.png').touch();(d/'joint_umap.h5ad').touch()
    calls=[]
    monkeypatch.setattr(m.subprocess,'run',lambda command,**kwargs:calls.append(command))
    monkeypatch.setattr(sys,'argv',['finish','--run-root',str(tmp_path)])
    m.main()
    assert len(calls)==5
    assert 'summarize' in calls[1]
    assert calls[2][1].endswith('replot_comparison.py')
    assert calls[3][-2:]==['--condition-set','ode_sgd']
    assert calls[4][-1]=='--normal'
    assert not any('train' in cmd or 'sample' in cmd or 'visualize' in cmd for cmd in calls)


def test_d_montages_preserve_both_png_sets(tmp_path):
    m=importlib.import_module(P+'combine_harmony_pngs')
    suite=m.Path(m.__file__).resolve().parents[1]
    for i,cfg in enumerate(sorted((suite/'configs/ode_sgd').glob('*.json'))):
        folder=tmp_path/cfg.stem/'figures/generative';folder.mkdir(parents=True)
        Image.new('RGB',(80,60),(i,20,30)).save(folder/'joint_umap.png')
        Image.new('RGB',(80,60),(i,40,50)).save(folder/'joint_umap_harmony.png')
    for normal in (False,True):
        out=m.combine(tmp_path,'ode_sgd',normal)
        with Image.open(out) as image:
            for i in range(4):
                assert image.getpixel((20+i*100,60))==((i,20,30) if normal else (i,40,50))
