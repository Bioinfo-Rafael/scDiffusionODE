"""Actual UMAP/scVelo/Dynamo/SDE/plot integration and negative boundary tests."""
import copy
import json
import numpy as np
import pytest
from test_adapter import A, fixture, make
from noeqthermo import pipeline as P
from noeqthermo import dynamo_bridge as B


def test_configs():
    for mode in ['smoke','full']:
        c=A.load_analysis_config(mode)
        assert c['sde']['D']==c['sde']['dt']==.2
    c=A.load_analysis_config('full')
    assert [c['sde'][k] for k in ['trajectories','steps','burn_in','grid_size']]==[400,1000000,100000,100]
    assert c['seed']==1234 and c['sde']['resume']


def test_selection_and_id_order(fixture):
    _,_,a,_,_=fixture
    c=A.load_analysis_config('smoke');prior=P._load_prior_symbols()
    shared,genes,count=P._shared_erythropoietic_adata({},c,prior,a)
    assert count==60 and genes==a.var_names.tolist() # deliberately different gene_name
    assert shared.uns['noeq_selection']['fallback']
    assert shared.uns['noeq_selection']['available_labels']['celltype']['Erythroid1']==30
    a.obs['Superclass']=['Erythropoietic']*30+['other']*30
    shared,_,count=P._shared_erythropoietic_adata({},c,prior,a)
    assert count==30 and not shared.uns['noeq_selection']['fallback']
    c['selection'].update(mode='filter',obs_key='celltype',values=['missing'])
    with pytest.raises(ValueError,match='no cells'):P._shared_erythropoietic_adata({},c,prior,a)


def test_lap_skips_undefined_states_and_pairs_both_directions(fixture,monkeypatch):
    _,_,a,_,_=fixture
    c=A.load_analysis_config('smoke')['lap']
    a.obsm['X_umap']=np.random.default_rng(2).normal(size=(len(a),2))
    a.obs['celltype']=['Erythroid1']*len(a)
    assert B.run_least_action_paths(a,celltype_key='celltype',config=c,diffusion=.2)[1]['status']=='skipped'
    a.obs['celltype']=['Erythroid1']*30+['Erythroid2']*30
    calls=[]
    class PD:
        def least_action(self,adata,start,end,**kwargs):
            calls.append((start[0],end[0],kwargs['D']))
            adata.uns[kwargs['add_key']]=dict(prediction=[np.zeros((2,2))],action=[np.zeros(2)],t=[np.arange(2)],mftp=[1.])
    class Dyn:pd=PD()
    monkeypatch.setattr(B,'import_dynamo',lambda _:Dyn())
    c.update(endpoint_labels=['Erythroid1','Erythroid2'])
    _,status=B.run_least_action_paths(a,celltype_key='celltype',config=c,diffusion=.2)
    assert status['completed_pairs']==2 and calls[0][:2]==calls[1][:2][::-1]
    assert calls[0][2]==calls[1][2]==.2


def test_end_to_end_and_resume(fixture,tmp_path):
    adapter=make(fixture)
    config=A.load_analysis_config('smoke')
    config['lap'].update(endpoint_labels=['Erythroid1','Erythroid2'],n_points=6)
    out=tmp_path/'output'
    result=A.run(adapter,config,out)
    assert result['status']=='completed'
    directory=out/'models'/A.CONDITION
    assert len(list(directory.glob('*.png')))==10
    with np.load(directory/'observed_gene_velocity.npz') as data:
        assert data['velocity'].shape==(60,8)
        assert data['genes'].tolist()==adapter.meta['gene_ids']
    with np.load(directory/'observed_umap_velocity.npz') as data:
        assert data['velocity_umap'].shape==(60,2) and np.isfinite(data['velocity_umap']).all()
    validation=json.loads((out/'numerical_validation.json').read_text())
    assert validation['probability_sum']==pytest.approx(1)
    A.run(adapter,config,out)
    manifest=json.loads((directory/'model_manifest.json').read_text())
    assert manifest['simulation']['resumed']
    changed=copy.deepcopy(config);changed['seed']+=1
    with pytest.raises(ValueError,match='different inputs'):A.run(adapter,changed,out)


def test_interrupted_sde_resume_is_identical(tmp_path):
    from noeqthermo.landscape import simulate_stationary_distribution
    config=A.load_analysis_config('smoke')['sde']
    config.update(trajectories=6,steps=80,burn_in=10,grid_size=8,checkpoint_every=20)
    bounds=np.array([[-2.,2.],[-2.,2.]])
    def field(x):return -.4*x
    reference=simulate_stationary_distribution(field,bounds,config,config,seed=1234,checkpoint_dir=tmp_path/'reference')
    calls=0
    def interrupted(x):
        nonlocal calls
        calls+=1
        if calls==44:raise InterruptedError('synthetic interruption after checkpoint')
        return field(x)
    with pytest.raises(InterruptedError):
        simulate_stationary_distribution(interrupted,bounds,config,config,seed=1234,checkpoint_dir=tmp_path/'resumed')
    resumed=simulate_stationary_distribution(field,bounds,config,config,seed=1234,checkpoint_dir=tmp_path/'resumed')
    assert resumed[3]['start_step']==40
    for expected,actual in zip(reference[:3],resumed[:3]):np.testing.assert_array_equal(expected,actual)


def test_sampled_topography_matches_dynamo_without_refinement(monkeypatch):
    """The bounded path preserves Dynamo sampling/bounds/confidence semantics."""
    import importlib
    import anndata as ad
    topo = importlib.import_module('dynamo.vectorfield.topography')
    utils = importlib.import_module('dynamo.vectorfield.utils')
    rng = np.random.default_rng(44)
    a = ad.AnnData(np.zeros((80, 1)))
    a.obsm['X_umap'] = rng.normal(size=(80, 2))
    a.uns['VecFld_umap'] = {}
    b = a.copy()
    def field(adata, basis):
        return adata.uns['VecFld_umap'], lambda x: -np.asarray(x)
    monkeypatch.setattr(topo, 'vecfld_from_adata', field)
    monkeypatch.setattr(utils, 'vecfld_from_adata', field)
    calls = []
    monkeypatch.setattr(topo.Topography2D, 'compute_nullclines', lambda *a, **k: calls.append(1))
    np.random.seed(1234)
    topo.topography(a, basis='umap', n=12)
    assert calls == [1]
    np.random.seed(1234)
    B._sampled_topography(b, 12)
    assert calls == [1]  # sampled path never requests nullclines
    for key in ['Xss', 'ftype', 'confidence', 'X_data', 'xlim', 'ylim']:
        np.testing.assert_allclose(a.uns['VecFld_umap'][key], b.uns['VecFld_umap'][key])


def test_benchmark_erythroid_config_selection(fixture):
    import importlib
    benchmark = importlib.import_module('data_preparation.20261007.benchmark.config')
    config = A.load_analysis_config('full', A.HERE / 'configs/full_benchmark_erythroid.json')
    assert config['selection']['values'] == list(benchmark.ERYTHROID_CELLTYPES)
    assert benchmark.ERYTHROID_CELLS == 9815
    assert config['max_cells'] == 0
    _, _, adata, _, _ = fixture
    labels = list(benchmark.ERYTHROID_CELLTYPES) + ['Epiblast']
    adata.obs['celltype'] = labels * 10
    shared, _, count = P._shared_erythropoietic_adata({}, config, P._load_prior_symbols(), adata)
    assert count == 50
    assert shared.obs_names.tolist() == adata.obs_names[adata.obs.celltype != 'Epiblast'].tolist()
    assert not shared.uns['noeq_selection']['fallback']
    adata.obs['celltype'] = ['Epiblast'] * len(adata)
    with pytest.raises(ValueError, match='no cells'):
        P._shared_erythropoietic_adata({}, config, P._load_prior_symbols(), adata)
