"""Real loader and original LoRA architecture fixtures; no optimization/training."""
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
import anndata as ad
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
A = importlib.import_module('work.20261013_D20_noeqThermo.src.d20_adapter')
torch.set_num_threads(1)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


@pytest.fixture
def fixture(tmp_path):
    run = tmp_path / 'run'
    folder = tmp_path / 'data'
    folder.mkdir()
    c = A.C.config(A.CONDITION)
    c.update(cell_unet_hidden_num=[16,12,8,8], expected_cells=60, expected_erythroid=60,
             n_hvg=8, expected_genes=8, inference_batch_size=7)
    genes = ['g'+str(i) for i in range(8)]
    cells = ['cell'+str(i) for i in range(60)]
    x = np.random.default_rng(31).uniform(.1,3,(60,8)).astype('float32')
    a = ad.AnnData(x, obs=pd.DataFrame({'celltype':['Erythroid1']*30+['Erythroid2']*30,
        'stage':['E7.5']*60}, index=cells), var=pd.DataFrame({'gene_name':['symbol'+str(i) for i in range(8)]},index=genes))
    a.uns['stage1_data'] = dict(preprocess=False,input_sha256='synthetic',hvg_method='seurat_log1p_copy_exact_rank')
    path=folder/'training.h5ad'; a.write_h5ad(path)
    edge=folder/'edges.tsv'; edge.write_text('from\tto\ng0\tg2\ng2\tg1\n')
    grn=dict(source=str(edge),source_sha256=A.C.sha256(edge),mapped_tsv=str(edge),mapped_sha256=A.C.sha256(edge),unique_edges=2)
    meta=dict(training_sha256=A.C.sha256(path),gene_ids=genes,cell_ids=cells,
        expression_scale=A.C.SCALE,input_sha256='synthetic',grn=grn)
    write(folder/'manifest.json',dict(training=str(path),metadata=meta,grn=grn))
    write(run/'effective_config.json',c)
    torch.manual_seed(1234)
    model=A.MODELS.build(c,genes,mask=A.DATA.mask_for(genes,meta),meta=meta)
    raw={k:v.clone() for k,v in model.state_dict().items()}
    ema={k:v.clone() for k,v in raw.items()}
    ema['ode_model.b'] += .3
    ck=dict(schema=A.K.SCHEMA,config=c,step=c['total_steps'],raw=raw,ema=ema,
        gene_ids=genes,cell_ids=cells,data_sha256=meta['training_sha256'],expression_scale=A.C.SCALE,
        grn=grn,provenance={'source_hashes':A.C.source_hashes()})
    checkpoint=run/'checkpoints/model030000.pt';checkpoint.parent.mkdir()
    torch.save(ck,checkpoint)
    write(checkpoint.parent/'latest.json',dict(checkpoint=str(checkpoint),sha256=A.C.sha256(checkpoint),step=30000))
    return run,folder,a,ck,checkpoint


def make(fixture, **kwargs):
    run,folder,*_=fixture
    return A.D20Adapter(run,data_dir=folder,device='cpu',**kwargs)


def evaluate(adapter, a):
    return adapter(A.DATA.dense_rows(a.X),a.var_names.tolist(),{'evaluation_cell_ids':a.obs_names.tolist()})[0]


def test_original_lora_branch_and_raw_ema(fixture):
    _,_,a,_,_=fixture
    ema=make(fixture);raw=make(fixture,weights='raw')
    def forbidden(*args,**kwargs):
        raise AssertionError('CellUNet/Hybrid used')
    ema.model.forward=forbidden;ema.model.ml_model.forward=forbidden
    value=evaluate(ema,a)
    with torch.no_grad():
        expected=ema.model.ode_model(torch.from_numpy(a.X),torch.full((len(a),1),49,dtype=torch.long)).numpy()
    np.testing.assert_allclose(value,expected,rtol=2e-5,atol=2e-6)
    assert value.shape==a.shape and np.isfinite(value).all()
    assert not np.allclose(value,evaluate(raw,a))
    assert ema.stamp['checkpoint_step']==30000


@pytest.mark.parametrize('change,match',[
    ('source','unreviewed source'),('config','effective config'),('genes','gene IDs'),
    ('data','training data'),('step','final training'),('weights','nonfinite checkpoint')])
def test_reject_checkpoint_tampering(fixture,change,match):
    _,_,_,ck,path=fixture
    if change=='source': ck['provenance']['source_hashes']['ODE/ode_20260609_mathmlp.py']='wrong'
    if change=='config': ck['config']=dict(ck['config'],lr=.05)
    if change=='genes': ck['gene_ids']=ck['gene_ids'][::-1]
    if change=='data': ck['data_sha256']='wrong'
    if change=='step': ck['step']=29000
    if change=='weights': ck['ema']['ode_model.b'][0]=float('nan')
    torch.save(ck,path)
    with pytest.raises(ValueError,match=match):make(fixture,checkpoint=str(path))


def test_latest_digest_and_real_cell_guard(fixture):
    _,_,a,_,path=fixture
    adapter=make(fixture)
    altered=a.copy();altered.X[0,0]+=1
    with pytest.raises(ValueError,match='real-cell'):evaluate(adapter,altered)
    with pytest.raises(ValueError,match='gene IDs'):evaluate(adapter,a[:,::-1])
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA256'):make(fixture)


def test_export_reuse_and_corruption(fixture):
    run,_,a,_,_=fixture
    adapter=make(fixture)
    expected=evaluate(adapter,a)
    pred=ad.AnnData(X=None,obs=a.obs.copy(),var=a.var.copy());pred.layers['velocity']=expected
    path=run/'predictions/ode_raw_t49.h5ad';path.parent.mkdir()
    pred.write_h5ad(path)
    stamp=dict(adapter.stamp,cell_ids=a.obs_names.tolist(),gene_ids=a.var_names.tolist(),output_sha256=A.C.sha256(path))
    write(path.with_suffix('.json'),stamp)
    cached=make(fixture)
    def forbidden(*args,**kwargs):raise AssertionError('cache caused duplicate inference')
    cached.model.ode_model.forward=forbidden
    np.testing.assert_array_equal(evaluate(cached,a[[7,2,15]]),expected[[7,2,15]])
    stamp['weights']='raw';write(path.with_suffix('.json'),stamp)
    with pytest.raises(ValueError,match='weights'):make(fixture)
    stamp['weights']='ema';stamp['output_sha256']='wrong';write(path.with_suffix('.json'),stamp)
    with pytest.raises(ValueError,match='SHA256'):make(fixture)


def test_exact_compatibility_rules_are_still_strict():
    module=importlib.import_module(A.P+'.adapters.source_compatibility')
    current=A.C.source_hashes();saved=dict(current)
    rules=A.C.read_json(A.C.HERE/'audit/model_expansion_source_compatibility.json')['files']
    changed=[]
    for p,r in rules.items():
        if '20260821_noeqThermo/' in p:
            assert current[p]==r['current']
            saved[p]=r['saved'][0];changed.append(p)
    assert module.verify_sources(saved,A.C.config(A.CONDITION),current)==sorted(changed)
    saved[changed[0]]='unreviewed'
    with pytest.raises(ValueError,match='unreviewed'):module.verify_sources(saved,A.C.config(A.CONDITION),current)
