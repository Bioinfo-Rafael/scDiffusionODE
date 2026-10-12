"""Small synthetic numerical and artifact contracts; never trains real data."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import torch
import anndata as ad

P='work.20261010_model_comparison'
C=importlib.import_module(P+'.common')
M=importlib.import_module(P+'.adapters.model')
O=importlib.import_module(P+'.adapters.objectives')
D=importlib.import_module(P+'.adapters.data')
K=importlib.import_module(P+'.adapters.checkpoint')
T=importlib.import_module(P+'.adapters.training')
V=importlib.import_module(P+'.adapters.velocity')
B=importlib.import_module(P+'.adapters.benchmark')
torch.set_num_threads(1)


def test_source_hashes_portable_checkout(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir()
    suite=root/'work/suite';(suite/'audit').mkdir(parents=True)
    historical=root/'work/old/model.py';historical.parent.mkdir()
    historical.write_text('value = 1\n')
    local=root/'work/old/results/devtools/ruff/__init__.py'
    local.parent.mkdir(parents=True);local.write_text('# local tooling\n')
    (suite/'common.py').write_text('# adapter\n')
    C.write_json(suite/'audit/source_index.json',[
        dict(path=str(historical.relative_to(root))),
        dict(path=str(local.relative_to(root)))])
    subprocess.run(['git','init','-q',str(root)],check=True)
    subprocess.run(['git','add','work/old/model.py'],cwd=root,check=True)
    monkeypatch.setattr(C,'ROOT',root);monkeypatch.setattr(C,'HERE',suite)
    expected={'work/old/model.py':C.sha256(historical),'work/suite/common.py':C.sha256(suite/'common.py')}
    assert C.source_hashes()==expected
    local.unlink()  # A remote checkout never receives ignored developer tooling.
    assert C.source_hashes()==expected
    historical.unlink()  # Missing tracked dependencies must still fail loudly.
    with pytest.raises(FileNotFoundError):C.source_hashes()


def test_checkpoint_pointer_after_run_relocation(tmp_path):
    run=tmp_path/'renamed_run';folder=run/'checkpoints';folder.mkdir(parents=True)
    checkpoint=folder/'model030000.pt';checkpoint.write_bytes(b'checkpoint fixture')
    old=tmp_path/'old_run/checkpoints/model030000.pt'
    pointer=folder/'latest.json'
    C.write_json(pointer,dict(checkpoint=str(old),step=30000))
    assert K.resolve(run)==checkpoint.resolve()
    digest=C.sha256(checkpoint)
    C.write_json(pointer,dict(checkpoint=str(old),step=30000,sha256=digest))
    assert K.resolve(run)==checkpoint.resolve()
    checkpoint.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='SHA256'):K.resolve(run)
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError,match='also checked'):K.resolve(run)
    # Explicit paths are never redirected to another checkpoint.
    assert K.resolve(run,str(old))==old


def small(name):
    c=C.config(name)
    c.update(cell_unet_hidden_num=[16,12,8,8],batch_size=4,expected_cells=16,expected_erythroid=16,
             expected_genes=6,n_hvg=6,log_interval=1,save_interval=2,inference_batch_size=4,sample_batch_size=2)
    return c


def test_dual_optimizer_matches_adamw_and_sgd():
    optim=importlib.import_module(P+'.adapters.optimizers')
    model=torch.nn.Module()
    model.ml_model=torch.nn.Linear(2,1);model.ode_model=torch.nn.Linear(2,1)
    reference=copy.deepcopy(model)
    c=dict(lr=1e-4,weight_decay=1e-4)
    opt=optim.CellAdamWODESGD(model,c)
    cell=torch.optim.AdamW(reference.ml_model.parameters(),**c)
    ode=torch.optim.SGD(reference.ode_model.parameters(),momentum=0.,**c)
    for _ in range(2):
        for p,q in zip(model.parameters(),reference.parameters()):
            p.grad=torch.full_like(p,.25);q.grad=p.grad.clone()
        opt.step();cell.step();ode.step()
    for p,q in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    with pytest.raises(ValueError,match='dual optimizer'):
        opt.load_state_dict(cell.state_dict())


def test_reviewed_d_source_compatibility_rejects_unknown_changes():
    compat=importlib.import_module(P+'.adapters.source_compatibility')
    current=C.source_hashes();saved=dict(current)
    rules=C.read_json(C.HERE/'audit/d_sgd_source_compatibility.json')['files']
    for path,rule in rules.items():
        old=rule['saved'][0]
        if old is None:saved.pop(path,None)
        else:saved[path]=old
    c=C.config(C.HERE/'configs/ode_sgd/D03_consistency_lambda0p1.json')
    assert compat.verify_sources(saved,c,current)
    for path in ['guided_diffusion/cell_model.py',str((C.HERE/'adapters/objectives.py').relative_to(C.ROOT))]:
        tampered=dict(current);tampered[path]='unreviewed'
        with pytest.raises(ValueError,match='unreviewed'):compat.verify_sources(saved,c,tampered)
    with pytest.raises(ValueError,match='source/config'):
        compat.verify_sources(saved,C.config('D03_consistency_lambda0p1'),current)


def test_old_d_checkpoint_load_and_resume_preserve_original(data,tmp_path):
    folder,a,meta=data;c=small('D03_consistency_lambda0p1');c['ode_optimizer']='SGD'
    run=C.run_dir(c,tmp_path/'old')
    path=T.train(c,run,data_dir=folder,target='cpu',stop=1,reuse=False)
    ck=K.read(path);saved=ck['provenance']['source_hashes']
    for filename,rule in C.read_json(C.HERE/'audit/d_sgd_source_compatibility.json')['files'].items():
        old=rule['saved'][0]
        if old is None:saved.pop(filename,None)
        else:saved[filename]=old
    ck['provenance']['git_commit']='82f6b98'
    torch.save(ck,path);digest=C.sha256(path)
    C.write_json(run/'checkpoints/latest.json',dict(checkpoint=str(path),step=1,sha256=digest))
    C.write_json(run/'provenance.json',ck['provenance'])
    K.load(c,run,meta)
    assert C.sha256(path)==digest
    assert C.read_json(run/'source_compatibility.json')['checkpoint_sha256']==digest
    T.train(c,run,data_dir=folder,target='cpu',stop=2,resume='latest',reuse=False)
    assert C.sha256(path)==digest
    assert C.read_json(run/'provenance_before_source_compatibility.json')==ck['provenance']


@pytest.mark.parametrize('name',['D01_consistency_lambda0','D02_consistency_lambda0p001','D03_consistency_lambda0p1','D04_consistency_lambda1'])
def test_d_loss_identical_before_model_expansion(name,data):
    import types
    old=types.ModuleType(P+'.adapters._old_objectives');old.__package__=P+'.adapters'
    source=subprocess.check_output(['git','show','82f6b98:work/20261010_model_comparison/adapters/objectives.py'],cwd=C.ROOT,text=True)
    exec(compile(source,'old_objectives.py','exec'),old.__dict__)
    folder,a,meta=data;c=small(name);c['ode_optimizer']='SGD'
    model=M.build(c,meta['gene_ids'],mask=D.mask_for(meta['gene_ids'],meta),meta=meta).train()
    other=copy.deepcopy(model);x=torch.from_numpy(a.X[:4]);t=torch.tensor([0,49,500,999]);noise=torch.randn_like(x)
    args=(C.diffusion(c),x,t,torch.ones(4))
    first,_=old.loss(c,model,*args,noise=noise);second,_=O.loss(c,other,*args,noise=noise)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    first.backward();second.backward()
    for p,q in zip(model.parameters(),other.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


@pytest.fixture
def data(tmp_path):
    folder=tmp_path/'data';folder.mkdir()
    genes=[f'g{i}' for i in range(6)];cells=[f'c{i}' for i in range(16)]
    a=ad.AnnData(np.random.default_rng(1).uniform(.1,2,(16,6)).astype('float32'),
        obs=pd.DataFrame({'celltype':['Erythroid1']*16,'stage':['E7.5']*16},index=cells),
        var=pd.DataFrame({'gene_name':[f'S{i}' for i in range(6)]},index=genes))
    a.uns['stage1_data']=dict(expression_scale=C.SCALE,preprocess=False,input_sha256='synthetic',hvg_method='seurat_log1p_copy_exact_rank')
    path=folder/'training.h5ad';a.write_h5ad(path)
    edge=folder/'edges.tsv';edge.write_text('from\tto\ng0\tg2\ng2\tg1\n')
    grn=dict(source=str(edge),source_sha256=C.sha256(edge),mapped_tsv=str(edge),mapped_sha256=C.sha256(edge),unique_edges=2)
    meta=dict(training_sha256=C.sha256(path),gene_ids=genes,cell_ids=cells,expression_scale=C.SCALE,input_sha256='synthetic',grn=grn)
    C.write_json(folder/'manifest.json',dict(training=str(path),metadata=meta,grn=grn))
    meta['exp_clip']=D.expression_clip(a.X,C.config('B05_exp'),meta['training_sha256'])
    return folder,a,meta


@pytest.mark.parametrize('name',C.conditions()+C.conditions(True)+C.expansion_conditions())
def test_configs_forward_backward_and_restore(name,data):
    folder,a,meta=data;c=small(name)
    C.validate(C.config(name))
    mask=D.mask_for(meta['gene_ids'],meta)
    model=M.build(c,meta['gene_ids'],mask=mask,meta=meta).train()
    x=torch.from_numpy(a.X[:4]);t=torch.tensor([0,49,499,999]);d=C.diffusion(c)
    if c['objective']=='ot':t=t%50
    value,parts=O.loss(c,model,d,x,t,torch.ones(4))
    assert torch.isfinite(value);value.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for p in model.parameters():
        if p.grad is not None:assert torch.isfinite(p.grad).all()
    if c['freeze_cellunet']:M.single.assert_frozen(model)
    model.eval();pred=model(x,t[:,None]);assert pred.shape==x.shape and torch.isfinite(pred).all()
    restored=M.build(c,meta['gene_ids'],mask=mask,meta=meta,state=model.state_dict()).eval()
    torch.testing.assert_close(pred,restored(x,t[:,None]),rtol=0,atol=0)
    assert all(v.shape==x.shape for v in M.branches(c,model,x,t[:,None]).values())


def test_lambda_zero_has_no_cell_gradient_contribution(data):
    _,a,meta=data;c=small('D01_consistency_lambda0');mask=D.mask_for(meta['gene_ids'],meta)
    model=M.build(c,meta['gene_ids'],mask=mask).train()
    baseline=M.build(small('A01_cellunet_startx'),meta['gene_ids'],state=model.ml_model.state_dict()).train()
    x=torch.from_numpy(a.X[:4]);noise=torch.randn_like(x,dtype=torch.float64);t=torch.tensor([0,49,500,999]);w=torch.ones(4)
    O.loss(c,model,C.diffusion(c),x,t,w,noise=noise)[0].backward()
    O.loss(small('A01_cellunet_startx'),baseline,C.diffusion(c),x,t,w,noise=noise)[0].backward()
    for p,q in zip(model.ml_model.parameters(),baseline.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    model.eval()
    assert torch.equal(model(x,t[:,None]),model.ml_model(x,t[:,None]))


@pytest.mark.parametrize('name',['C01_simple_softplus_blend','C03_simple_softplus_additive'])
def test_hybrid_formula(name,data):
    _,a,meta=data;c=small(name);model=M.build(c,meta['gene_ids'],mask=D.mask_for(meta['gene_ids'],meta)).eval()
    x=torch.from_numpy(a.X[:4]);t=torch.tensor([0,49,500,999])[:,None];r=(1-t/500).clamp_min(0)
    ode=model.ode_model(x,t);cell=model.ml_model(x,t)
    expected=r*ode+((1-r)*cell if c['hybrid']=='blend' else cell)
    torch.testing.assert_close(model(x,t),expected)


def test_mask_direction_and_zero_mapping(data):
    _,a,meta=data;mask=D.mask_for(meta['gene_ids'],meta)
    assert mask[2,0]==1 and mask[0,2]==0 and mask[1,2]==1
    c=small('B01_simple_softplus');model=M.build(c,meta['gene_ids'],mask=mask)
    with torch.no_grad():
        model.ode_model.W.zero_();model.ode_model.W[2,0]=3
    assert model.ode_model.off_mask_penalty()==0
    with torch.no_grad():model.ode_model.W[0,2]=3
    assert model.ode_model.off_mask_penalty()>0
    with pytest.raises(ValueError,match='zero'):M.build(c,meta['gene_ids'],mask=torch.zeros(6,6))


def test_grn_symbols_ambiguity_and_human_not_guessed(data):
    folder,a,_=data;p=folder/'symbol.tsv';p.write_text('from\tto\nS0\tS2\nS0\tS2\ns0\tS2\n')
    edges,report=D.mapped_edges(a,p)
    assert edges==[('g0','g2')] and report['matched_rows']==2
    a.var.loc['g3','gene_name']='S0'
    assert D.mapped_edges(a,p)[0]==[]


def test_freeze_hash_optimizer_and_ema(data):
    _,a,meta=data;c=small('E04_hill_after_linear_additive')
    model=M.build(c,meta['gene_ids'],mask=D.mask_for(meta['gene_ids'],meta));model.train()
    state=copy.deepcopy(model.ml_model.state_dict());before=M.single.state_hash(state)
    opt=M.single.optimizer_for(model,c);ema=copy.deepcopy(model.state_dict())
    x=torch.from_numpy(a.X[:4]);t=torch.tensor([0,49,500,999])
    O.loss(c,model,C.diffusion(c),x,t,torch.ones(4))[0].backward();opt.step()
    M.single.update_ema(ema,model,.9999);M.single.assert_frozen(model,opt,before)
    for key,value in state.items():assert torch.equal(value,ema['ml_model.'+key])


@pytest.mark.parametrize('name',['A01_cellunet_startx','B02_hill_after_linear','C04_hill_after_linear_additive','D03_consistency_lambda0p1',
                               'D01_consistency_lambda0_sgd','D03_consistency_lambda0p1_sgd',
                               'B05_exp','D07_exp_consistency_lambda0p1','D11_lowrank_consistency_lambda0p1',
                               'D15_matsum_consistency_lambda0p1','D19_lora_consistency_lambda0p1'])
def test_train_resume_roundtrip_and_velocity(name,data,tmp_path):
    folder,a,meta=data;c=small(name.removesuffix('_sgd'))
    if name.endswith('_sgd'):c['ode_optimizer']='SGD'
    run=C.run_dir(c,tmp_path/'split')
    p=T.train(c,run,data_dir=folder,target='cpu',stop=1,reuse=False)
    p=T.train(c,run,data_dir=folder,target='cpu',stop=2,resume='latest',reuse=False)
    loaded,ck,_=K.load(c,run,meta)
    assert ck['step']==2
    full=C.run_dir(c,tmp_path/'full');q=T.train(c,full,data_dir=folder,target='cpu',stop=2,reuse=False)
    full_ck=K.read(q)
    if name.endswith('_sgd'):
        assert ck['optimizer']['schema']=='cell_adamw_ode_sgd_v1'
        assert ck['optimizer']['cell']['state']
        assert ck['optimizer']['ode']['state']=={}
        assert ck['optimizer']['ode']['param_groups'][0]['momentum']==0
        assert ck['optimizer']['cell']['param_groups'][0]['lr']==ck['optimizer']['ode']['param_groups'][0]['lr']
    assert M.single.state_hash(ck['raw'])==M.single.state_hash(full_ck['raw'])
    assert M.single.state_hash(ck['ema'])==M.single.state_hash(full_ck['ema'])
    paths=V.export(c,run,data_dir=folder,target='cpu')
    for path in paths:
        pred=ad.read_h5ad(path)
        assert list(pred.obs_names)==meta['cell_ids'] and list(pred.var_names)==meta['gene_ids']
        assert pred.layers['velocity'].shape==a.shape
        field=pred.uns['model_comparison']['field_source']
        expected=M.branches(c,loaded,torch.from_numpy(a.X[:4]),torch.full((4,1),49))[field].detach().numpy()
        np.testing.assert_allclose(pred.layers['velocity'][:4],expected,rtol=1e-6,atol=1e-6)
        assert pred.uns['model_comparison']['training_target']=='START_X'
    assert V.export(c,run,data_dir=folder,target='cpu')==paths
    changed=copy.deepcopy(meta);changed['gene_ids']=list(reversed(changed['gene_ids']))
    with pytest.raises(ValueError,match='gene'):K.load(c,run,changed)
    if name=='A01_cellunet_startx':
        sampled=V.sample(c,run,data_dir=folder,target='cpu',count=2)
        assert ad.read_h5ad(sampled).shape==(2,6)


def stage_fixture(c,meta,path):
    """A labeled synthetic final-step fixture; does NOT represent trained biological weights."""
    baseline=small('A01_cellunet_startx');net=M.build(baseline,meta['gene_ids'])
    ck=dict(schema=K.SCHEMA,config=baseline,step=30000,raw=net.state_dict(),ema=net.state_dict(),
            gene_ids=meta['gene_ids'],cell_ids=meta['cell_ids'],data_sha256=meta['training_sha256'],expression_scale=C.SCALE)
    torch.save(ck,path);return ck


def test_stage2_real_runner_and_checkpoint_validation(data,tmp_path):
    folder,a,meta=data;c=small('E04_hill_after_linear_additive');stage=tmp_path/'synthetic_stage1.pt'
    ck=stage_fixture(c,meta,stage);run=C.run_dir(c,tmp_path/'stage2')
    p=T.train(c,run,data_dir=folder,target='cpu',stop=1,stage1=stage)
    p=T.train(c,run,data_dir=folder,target='cpu',stop=2,resume='latest')
    net,saved,_=K.load(c,run,meta)
    assert M.single.state_hash(net.ml_model)==M.single.state_hash(ck['ema'])
    assert saved['optimizer']['param_groups'][0]['lr']==c['lr']*(1-1/30000)
    assert saved['config']['total_steps']==10000 and saved['config']['lr_anneal_steps']==30000
    K.stage1_state(stage,c,meta)
    ck['step']=29999;torch.save(ck,stage)
    with pytest.raises(ValueError,match='final'):K.stage1_state(stage,c,meta)


def test_prediction_target_is_native():
    from guided_diffusion.gaussian_diffusion import ModelMeanType
    for name,kind in [('A01_cellunet_startx',ModelMeanType.START_X),('X_hill_epsilon_mse',ModelMeanType.EPSILON)]:
        assert C.diffusion(C.config(name)).model_mean_type==kind
    class Constant(torch.nn.Module):
        def forward(self,x,t):return torch.zeros_like(x)
    x=torch.ones(3,6);t=torch.zeros(3,dtype=torch.long);noise=torch.full_like(x,2)
    for name,expected in [('A01_cellunet_startx',1.),('X_hill_epsilon_mse',4.)]:
        terms=C.diffusion(C.config(name)).training_losses(Constant(),x,t,noise=noise)
        torch.testing.assert_close(terms['loss'],torch.full((3,),expected))


def test_benchmark_cli_uses_three_fold_full_trained(tmp_path):
    cmd=B.command(tmp_path/'prediction.h5ad',tmp_path/'genes.txt',tmp_path/'result','method')
    assert cmd[cmd.index('--evaluation')+1]=='3fold'
    assert '--prediction' in cmd and '--fold-predictions' not in cmd and '--profile' not in cmd
    assert cmd[1].endswith('data_preparation/20261007/benchmark/run.py')


def test_legacy_common_does_not_leak():
    from types import ModuleType
    other=ModuleType('other');previous=sys.modules.get('common');sys.modules['common']=other
    try:
        D.load_script('train')
        assert sys.modules['common'] is other
    finally:
        if previous is None:sys.modules.pop('common',None)
        else:sys.modules['common']=previous


def test_cli_all_dry_run_no_artifacts(tmp_path):
    # -S excludes site-packages: dry-run must not depend on NumPy/PyTorch.
    out=subprocess.check_output([sys.executable,'-S','-m',P+'.cli','launch','--all','--dry-run','--run-root',str(tmp_path/'never_created')],cwd=C.ROOT,text=True)
    for name in C.conditions():assert name in out
    assert not (tmp_path/'never_created').exists()


def test_no_shared_config_mutation():
    a=C.config('B01_simple_softplus');a['field_kwargs']['bad']=True
    assert 'bad' not in C.config('B01_simple_softplus')['field_kwargs']
    a=C.config('B01_simple_softplus');a['prediction']='EPSILON'
    with pytest.raises(ValueError,match='pilot'):C.validate(a)
    a=C.config('X_lincomb_ratio_reg');a['ode']='simple_softplus'
    with pytest.raises(ValueError,match='ratio-penalty'):C.validate(a)


def test_protected_sources():
    for path,digest in C.read_json(C.HERE/'audit/protected_files.json').items():
        assert C.sha256(C.ROOT/path)==digest,path


def test_summary_keeps_folds_and_protocols(tmp_path):
    S=importlib.import_module(P+'.adapters.summary')
    root=tmp_path/'results';root.mkdir();run=root/'A01_cellunet_startx';run.mkdir()
    C.write_json(run/'effective_config.json',C.config('A01_cellunet_startx'))
    def fixture(path,protocol,profile):
        path.mkdir(parents=True)
        C.write_json(path/'metadata.json',dict(status='complete',inference_protocol=protocol,profile=profile,reference_manifest_sha256=profile))
        rows=[dict(fold=i,**{m:(i+1)/10 for m in B.METRICS}) for i in range(3)]
        pd.DataFrame(rows).to_csv(path/'metrics_per_fold.csv',index=False)
        pd.DataFrame([dict(metric=m,mean=.2,std=float(np.std([.1,.2,.3])),ddof=0,n_folds=3) for m in B.METRICS]).to_csv(path/'metrics_summary.csv',index=False)
    fixture(run/'metrics/cellunet_raw_t49','full-trained','normalized')
    external=tmp_path/'scvelo';fixture(external,'per-fold','official-raw')
    rows,pending=S.collect(root,[external]);assert len(rows)==2
    assert rows[0]['cbdir_fold2']==.3
    assert {r['profile'] for r in rows}=={'normalized','official-raw'}
    out=S.summarize(root,extra=[external]);assert (out/'comparison.csv').exists()
    assert len(list(out.glob('metrics_*.png')))==2
    C.write_json(external/'metadata.json',dict(status='failed'))
    rows,pending=S.collect(root,[external]);assert len(rows)==1 and pending


def test_expansion_launcher_and_summary(tmp_path):
    root=tmp_path/'expansion'
    out=subprocess.check_output([sys.executable,'-S',str(C.HERE/'scripts/launch_model_expansion.py'),
                                 '--dry-run','--run-root',str(root)],cwd=C.ROOT,text=True)
    assert len(C.expansion_conditions())==32
    assert len(C.conditions())==17
    assert out.count('--stage1-checkpoint')==8
    for name in C.expansion_conditions():assert name in out
    assert not root.exists()
    configs=[C.config(n) for n in C.expansion_conditions()]
    assert all(c['hybrid']=='additive' for c in configs if c['family']=='joint')
    assert len([c for c in configs if c['family']=='consistency'])==16
    C.write_json(root/'campaign.json',dict(conditions=C.expansion_conditions()))
    S=importlib.import_module(P+'.adapters.summary')
    summary=S.summarize(root)
    assert {r['condition'] for r in C.read_json(summary/'status.json')['pending']}==set(C.expansion_conditions())


@pytest.mark.parametrize('kind',['exp','lowrank','matsum','lora'])
def test_expansion_consistency_weighting_and_lambda_zero(kind,data):
    folder,a,meta=data
    name=next(n for n in C.expansion_conditions() if n.startswith('D') and f'_{kind}_' in n and n.endswith('lambda0'))
    c=small(name);mask=D.mask_for(meta['gene_ids'],meta)
    model=M.build(c,meta['gene_ids'],mask=mask,meta=meta).train()
    baseline=M.build(small('A01_cellunet_startx'),meta['gene_ids'],state=model.ml_model.state_dict()).train()
    x=torch.from_numpy(a.X[:4]);noise=torch.randn_like(x,dtype=torch.float64);t=torch.tensor([0,49,500,999]);w=torch.ones(4)
    value,parts=O.loss(c,model,C.diffusion(c),x,t,w,noise=noise)
    np.testing.assert_allclose(parts['ode_regularization_final_weighted'],5*parts['ode_offmask_base_raw'],rtol=1e-6)
    assert parts['cell_ode_consistency_final_weighted_20260830']==0
    value.backward()
    O.loss(small('A01_cellunet_startx'),baseline,C.diffusion(c),x,t,w,noise=noise)[0].backward()
    for p,q in zip(model.ml_model.parameters(),baseline.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


@pytest.mark.parametrize('kind',['exp','lowrank','matsum','lora'])
def test_expansion_frozen_blend_additive_and_cache(kind,data):
    _,a,meta=data;mask=D.mask_for(meta['gene_ids'],meta)
    for mode in ('blend','additive'):
        name=next(n for n in C.expansion_conditions() if n.startswith('E') and n.endswith(f'_{kind}_{mode}'))
        c=small(name);model=M.build(c,meta['gene_ids'],mask=mask,meta=meta).train()
        x=torch.from_numpy(a.X[:4]);t=torch.tensor([0,49,500,999])[:,None];r=(1-t/500).clamp_min(0)
        model.eval()
        ode=model.ode_model(x,t);cell=model.ml_model(x,t)
        torch.testing.assert_close(model(x,t),r*ode+((1-r)*cell if mode=='blend' else cell))
        if kind=='lowrank':
            model.train()
            for times in (t,torch.full_like(t,999)):
                model.zero_grad(set_to_none=True)
                value=O.loss(c,model,C.diffusion(c),x,times.flatten(),torch.ones(4))[0]
                if value.requires_grad:value.backward()
            assert model.ode_model._cached_W_sub is None


def test_expression_clip_and_stable_exponential(data):
    from scipy import sparse
    _,a,meta=data;c=small('B05_exp')
    x=np.zeros((257,3),dtype=np.float32);x[-1,-1]=123
    bounds=D.expression_clip(x,c,'hash')
    assert bounds['observed_min']==0 and bounds['observed_max']==123
    assert bounds['production_min']==0 and bounds['production_max']==140
    assert bounds['rounding_step']==20
    assert D.expression_clip(sparse.csr_matrix(x),c,'hash')==bounds
    positive=D.expression_clip(np.array([[50.,100.]],dtype=np.float32),c,'hash')
    assert positive['production_min']==45 and positive['production_max']==105
    with pytest.raises(ValueError,match='nonzero'):D.expression_clip(np.zeros((2,2)),c,'hash')
    with pytest.raises(ValueError,match='invalid expression'):D.expression_clip(np.array([[np.nan]]),c,'hash')
    meta=dict(meta,exp_clip=dict(bounds,training_sha256=meta['training_sha256']))
    net=M.build(c,meta['gene_ids'],mask=D.mask_for(meta['gene_ids'],meta),meta=meta)
    field=net.ode_model
    with torch.no_grad():
        field.W.zero_();field.b.copy_(torch.tensor([-1e6,-2.,0.,2.,1e6,1e6]))
    zeros=torch.zeros(2,6);out=field(zeros)
    assert torch.isfinite(out).all() and (out>=0).all() and (out<=140).all()
    torch.testing.assert_close(out[0,:4],torch.tensor([0.,np.exp(-2),1.,np.exp(2)],dtype=torch.float32))
    torch.testing.assert_close(out[0,4],torch.tensor(140.))
    out.sum().backward();assert torch.isfinite(field.b.grad).all()
    assert field.b.grad[4]==0 and field.b.grad[2]==2
    state=copy.deepcopy(net.state_dict());state['ode_model.production_bounds'][1]=141
    with pytest.raises(ValueError,match='bound buffers'):
        M.build(c,meta['gene_ids'],mask=D.mask_for(meta['gene_ids'],meta),meta=meta,state=state)


@pytest.mark.parametrize('name',['E05_exp_blend','E09_exp_additive','E06_lowrank_blend','E10_lowrank_additive'])
def test_expansion_stage2_runner_resume(name,data,tmp_path,monkeypatch):
    folder,a,meta=data;c=small(name);stage=tmp_path/'stage.pt';baseline=stage_fixture(c,meta,stage)
    if c['ode']=='lowrank':
        class InactiveSampler:
            def sample(self,n,device):
                return torch.full((n,),999,device=device,dtype=torch.long),torch.ones(n,device=device)
        monkeypatch.setattr(T,'sampler',lambda *args:InactiveSampler())
    run=C.run_dir(c,tmp_path/'stage2')
    T.train(c,run,data_dir=folder,target='cpu',stop=1,stage1=stage)
    T.train(c,run,data_dir=folder,target='cpu',stop=2,resume='latest')
    net,ck,_=K.load(c,run,meta)
    assert M.single.state_hash(net.ml_model)==M.single.state_hash(baseline['ema'])
    if c['ode']=='exp':
        assert ck['provenance']['exp_clip']==meta['exp_clip']
        changed=copy.deepcopy(meta);changed['exp_clip']['production_max']+=1
        with pytest.raises(ValueError,match='bounds differ'):K.load(c,run,changed)


def test_benchmark_adapter_reuses_only_matching_complete_artifact(data,tmp_path,monkeypatch):
    folder,a,meta=data;c=small('B02_hill_after_linear');run=C.run_dir(c,tmp_path/'eval')
    T.train(c,run,data_dir=folder,target='cpu',stop=1,reuse=False)
    V.export(c,run,data_dir=folder,target='cpu')
    ref=tmp_path/'ref';ref.mkdir();C.write_json(ref/'manifest.json',dict(status='complete',profile='normalized',input={'sha256':'synthetic'}))
    calls=[]
    def mock_run(cmd,**kwargs):
        calls.append(cmd);out=Path(cmd[cmd.index('--output-dir')+1]);out.mkdir()
        C.write_json(out/'metadata.json',dict(status='complete'))
    monkeypatch.setattr(B.subprocess,'run',mock_run)
    result=B.evaluate(c,run,reference=ref)
    assert len(calls)==1 and '--prediction' in calls[0]
    assert B.evaluate(c,run,reference=ref)==result and len(calls)==1
    C.write_json(ref/'manifest.json',dict(status='complete',profile='normalized',input={'sha256':'synthetic'},changed=True))
    with pytest.raises(ValueError,match='provenance'):B.evaluate(c,run,reference=ref)


def test_generative_and_training_visualization(data,tmp_path):
    Z=importlib.import_module(P+'.adapters.visualization')
    folder,a,meta=data;c=small('B02_hill_after_linear');run=C.run_dir(c,tmp_path/'viz')
    T.train(c,run,data_dir=folder,target='cpu',stop=1,reuse=False)
    model,_,_=K.load(c,run,meta)
    out=Z.training_plots(run,model);assert (out/'parameter_statistics.csv').is_file()
    gen=ad.AnnData(a.X[:8].copy()+.1,var=a.var.copy())
    gen.write_h5ad(run/'samples/generated.h5ad')
    C.write_json(run/'samples/metadata.json',dict(output_sha256=C.sha256(run/'samples/generated.h5ad')))
    out=Z.generative(c,run,a,limit=16)
    assert (out/'joint_umap.png').is_file() and (out/'collapse_diagnostics.json').is_file()


def test_velocity_visualization_official_pickle_contract(tmp_path):
    import pickle
    python=C.ROOT/'data_preparation/20261007/data/benchmark/.venv-eval/bin/python'
    if not python.exists():pytest.skip('existing benchmark plotting environment unavailable')
    run=tmp_path/'velocity_viz';result=run/'metrics/ode_raw_t49';result.mkdir(parents=True)
    C.write_json(result/'metadata.json',dict(status='complete'))
    rng=np.random.default_rng(5)
    for fold in range(3):
        out=result/f'fold_{fold}';(out/'processed').mkdir(parents=True);(out/'postprocess').mkdir()
        a=ad.AnnData(np.ones((100,6),dtype='float32'),
            obs=pd.DataFrame({'celltype':pd.Categorical(['Erythroid1']*50+['Erythroid2']*50),
                              'stage':pd.Categorical(['E7.5']*50+['E8.0']*50)},index=[f'c{i}' for i in range(100)]))
        a.obsm['X_umap']=rng.normal(size=(100,2));a.layers['candidate_velocity']=rng.normal(size=(100,6)).astype('float32')
        a.write_h5ad(out/'processed/adata_run_candidate_full.h5ad')
        post=dict(cell_label=a.obs.celltype,exp_emb=a.obsm['X_umap'],velocity_emb=np.column_stack([np.ones(100),np.ones(100)*.2]))
        with (out/'postprocess/candidate_full.pkl').open('wb') as f:pickle.dump(post,f)
    subprocess.run([str(python),'-m',P+'.cli','velocity-worker','--run',str(run)],cwd=C.ROOT,check=True,capture_output=True,text=True)
    assert len(list((run/'figures').rglob('stream_*.png')))==6
