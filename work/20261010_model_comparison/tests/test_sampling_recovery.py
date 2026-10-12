"""Sampling divergence must not change the model or suppress velocity evaluation."""
import importlib
import subprocess
import uuid
import torch
import pytest

P='work.20261010_model_comparison'
C=importlib.import_module(P+'.common')
L=importlib.import_module(P+'.scripts.launch_model_expansion')
DIAG=importlib.import_module(P+'.scripts.diagnose_sampling')


def test_sampling_failure_continues_independent_stages(tmp_path,monkeypatch):
    names=['B06_lowrank','B07_matsum'];configs=[C.config(n) for n in names]
    manifest=dict(conditions=names,configs=configs)
    monkeypatch.setattr(C,'expansion_campaign',lambda *args,**kwargs:manifest)
    run=tmp_path/'campaign'
    for name in names:
        C.write_json(run/name/'checkpoints/latest.json',dict(sha256='verified-'+name))
    calls=[]
    def fake(command,**kwargs):
        calls.append(command)
        if '--stages' not in command:return
        name=command[command.index('--condition')+1]
        action=command[command.index('--stages')+1]
        if action=='sample':
            if name=='B06_lowrank':
                C.write_json(run/name/'logs'/f'failure_sample_{uuid.uuid4().hex}.json',
                             dict(traceback='Traceback\nFloatingPointError: ancestral state contains NaN/Inf\n'))
                raise subprocess.CalledProcessError(1,command)
            path=run/name/'samples/generated.h5ad';path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    monkeypatch.setattr(L.subprocess,'run',fake)
    args=['--run-root',str(run),'--stages','sample','export_velocity','evaluate','visualize']
    L.main(args)
    progress=C.read_json(run/'campaign_progress.json')
    low=progress['B06_lowrank']
    assert low['sample']['status']=='failed_nonfinite'
    assert low['visualize_samples']['status']=='unavailable'
    for action in ('export_velocity','evaluate','visualize_training','visualize_velocity'):
        assert low[action]['status']=='complete'
    assert progress['B07_matsum']['sample']['status']=='complete'
    assert '--allow-missing' in calls[-1]
    calls.clear();L.main(args)
    assert not any('--stages' in cmd and cmd[cmd.index('--stages')+1]=='sample' and 'B06_lowrank' in cmd for cmd in calls)
    assert C.read_json(run/'campaign_progress.json')['B06_lowrank']['sample']['reused_failure']
    calls.clear();L.main(args+['--retry-failed-sampling'])
    assert any('--stages' in cmd and cmd[cmd.index('--stages')+1]=='sample' and 'B06_lowrank' in cmd for cmd in calls)


def test_unrelated_sampling_error_still_stops(tmp_path,monkeypatch):
    c=C.config('B06_lowrank');run=tmp_path/'campaign'
    monkeypatch.setattr(C,'expansion_campaign',lambda *args,**kwargs:dict(conditions=[c['condition']],configs=[c]))
    def fake(command,**kwargs):
        C.write_json(run/c['condition']/'logs/failure_sample_bad.json',dict(traceback='ValueError: checkpoint mismatch'))
        raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(L.subprocess,'run',fake)
    with pytest.raises(subprocess.CalledProcessError):L.main(['--run-root',str(run),'--stages','sample','export_velocity'])
    assert not (run/c['condition']/'samples/nonfinite_failure.json').exists()


def test_sampler_trace_reports_first_nonfinite_without_modification():
    model=torch.nn.Linear(2,2);rows=[]
    class FakeDiffusion:
        def p_sample_loop_progressive(self,model,shape,**kwargs):
            assert shape==(2,2)
            assert kwargs['clip_denoised'] is False and kwargs['nw']==.5
            yield dict(sample=torch.ones(shape)*100,pred_xstart=torch.ones(shape)*200)
            yield dict(sample=torch.full(shape,float('inf')),pred_xstart=torch.full(shape,float('nan')))
            raise AssertionError('must stop at first nonfinite step')
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=DIAG.trace_sampling(FakeDiffusion(),model,count=4,batch_size=2,genes=2,steps=1000,emit=rows.append)
    assert result['status']=='failed_nonfinite' and result['first_failure']['timestep']==998
    assert rows[0]['sample']['max_abs']==100
    assert rows[1]['sample']['nonfinite_elements']==4 and rows[1]['pred_xstart']['min'] is None
    for k,v in before.items():assert torch.equal(v,model.state_dict()[k])
