"""Strict checkpoint boundaries: architecture, target, ordered IDs, data and source hashes."""
from __future__ import annotations
import torch
from ..common import *
from .model import build, single
from .data import mask_for

SCHEMA = 'model_comparison_v1'


def resolve(run, value=None):
    if value and value != 'latest': return root_path(value)
    pointer = read_json(Path(run) / 'checkpoints/latest.json')
    path = root_path(pointer['checkpoint'])
    if 'sha256' in pointer:
        require(sha256(path)==pointer['sha256'], 'checkpoint SHA256 mismatch')
    return path


def read(path):
    # Only trusted repository-generated bundles (optimizer + Python/NumPy RNG).
    return torch.load(path, map_location='cpu', weights_only=False)


def check_ids(ck, meta):
    require(ck['gene_ids']==meta['gene_ids'], 'checkpoint gene IDs/order mismatch')
    require(ck['cell_ids']==meta['cell_ids'], 'checkpoint cell IDs/order mismatch')
    require(ck['data_sha256']==meta['training_sha256'], 'checkpoint training data mismatch')
    require(ck['expression_scale']==SCALE, 'checkpoint expression scale mismatch')


def stage1_state(path, c, meta):
    """Accept 10/09 bundle or our A01; reject unverifiable legacy bare state dicts."""
    ck = read(path)
    require(ck.get('schema') in ('cellunet_stage1_start_x_v1',SCHEMA),
            'Stage1 needs verifiable 10/09/A01 bundle; bare 9/15 weights lack new cell/data provenance')
    check_ids(ck, meta)
    if ck['schema']==SCHEMA:
        saved=ck['config']
        require(saved['family']=='baseline' and saved['condition']==c['stage1_condition'], 'not shared A01')
        require(saved['prediction']=='START_X', 'Stage1 must predict START_X')
        widths=saved['cell_unet_hidden_num']; final=saved['total_steps']
    else:
        saved=ck['training']; widths=saved['cell_unet_hidden_num']; final=saved['total_steps']
        require(saved['predict_xstart'] and ck['prediction']=='x_start', 'Stage1 target mismatch')
        expected = baseline_training_config(c)
        require(saved==expected, '10/09 training configuration differs from A01')
        for filename,digest in ck['source_hashes'].items():
            require(sha256(ROOT / filename)==digest, f'Stage1 source changed: {filename}')
    require(widths==c['cell_unet_hidden_num'], 'Stage1 architecture mismatch')
    require(ck['step']==final==30000, 'shared Stage1 must be final 30000-update EMA')
    from guided_diffusion.cell_model import Cell_Unet
    probe=Cell_Unet(input_dim=len(meta['gene_ids']),hidden_num=widths)
    probe.load_state_dict(ck['ema'],strict=True)
    for name,value in ck['ema'].items():require(torch.isfinite(value).all(), f'nonfinite Stage1 {name}')
    return ck['ema'], dict(checkpoint=str(Path(path).resolve()), sha256=sha256(path),
                          state_hash=single.state_hash(ck['ema']),step=ck['step'],weights='ema')


def baseline_training_config(c):
    original=module('work.20261009_newBenchmark.common').config()['training']
    original.update(cell_unet_hidden_num=c['cell_unet_hidden_num'],batch_size=c['batch_size'],
                    lr=c['lr'],weight_decay=c['weight_decay'],ema_rate=str(c['ema_rate']),
                    seed=c['seed'],total_steps=c['total_steps'],lr_anneal_steps=c['lr_anneal_steps'],
                    log_interval=c['log_interval'],save_interval=c['save_interval'],
                    num_samples=c['num_samples'],sample_batch_size=c['sample_batch_size'])
    return original


def load(c, run, meta, *, value=None, weights='ema', target='cpu'):
    path=resolve(run,value);ck=read(path)
    if ck.get('schema')=='cellunet_stage1_start_x_v1':
        require(c['family']=='baseline', 'legacy bundle is baseline only')
        check_ids(ck,meta)
        require(ck['training']==baseline_training_config(c), 'baseline training config changed')
        for file,digest in ck['source_hashes'].items():
            require(sha256(ROOT/file)==digest, f'legacy source changed: {file}')
        model=build(c,meta['gene_ids'],state=ck[weights])
    else:
        require(ck.get('schema')==SCHEMA, 'unknown checkpoint schema')
        require(ck['config']==c, 'effective config differs from checkpoint')
        check_ids(ck,meta)
        require(ck['provenance']['source_hashes']==source_hashes(), 'source/config code changed since checkpoint')
        require(ck['grn']==meta['grn'], 'checkpoint GRN differs')
        mask=None if c['family']=='baseline' else mask_for(meta['gene_ids'],meta)
        model=build(c,meta['gene_ids'],mask=mask,meta=meta,state=ck[weights])
        if c['freeze_cellunet']:
            single.assert_frozen(model,expected=ck['stage1']['state_hash'])
    return model.to(target).eval(),ck,path


def save(c, run, meta, model, opt, ema, step, batches, rng, prov, stage1=None):
    path=confined(run / 'checkpoints' / f'model{step:06d}.pt')
    require(not path.exists(),f'checkpoint exists: {path}')
    ck=dict(schema=SCHEMA,config=c,step=step,raw=model.state_dict(),ema=ema,
        optimizer=opt.state_dict(),batches=batches.state(),rng=rng,provenance=prov,
        gene_ids=meta['gene_ids'],cell_ids=meta['cell_ids'],data_sha256=meta['training_sha256'],
        expression_scale=SCALE,grn=meta['grn'],stage1=stage1)
    tmp=path.with_suffix('.tmp');torch.save(ck,tmp);os.replace(tmp,path)
    write_json(run/'checkpoints/latest.json',dict(checkpoint=str(path),sha256=sha256(path),step=step))
    return path
