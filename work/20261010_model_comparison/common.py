"""Configuration and artifact boundaries. Numerical work lives in imported sources."""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCALE = 'spliced_independent_normalize_total_1e4'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def digest_ids(values):
    return hashlib.sha256(json.dumps(list(map(str, values)), ensure_ascii=False).encode()).hexdigest()

def ids(values):
    result = list(map(str, values))
    require(len(set(result)) == len(result) and all(v.strip() for v in result), 'duplicate/blank IDs')
    return result

def confined(path):
    path = Path(path).expanduser().resolve()
    require(path.is_relative_to(HERE) and path != HERE, f'output must be below {HERE}: {path}')
    return path

def write_json(path, value):
    path = confined(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text())

def root_path(path):
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

def module(name):
    return importlib.import_module(name)

def conditions(extended=False):
    folder = HERE / 'configs' / ('extended' if extended else 'pilot')
    return [p.stem for p in sorted(folder.glob('*.json'))]

def branch_names(c):
    if c['family']=='baseline':return ['cellunet_raw']
    if c['family']=='ode_only':return ['ode_raw']
    return ['ode_raw','cellunet_raw'] + ([] if c['family']=='consistency' else ['hybrid_raw'])

def config(name):
    p = Path(name)
    if not p.is_file():
        matches = [HERE / 'configs' / group / (str(name) + '.json') for group in ('pilot', 'extended')]
        matches = [p for p in matches if p.is_file()]
        require(len(matches) == 1, f'unknown condition: {name}')
        p = matches[0]
    c = read_json(HERE / 'configs/base.json')
    c.update(read_json(p))
    validate(c)
    return c

def validate(c):
    require(c['family'] in ('baseline', 'ode_only', 'joint', 'consistency', 'two_step'), 'unknown family')
    require(c['prediction'] in ('START_X', 'EPSILON'), 'invalid prediction')
    require(c['diffusion_steps'] == 1000 and c['noise_schedule'] == 'linear', 'requires original diffusion')
    require(not c['rescale_timesteps'] and not c['timestep_respacing'], 'unscaled original timesteps required')
    require(c['hybrid'] in ('none', 'blend', 'additive', 'standard', 'ratio_reg', 'normed_learned_scale', 'scale_model', 'ts_sigmoid'), 'unknown composition')
    require(c['objective'] in ('mse', 'ot'), 'objective needs a dedicated adapter')
    require(c['total_steps'] > 0 and c['lr_anneal_steps'] >= c['total_steps'], 'invalid stop/anneal steps')
    require(c['batch_size'] > 0 and c['lr'] > 0, 'invalid optimizer settings')
    require(c['family'] != 'consistency' or c['hybrid'] == 'none', 'consistency samples only CellUNet')
    require(c['family'] != 'two_step' or c['stage1_condition'] == 'A01_cellunet_startx', 'Stage1 must be shared A01')
    require(c['family'] not in ('joint', 'two_step') or c['hybrid'] != 'none', 'Hybrid required')
    require(c['family'] not in ('baseline','ode_only') or c['hybrid']=='none','standalone family cannot declare a Hybrid')
    require(not c['freeze_cellunet'] or c['family']=='two_step','freezing requires a shared Stage1 two-step condition')
    require(isinstance(c['field_timestep'],int) and 0<=c['field_timestep']<1000,'field timestep must be 0..999')
    require(c['soft_weighting'] in ('internal','external_5'),'unknown source penalty convention')
    require(c['family']!='baseline' or c['ode'] is None,'baseline has no ODE branch')
    if c['family']=='consistency':
        require(c['ode'] in ('simple_softplus','hill_after_linear','centered_signed_hill','shifted_hill_rho'),
                'consistency adapter requires the exact 8/30 single-field penalty interface')
        require(c['consistency_lambda']>=0,'negative consistency coefficient')
    if c['hybrid']=='ratio_reg':
        require(c['ode'] in ('geneode','lowrank','lincomb','matsum','lora','lincomb_configurable',
                'racipe','exp','hill_after_linear_experts','centered_hill_experts','shifted_hill_experts'),
                'field has no source ratio-penalty hook; unsupported combination')
    require(c['objective'] != 'ot' or c['family'] not in ('baseline', 'consistency'), 'unsupported OT combination')
    if c['condition'] in conditions():
        expected = dict(prediction='START_X', batch_size=128, lr=1e-4, ema_rate=0.9999, seed=1234,
                        num_samples=3000, optimizer='AdamW', off_mask_lambda=5., ode_reg_lambda=1.)
        require(all(c[k] == v for k, v in expected.items()), 'pilot common settings changed')
        require(c['total_steps'] == (10000 if c['family'] == 'two_step' else 30000), 'pilot update budget changed')

def run_dir(c, run_root=None):
    base = confined(run_root or HERE / 'runs/default')
    return confined(base / c['condition'])

def paths_at(run):
    run = confined(run)
    for name in ('checkpoints', 'logs', 'samples', 'predictions', 'metrics', 'figures'):
        (run / name).mkdir(parents=True, exist_ok=True)
    return run

def source_hashes():
    index = read_json(HERE / 'audit/source_index.json')
    paths = [ROOT / r['path'] for r in index]
    paths += list(HERE.glob('*.py')) + list((HERE / 'adapters').glob('*.py')) + list((HERE / 'configs').rglob('*.json'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}

def provenance(c, meta):
    changes = ['common Mouse linear-spliced HVG data and ordered IDs',
               'Mouse GRN symbol-to-ID mapping and explicit matrix orientation']
    if c['family'] in ('ode_only','consistency') and c['prediction']=='START_X':
        changes.append('historical EPSILON target changed to START_X in adapter')
    if c['family']=='consistency':
        changes += ['8/30 horizon 100000 -> 30000; 9/16 zero-based LR schedule',
                    '10/08 ODE SGD -> AdamW; CellUNet-only sampling retained']
    if c['family']=='joint':changes.append('source two-step wrappers composed with trainable CellUNet')
    if c['family']=='two_step':changes.append('new-data A01 final EMA; stop=10000, anneal horizon=30000')
    return dict(effective_config=c, source_hashes=source_hashes(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        data_sha256=meta['training_sha256'], gene_ids_sha256=digest_ids(meta['gene_ids']),
        cell_ids_sha256=digest_ids(meta['cell_ids']), expression_scale=SCALE,
        training_target=c['prediction'], velocity_semantics='experimental proxy; not established biological ds/dt',
        changes_from_historical_sources=changes,
        grn=meta.get('grn'),
        compute_protocol='shared 30000-update Stage1 + Stage2' if c['family']=='two_step' else 'from random initialization',
        lr_annealing='post-update zero-based index / lr_anneal_steps (9/16 and 10/09 convention)')

def device(name):
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)

def diffusion(c):
    from guided_diffusion.script_util import create_gaussian_diffusion
    from guided_diffusion.gaussian_diffusion import ModelMeanType, LossType
    d = create_gaussian_diffusion(steps=c['diffusion_steps'], noise_schedule=c['noise_schedule'],
        learn_sigma=False, use_kl=False, predict_xstart=c['prediction']=='START_X',
        rescale_timesteps=False, rescale_learned_sigmas=False, timestep_respacing='')
    require(d.model_mean_type == getattr(ModelMeanType, c['prediction']) and d.loss_type == LossType.MSE, 'target mismatch')
    return d
