"""Isolated experiment paths, strict four-condition configuration, and legacy imports."""
from pathlib import Path
import hashlib
import importlib
import json
import os
import random
import sys

SUITE = Path(__file__).resolve().parents[1]
REPO = SUITE.parents[1]
sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLCONFIGDIR', str(SUITE / 'runs/.cache/matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(SUITE / 'runs/.cache/numba'))
for p in (REPO, SUITE):
    if str(p) not in sys.path:
        sys.path.append(str(p))
CONDITIONS = {'lambda1': 1., 'lambda0p1': .1, 'lambda0p001': .001, 'lambda0': 0.}
STEPS = (0, 200, 400, 600, 800, 1000)

def legacy(name):
    return importlib.import_module('work.20260830.' + name)

def read_json(path):
    return json.loads(Path(path).read_text())

def inside(path):
    path = Path(path).expanduser().resolve()
    path.relative_to(SUITE)
    return path

def write_json(path, value):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def validate(c):
    if c['experiment'] not in CONDITIONS or c['cell_ode_reg_lambda_20260830'] != CONDITIONS[c['experiment']]:
        raise ValueError('Only the four specified lambda conditions are allowed')
    for key, expected in dict(ode_type='hill_after_linear', lr=1e-4, ode_lr=1e-4,
                              ode_momentum=0., weight_decay=1e-4, ode_weight_decay=1e-4,
                              diffusion_steps=1000, timestep_respacing='', use_ddim=False,
                              use_fp16=False, schedule_sampler='uniform', SoftReg=True,
                              use_mask_reg=True, ema_rate='0.9999',
                              cell_ode_reg_schedule_20260830='constant').items():
        if c[key] != expected:
            raise ValueError(f'{key} must be {expected!r} in this experiment')
    if int(c['total_steps']) <= 0 or c['total_steps'] != c['lr_anneal_steps']:
        raise ValueError('total_steps and lr_anneal_steps must be equal and positive')
    for key in ('batch_size','sample_batch_size','num_samples','save_interval','log_interval'):
        if int(c[key]) <= 0:
            raise ValueError(f'{key} must be positive')
    # Default full batches preserve the legacy loss exactly. Reject ambiguous legacy
    # microbatch accumulation, which sums means instead of weighting by batch size.
    if c['microbatch'] not in (-1, c['batch_size']):
        raise ValueError('Use microbatch=-1 (legacy default) or batch_size')
    return c

def config_for(name, overrides=()):
    c = read_json(SUITE / 'configs/base.json')
    c.update(read_json(SUITE / f'configs/{name}.json'))
    for item in overrides:
        key, value = item.split('=', 1)
        if key not in c:
            raise ValueError(f'Unknown config key: {key}')
        try:
            c[key] = json.loads(value)
        except json.JSONDecodeError:
            c[key] = value
    if any(x.startswith('total_steps=') for x in overrides):
        c['lr_anneal_steps'] = c['cell_ode_reg_schedule_steps_20260830'] = c['total_steps']
    return validate(c)

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def device_for(c):
    import torch
    choice = c['device']
    if choice == 'auto':
        choice = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(choice)

def diffusion_for(c):
    from guided_diffusion.script_util import create_gaussian_diffusion
    keys = ('learn_sigma','noise_schedule','use_kl','predict_xstart','rescale_timesteps',
            'rescale_learned_sigmas','timestep_respacing')
    return create_gaussian_diffusion(steps=c['diffusion_steps'], **{k:c[k] for k in keys})

def load_cells(c):
    import scanpy as sc
    import numpy as np
    a = sc.read_h5ad(c['data_dir'])
    genes = list(map(str, a.var['gene_name']))
    if len(genes) != a.n_vars or len(set(genes)) != len(genes):
        raise ValueError('gene_name must be unique')
    x = a.X.toarray() if hasattr(a.X, 'toarray') else a.X
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite training data')
    from sklearn.preprocessing import LabelEncoder
    labels = LabelEncoder().fit_transform(a.obs['celltype'])
    return x, genes, labels

def build(c, genes, device):
    return legacy('models.factory').build_model_from_config(c, genes, 1000, device)

def torch_load(path):
    import torch
    # Only load locally produced, trusted experiment checkpoints. RNG state uses numpy.
    return torch.load(path, map_location='cpu', weights_only=False)

def checkpoint(run, ema=True):
    run = inside(run)
    c = read_json(run / 'exp_config.json')
    state = read_json(run / 'checkpoints/latest.json')
    name = f"ema_{c['ema_rate']}_{state['completed_steps']:06d}.pt" if ema else f"model{state['completed_steps']:06d}.pt"
    return run / 'checkpoints/segment_000/model' / name

def campaign_runs(campaign, require_all=True):
    campaign = inside(campaign)
    runs = [campaign / n for n in CONDITIONS if (campaign/n/'exp_config.json').exists()]
    if not runs or (require_all and len(runs) != 4):
        raise ValueError('This operation requires all four trained conditions')
    configs = [validate(read_json(r/'exp_config.json')) for r in runs]
    exclude = {'experiment','cell_ode_reg_lambda_20260830'}
    signatures = [{k:v for k,v in c.items() if k not in exclude} for c in configs]
    if any(c != signatures[0] for c in signatures[1:]):
        raise ValueError('Conditions must share every setting except lambda/name')
    return runs
