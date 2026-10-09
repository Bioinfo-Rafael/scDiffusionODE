"""Standalone Stage 1 adapters. No GRN, ODE or Stage 2 is constructed."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
SCALE = 'spliced_independent_normalize_total_1e4'
BENCH = importlib.import_module('data_preparation.20261007.benchmark.config')
ERYTHROID = BENCH.ERYTHROID_CELLTYPES
SOURCE = importlib.import_module('work.20260915_x0predict.common')
MODELS = importlib.import_module('work.20260913_2step.models')
OBJECTIVES = importlib.import_module('work.20260915_x0predict.training.objectives')
SOURCE_FILES = [
    'guided_diffusion/cell_model.py', 'guided_diffusion/gaussian_diffusion.py',
    'guided_diffusion/respace.py', 'guided_diffusion/script_util.py',
    'guided_diffusion/resample.py', 'guided_diffusion/cell_datasets_loader.py',
    'work/20260913_2step/models/__init__.py',
    'work/20260915_x0predict/training/objectives.py',
    'work/20260915_x0predict/training/runner.py',
    'work/20260915_x0predict/configs/base.json',
    'work/20260915_x0predict/common.py',
    'work/20260913_2step/common.py',
    'work/20260913_2step/configs/base.json',
    'work/20260913_2step/configs/stage1_cellunet.json',
    'work/20260915_x0predict/configs/stage1_cellunet.json',
    'work/20260915_x0predict/sampling/trajectory.py',
    'work/20260830/hematopoietic_viz/core.py',
    'work/20260830/hematopoietic_viz/plotting.py',
    'work/20260803_ODE_hill_exp/configs/base.json',
]


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
    return hashlib.sha256(json.dumps(list(values), ensure_ascii=False).encode()).hexdigest()


def ids(values):
    result = list(map(str, values))
    require(len(set(result)) == len(result) and all(x.strip() for x in result), 'duplicate/blank IDs')
    return result


def confined(path):
    path = Path(path).expanduser().resolve()
    require(path.is_relative_to(HERE) and path != HERE, f'output must be below {HERE}')
    return path


def write_json(path, obj):
    path = confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + '\n')
    os.replace(tmp, path)


def config(path=None):
    c = json.loads(Path(path or HERE / 'config.json').read_text())
    base = SOURCE.effective_config('stage1_cellunet')
    # Inherit only Stage 1 settings; no unused GRN paths or Hybrid settings.
    keys = ('cell_unet_hidden_num', 'diffusion_steps', 'noise_schedule', 'timestep_respacing',
            'learn_sigma', 'use_kl', 'predict_xstart', 'rescale_timesteps',
            'rescale_learned_sigmas', 'schedule_sampler', 'lr', 'weight_decay',
            'lr_anneal_steps', 'total_steps', 'batch_size', 'ema_rate',
            'log_interval', 'save_interval', 'seed', 'num_samples', 'sample_batch_size')
    c['training'] = {k: base[k] for k in keys}
    c['training']['objective'] = 'stage1'
    c['source_hashes'] = {p: sha256(ROOT / p) for p in SOURCE_FILES}
    for key in ('input', 'runs'):
        p = Path(c[key]).expanduser()
        c[key] = str((ROOT / p if not p.is_absolute() else p).resolve())
    confined(c['runs'])
    return c


def paths(c):
    run = confined(c['runs'])
    for name in ('data', 'checkpoints', 'samples', 'figures', 'predictions', 'metrics', 'logs'):
        (run / name).mkdir(parents=True, exist_ok=True)
    return run


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, default=HERE / 'config.json')
    return p


def device_arg(p):
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])


def device(name):
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)


def diffusion(c):
    from guided_diffusion.script_util import create_gaussian_diffusion
    t = c['training']
    d = create_gaussian_diffusion(steps=t['diffusion_steps'], **{k: t[k] for k in (
        'learn_sigma', 'noise_schedule', 'use_kl', 'predict_xstart', 'rescale_timesteps',
        'rescale_learned_sigmas', 'timestep_respacing')})
    SOURCE.assert_start_x(d)
    require(d.num_timesteps == 1000, 'requires unrespaced 1000 steps')
    return d


def read_data(c):
    import anndata as ad
    a = ad.read_h5ad(paths(c) / 'data/training.h5ad')
    meta = json.loads((paths(c) / 'data/metadata.json').read_text())
    require(sha256(paths(c) / 'data/training.h5ad') == meta['training_sha256'], 'training file changed')
    require(ids(a.var_names) == meta['gene_ids'], 'training gene order mismatch')
    require(ids(a.obs_names) == meta['cell_ids'], 'training cell order mismatch')
    require(a.n_obs == c['expected_cells'] and a.n_vars == c['n_hvg'], 'training shape mismatch')
    return a, meta


def resolve_checkpoint(c, value=None):
    return Path(value).resolve() if value else Path(json.loads(
        (paths(c) / 'checkpoints/latest.json').read_text())['checkpoint'])


def load_checkpoint(c, value=None, target='cpu', weights='ema'):
    import torch
    path = resolve_checkpoint(c, value)
    # Only locally generated, trusted torch bundles should be opened.
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a, meta = read_data(c)
    require(ck['schema'] == 'cellunet_stage1_start_x_v1', 'not a standalone Stage 1 checkpoint')
    require(ck['training'] == c['training'], 'training config differs from checkpoint')
    require(ck['source_hashes'] == c['source_hashes'], 'source implementation changed')
    require(ck['data_sha256'] == meta['training_sha256'], 'checkpoint data hash mismatch')
    require(ck['gene_ids'] == ids(a.var_names), 'checkpoint gene IDs/order mismatch')
    require(ck['cell_ids'] == ids(a.obs_names), 'checkpoint cell IDs/order mismatch')
    model = MODELS.build_model(c['training'], ck['gene_ids'], state=ck[weights])
    model.to(target).eval()
    return model, ck, a, path


def dense_rows(x):
    import numpy as np
    from scipy import sparse
    return np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=np.float32)


def cli(main):
    try:
        main()
    except Exception:
        # Every CLI failure retains a traceback without claiming successful metrics.
        log = HERE / 'runs/logs' / ('failure_' + uuid.uuid4().hex + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc())
        raise


def write_h5ad(a, path):
    """Publish complete files only, never replacing a previous artifact."""
    path = confined(path)
    require(not path.exists(), f'refusing overwrite: {path}')
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.h5ad')
    try:
        a.write_h5ad(tmp)
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
