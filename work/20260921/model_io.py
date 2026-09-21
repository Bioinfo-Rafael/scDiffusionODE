"""Read-only Stage1 discovery. No imports from historical work wrappers."""
import hashlib
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE1 = ROOT / 'work/20260915_x0predict'
CORE = ['guided_diffusion/cell_model.py', 'guided_diffusion/gaussian_diffusion.py',
        'guided_diffusion/script_util.py', 'guided_diffusion/respace.py',
        'guided_diffusion/nn.py', 'guided_diffusion/losses.py']


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state_hash(state):
    """Byte-compatible with 20260915 common.state_hash (parameters AND buffers)."""
    if hasattr(state, 'state_dict'):
        state = state.state_dict()
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(f'{name}:{t.dtype}:{tuple(t.shape)}'.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def gene_hash(genes):
    """Byte-compatible with 20260830 hematopoietic_viz/core.py."""
    h = hashlib.sha256()
    for i, gene in enumerate(genes):
        h.update(str(i).encode('ascii') + b'\0' + str(gene).encode('utf-8') + b'\n')
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def validate_config(c):
    required = dict(suite_version='20260915_x0predict_v1', condition='stage1_cellunet',
                    objective='stage1', predict_xstart=True, diffusion_steps=1000,
                    noise_schedule='linear', timestep_respacing='', learn_sigma=False,
                    use_kl=False, rescale_timesteps=False, rescale_learned_sigmas=False,
                    class_cond=False, ts_layer=None, use_ddim=False,
                    clip_denoised=False, use_fp16=False)
    differences = {k: {'expected': v, 'actual': c.get(k)} for k, v in required.items()
                   if k not in c or c[k] != v}
    if differences:
        raise ValueError(f'Not the audited Stage1 configuration: {differences}')
    # Stage1 build_diffusion does not pass sigma_small; factory default is False.
    if c.get('sigma_small', False) or c.get('nw', 0.5) != 0.5:
        raise ValueError('Sampling variance configuration differs from audited Stage1')


def validate_campaign(campaign):
    import torch
    campaign = Path(campaign).resolve()
    marker = campaign / 'canonical_stage1.json'
    record = read_json(marker)
    path = repo_path(record['checkpoint'])
    if not path.is_relative_to(campaign / 'stage1_cellunet'):
        raise ValueError('Canonical checkpoint is outside this campaign/stage1_cellunet')
    if file_hash(path) != record['checkpoint_sha256']:
        raise ValueError('Canonical checkpoint SHA256 mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if set(payload) != {'state_dict', 'metadata'}:
        raise ValueError('Expected self-contained state_dict + metadata checkpoint')
    meta = payload['metadata']
    c = meta['effective_config']
    validate_config(c)
    if (meta['checkpoint_kind'] != 'ema' or meta['step'] != c['total_steps']
            or record['step'] != meta['step'] or record.get('status') != 'completed'):
        raise ValueError('Canonical checkpoint must be completed final Stage1 EMA')
    if meta['model_mean_type'] != 'START_X' or meta['predict_xstart'] is not True:
        raise ValueError('Checkpoint prediction metadata must be START_X')
    if read_json(campaign / 'configs/stage1_cellunet.json') != c:
        raise ValueError('Campaign config differs from checkpoint effective_config')
    expected_preprocessing = 'load_data(train_vae=True, preprocess=False, layer=None); unchanged X/gene order'
    if meta['preprocessing'] != expected_preprocessing:
        raise ValueError('Unrecognized training representation metadata')
    if state_hash(payload['state_dict']) != record['cellunet_hash']:
        raise ValueError('Canonical parameter/buffer hash mismatch')
    genes = meta['gene_names']
    if len(set(genes)) != len(genes) or not genes:
        raise ValueError('Invalid checkpoint gene names')
    for key in ('gene_order_hash', 'data_sha256'):
        if meta[key] != record[key]:
            raise ValueError(f'Canonical/checkpoint {key} mismatch')
    if gene_hash(genes) != meta['gene_order_hash']:
        raise ValueError('Checkpoint gene-order hash mismatch')
    current_source = {p: file_hash(ROOT / p) for p in CORE}
    source_comparison = {p: {'current': h, 'training': meta.get('source_sha256', {}).get(p)}
                         for p, h in current_source.items()}
    for p, pair in source_comparison.items():
        if pair['training'] is not None and pair['current'] != pair['training']:
            raise ValueError(f'Direct dependency changed since training: {p}')
    return payload, dict(campaign=str(campaign), checkpoint=str(path), canonical_record=record,
                         canonical_sha256=file_hash(marker), checkpoint_sha256_before=record['checkpoint_sha256'],
                         model_hash_before=record['cellunet_hash'], dataset_sha256_expected=meta['data_sha256'],
                         gene_order_hash=meta['gene_order_hash'], effective_config=c,
                         checkpoint_metadata=meta, source_comparison=source_comparison)


def discover(campaign=None, stage1_root=STAGE1):
    root = Path(stage1_root).resolve()
    if campaign:
        p = Path(campaign).expanduser()
        candidates = [p.resolve() if p.is_dir() else root / 'runs' / campaign]
    else:
        candidates = sorted(p.parent for p in (root / 'runs').glob('*/canonical_stage1.json'))
    valid = []
    for path in candidates:
        try:
            payload, provenance = validate_campaign(path)
        except (ValueError, KeyError, OSError, RuntimeError, EOFError,
                pickle.UnpicklingError, TypeError, AttributeError) as exc:
            print(f'INVALID_CAMPAIGN {path}: {exc}', flush=True)
            if campaign:
                raise
        else:
            print(f'VALID_CAMPAIGN {path} checkpoint={provenance["checkpoint"]}', flush=True)
            valid.append(path)
            del payload, provenance
    if len(valid) != 1:
        raise ValueError(f'Found {len(valid)} valid Stage1 campaigns. Specify --campaign explicitly. '
                         f'Candidates: {[str(p) for p in valid]}; searched {root / "runs"}')
    return validate_campaign(valid[0])


def restore(payload, device):
    from guided_diffusion.cell_model import Cell_Unet
    from guided_diffusion.script_util import create_gaussian_diffusion
    from guided_diffusion.gaussian_diffusion import ModelMeanType, ModelVarType, LossType
    c = payload['metadata']['effective_config']
    validate_config(c)
    model = Cell_Unet(input_dim=len(payload['metadata']['gene_names']),
                      hidden_num=c['cell_unet_hidden_num'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model.to(device).eval().requires_grad_(False)
    keys = ('learn_sigma', 'noise_schedule', 'use_kl', 'predict_xstart',
            'rescale_timesteps', 'rescale_learned_sigmas', 'timestep_respacing')
    diffusion = create_gaussian_diffusion(steps=c['diffusion_steps'], **{k: c[k] for k in keys})
    assert diffusion.model_mean_type == ModelMeanType.START_X
    assert diffusion.model_var_type == ModelVarType.FIXED_LARGE
    assert diffusion.loss_type == LossType.MSE
    assert diffusion.num_timesteps == 1000 and diffusion.timestep_map == list(range(1000))
    assert state_hash(model) == state_hash(payload['state_dict'])
    return model, diffusion


def load_real(meta, data_path=None):
    import anndata
    import numpy as np
    from scipy import sparse
    path = repo_path(data_path or meta['effective_config']['data_dir'])
    actual_hash = file_hash(path)
    if actual_hash != meta['data_sha256']:
        raise ValueError('Dataset SHA256 differs from training; --data-path only permits relocation')
    adata = anndata.read_h5ad(path)
    genes = [str(g) for g in adata.var['gene_name'].tolist()]
    if genes != meta['gene_names'] or gene_hash(genes) != meta['gene_order_hash']:
        raise ValueError('Dataset gene ordering differs from checkpoint')
    # Training collate casts X to float32. No filtering, normalization or latent encoding.
    x = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Nonfinite or invalid adata.X')
    return x, adata.obs.copy(), dict(dataset_path=str(path), dataset_sha256=actual_hash,
                                    representation='adata.X cast to float32; no additional preprocessing')
