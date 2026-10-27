"""Read trained D16/D20; reuse original field constructors and provenance validation."""
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from artifacts import sha256

ROOT = Path(__file__).resolve().parents[2]
P = 'work.20261010_model_comparison'
CONDITIONS = {'matsum': 'D16_matsum_consistency_lambda1', 'lora': 'D20_lora_consistency_lambda1'}


def root_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_obs_and_umap(path):
    """Read only HDF5 obs and saved UMAP, never full expression/layers from the source."""
    import h5py
    def array(node):
        if isinstance(node, h5py.Group):
            if 'categories' not in node or 'codes' not in node:
                raise ValueError(f'Unsupported metadata encoding: {node.name}')
            return pd.Categorical.from_codes(node['codes'][()], array(node['categories']))
        return node.asstr()[()] if node.dtype.kind in 'SO' else node[()]
    with h5py.File(path, 'r') as f:
        obs = f['obs']
        key = obs.attrs['_index']
        index = array(obs[key])
        columns = list(obs.attrs['column-order'])
        table = pd.DataFrame({str(k): array(obs[k]) for k in columns}, index=index)
        umap = f['obsm/X_umap'][()] if 'obsm/X_umap' in f else None
    return table, umap


def load_trained_model(kind, cfg, device):
    """Read-only boundary: final checkpoint, config, ordered IDs, hashes and GRN must match."""
    common = importlib.import_module(P + '.common')
    checkpoints = importlib.import_module(P + '.adapters.checkpoint')
    data = importlib.import_module(P + '.adapters.data')
    models = importlib.import_module(P + '.adapters.model')
    verify = importlib.import_module(P + '.adapters.source_compatibility').verify_sources
    run = root_path(cfg['run_root']) / CONDITIONS[kind]
    config = common.read_json(run / 'effective_config.json')
    common.validate(config)
    expected = dict(condition=CONDITIONS[kind], ode=kind, family='consistency',
                    hybrid='none', consistency_lambda=1., prediction='START_X')
    if any(config[k] != v for k, v in expected.items()):
        raise ValueError(f'Expected trained D16/D20 consistency lambda=1: {run}')
    adata, meta = data.read_data(config, root_path(cfg['data_dir']))
    path = checkpoints.resolve(run)
    ck = checkpoints.read(path)
    if ck.get('schema') != checkpoints.SCHEMA or ck['config'] != config:
        raise ValueError('Checkpoint schema/config mismatch')
    checkpoints.check_ids(ck, meta)
    if ck['step'] != config['total_steps'] or ck['grn'] != meta['grn']:
        raise ValueError('Checkpoint is not final or GRN differs')
    pointer = common.read_json(run / 'checkpoints/latest.json')
    if pointer['step'] != ck['step']:
        raise ValueError('Latest pointer step mismatch')
    reviewed = verify(ck['provenance']['source_hashes'], config)
    mask = data.mask_for(meta['gene_ids'], meta)
    field = models.build_field(config, meta['gene_ids'], mask, meta)
    state = {k[len('ode_model.'):]: v for k, v in ck[cfg['weights']].items() if k.startswith('ode_model.')}
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite checkpoint parameter')
    field.load_state_dict(state, strict=True)
    field.to(device).eval().requires_grad_(False)
    if field.K != 8:
        raise ValueError('This analysis expects eight basis matrices')
    # Preserve training IDs; add source annotations only after exact ID/hash checks.
    source = root_path(config['input'])
    source_available = source.is_file()
    if source_available:
        if sha256(source) != meta['input_sha256']:
            raise ValueError('Original annotation source hash mismatch')
        obs, umap = read_obs_and_umap(source)
        if not obs.index.is_unique:
            raise ValueError('Source cell IDs are not unique')
        rows = obs.index.get_indexer(adata.obs_names)
        if (rows < 0).any():
            raise ValueError('Training cell IDs missing in annotation source')
        aligned = obs.iloc[rows].copy()
        for key in ('celltype', 'stage'):
            if not np.array_equal(aligned[key].astype(str), adata.obs[key].astype(str)):
                raise ValueError(f'Source/training annotation mismatch: {key}')
        adata.obs = aligned
        if umap is not None:
            adata.obsm['X_umap_source'] = umap[rows]
    stamp = dict(condition=CONDITIONS[kind], checkpoint=str(path), checkpoint_sha256=sha256(path),
                 weights=cfg['weights'], step=ck['step'], training_sha256=meta['training_sha256'],
                 cell_ids_sha256=common.digest_ids(adata.obs_names), gene_ids_sha256=common.digest_ids(adata.var_names),
                 annotation_source=str(source), source_available=source_available,
                 source_compatibility_reviewed=reviewed, effective_config=config,
                 inference='clean normalized training X, fixed diffusion t; no noise, optimizer or CellUNet forward')
    return field, adata, stamp


def shared_matrices(field):
    """B=A for MatSum; B=UV^T for LoRA. Preserve W0 separately."""
    if field.model_type == 'matsum':
        basis = field.A.detach()
        base = torch.zeros_like(basis[0])
    else:
        basis = torch.einsum('kir,kjr->kij', field.U, field.V).detach()
        base = field.W0.detach()
    return basis.cpu().double().numpy(), base.cpu().double().numpy()


def extract_coefficients(field, adata, timestep, batch_size):
    a = np.empty((adata.n_obs, field.K), dtype=np.float64)
    device = next(field.parameters()).device
    with torch.inference_mode():
        for start in range(0, adata.n_obs, batch_size):
            block = adata.X[start:start + batch_size]
            block = block.toarray() if hasattr(block, 'toarray') else np.asarray(block)
            x = torch.as_tensor(block, dtype=torch.float32, device=device)
            t = torch.full((len(x),), timestep, device=device)
            a[start:start + len(x)] = field._coeffs(x, t).cpu().double().numpy()
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite coefficients')
    return a
