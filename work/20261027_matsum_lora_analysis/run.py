#!/usr/bin/env python3
"""Entry point: load -> coefficients -> shared bases -> cell-dependent matrices. No training."""
import argparse
from datetime import datetime, timezone, timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

# Set before importing NumPy/Torch/UMAP. Avoid machine-dependent BLAS oversubscription.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(name, '1')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from artifacts import log, sha256, write_json


def smoke_fixture(kind, cfg, device):
    """Artificial data + original field classes; only for execution/numerical QA."""
    import numpy as np
    import torch
    import anndata as ad
    import pandas as pd
    from ODE.ode_20260609_mathmlp import MatSumField, LoRAField
    rng = np.random.default_rng(cfg['seed'])
    labels = np.repeat(['fixture_A', 'fixture_B', 'fixture_C'], 16)
    x = rng.gamma(2, 1, size=(len(labels), 16)).astype(np.float32)
    for i in range(3): x[i * 16:(i + 1) * 16, i * 4:(i + 1) * 4] += 2
    obs = pd.DataFrame({'celltype': labels, 'stage': np.tile(['E7.0', 'E8.0'], 24),
                        'sample': np.tile(['s1', 's2', 's3', 's4'], 12)}, index=[f'fixture_cell_{i}' for i in range(len(x))])
    adata = ad.AnnData(x, obs=obs, var=pd.DataFrame(index=[f'gene_{i}' for i in range(x.shape[1])]))
    adata.obsm['X_umap_source'] = rng.normal(size=(len(x), 2))
    torch.manual_seed(cfg['seed'])
    cls = MatSumField if kind == 'matsum' else LoRAField
    field = cls(16, K=8, time_dim=8, hidden=16, **({'rank': 3} if kind == 'lora' else {}))
    field.to(device).eval().requires_grad_(False)
    return field, adata, {'fixture': True, 'warning': 'ARTIFICIAL DATA / RANDOM WEIGHTS; NOT TRAINED D16/D20'}


def verify_reconstruction(field, adata, a, basis, base, timestep):
    """Independent exported-factor reconstruction must match original forward/compute_W."""
    import numpy as np
    import torch
    block = adata.X[:4]; block = block.toarray() if hasattr(block, 'toarray') else np.asarray(block)
    device = next(field.parameters()).device
    x = torch.as_tensor(block, dtype=torch.float32, device=device)
    with torch.inference_mode():
        w = base + np.einsum('bk,kij->bij', a[:len(x)], basis)
        original_w = field.compute_W(x, timestep).cpu().numpy()
        np.testing.assert_allclose(w, original_w, rtol=3e-5, atol=3e-6)
        z = np.einsum('bij,bj->bi', w, block) + field.b.detach().cpu().numpy()
        gamma = torch.nn.functional.softplus(field.gamma).cpu().numpy()
        reconstructed = np.logaddexp(0, z) - gamma * block
        original = field(x, timestep).cpu().numpy()
        np.testing.assert_allclose(reconstructed, original, rtol=5e-5, atol=5e-5)
    return {'cells_checked': len(x), 'max_absolute_forward_error': float(np.max(np.abs(reconstructed - original)))}


def analyze_model(kind, field, adata, stamp, cfg, output):
    """The whole scientific execution order is intentionally visible in one function."""
    log('Loading plotting and analysis libraries (Matplotlib / scikit-learn)')
    from analyze_coefficients import run_coefficient_analysis
    from analyze_basis import run_basis_analysis
    from analyze_cells import run_cell_analysis
    from load_model import shared_matrices, extract_coefficients
    log('Plotting and analysis libraries ready')
    folder = output / kind; folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'model_provenance.json', stamp)
    adata.var.to_csv(folder / 'genes.csv', index_label='gene_id')
    basis, base = shared_matrices(field)

    log(f'{kind}: 8 shared bases')
    zoom = run_basis_analysis(basis, base, adata.var_names, folder / 'basis', cfg)

    for timestep in cfg['timesteps']:
        step_folder = folder / f't{timestep:03d}'; step_folder.mkdir(parents=True, exist_ok=True)
        log(f'{kind}: extracting a(x,t={timestep}) for {adata.n_obs} cells')
        a = extract_coefficients(field, adata, timestep, cfg['coefficient_batch_size'])
        check = verify_reconstruction(field, adata, a, basis, base, timestep)
        write_json(step_folder / 'forward_reconstruction_check.json', check)

        log(f'{kind}, t={timestep}: coefficient statistics / plots / clustering')
        run_coefficient_analysis(a, adata, step_folder / 'coefficients', cfg)

        log(f'{kind}, t={timestep}: cell-dependent matrices / gene rankings / graph spectra')
        run_cell_analysis(a, adata, basis, base, field, zoom, folder / 'basis', step_folder / 'cell_matrices', cfg)


def load_config(args):
    cfg = json.loads(args.config.read_text())
    expected_keys = set(json.loads((HERE / 'config.json').read_text()))
    if set(cfg) != expected_keys:
        raise ValueError(f'Config keys differ: missing={expected_keys - set(cfg)}, unknown={set(cfg) - expected_keys}')
    if args.run_root: cfg['run_root'] = str(args.run_root)
    if args.data_dir: cfg['data_dir'] = str(args.data_dir)
    if args.models: cfg['models'] = args.models
    if args.weights: cfg['weights'] = args.weights
    if args.smoke:
        cfg.update(spectral_dimensions=4, spectral_cells_per_type=1, heatmap_top_genes=8,
                   scatter_cells_per_type=10, cluster_counts=[3, 5], cluster_seeds=[0, 1],
                   silhouette_sample_size=48, umap_neighbors=8, histogram_bins=20)
    if not cfg['models'] or any(k not in ('matsum', 'lora') for k in cfg['models']):
        raise ValueError('models must contain matsum and/or lora')
    if len(set(cfg['models'])) != len(cfg['models']) or len(set(cfg['timesteps'])) != len(cfg['timesteps']):
        raise ValueError('Duplicate models/timesteps would overwrite outputs')
    if cfg['weights'] not in ('ema', 'raw'):
        raise ValueError('weights must be ema or raw')
    if not cfg['timesteps'] or any(type(t) is not int or not 0 <= t < 1000 for t in cfg['timesteps']):
        raise ValueError('timesteps must be integer diffusion indices from 0 to 999')
    for key in ('coefficient_batch_size', 'matrix_batch_size', 'spectral_dimensions', 'spectral_cells_per_type',
                'heatmap_top_genes', 'scatter_cells_per_type', 'histogram_bins', 'silhouette_sample_size', 'umap_neighbors'):
        if type(cfg[key]) is not int or cfg[key] < 1: raise ValueError(f'{key} must be a positive integer')
    if cfg['umap_neighbors'] < 2 or not cfg['cluster_seeds']:
        raise ValueError('UMAP needs >=2 neighbors and clustering needs at least one seed')
    if cfg['near_zero_mean_relative'] < 0 or cfg['near_zero_mean_absolute'] < 0:
        raise ValueError('Near-zero thresholds must be nonnegative')
    return cfg


class Tee:
    """Record the same readable progress displayed on the terminal."""
    def __init__(self, terminal, file): self.terminal, self.file = terminal, file
    def write(self, text): self.terminal.write(text); self.file.write(text)
    def flush(self): self.terminal.flush(); self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--run-root', type=Path, help='Directory containing trained D16 and D20 runs')
    parser.add_argument('--data-dir', type=Path, help='Existing model_comparison manifest directory')
    parser.add_argument('--models', nargs='+', choices=['matsum', 'lora'])
    parser.add_argument('--weights', choices=['ema', 'raw'])
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--output', type=Path, help='New/empty output directory; never overwrite results')
    parser.add_argument('--check-only', action='store_true', help='Validate trained data/checkpoints and reconstruct four cells, then stop')
    parser.add_argument('--smoke', action='store_true', help='Complete pipeline on artificial fixtures; does not read trained runs')
    args = parser.parse_args()
    cfg = load_config(args)
    log('Starting analysis entry point; loading NumPy')
    import numpy as np
    log('Loading PyTorch')
    import torch
    log('Loading checkpoint/data adapter')
    from load_model import load_trained_model, shared_matrices, extract_coefficients
    log('Core libraries ready')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mode = 'smoke' if args.smoke else ('check' if args.check_only else 'full')
    timestamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    output = (args.output or HERE / 'outputs' / f'{mode}_{cfg["weights"]}_{timestamp}').resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output is nonempty: {output}; use a new directory')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'resolved_config.json', cfg)
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'scipy', 'pandas', 'torch', 'matplotlib', 'scikit-learn', 'umap-learn', 'anndata', 'h5py')}
    manifest = dict(status='running', mode=mode, device=device, versions=versions, command=sys.argv,
                    analysis_sources={p.name: sha256(p) for p in sorted(HERE.glob('*.py'))},
                    started_jst=timestamp, config=cfg, completed_models=[])
    write_json(output / 'manifest.json', manifest)
    original_stdout = sys.stdout
    try:
        with (output / 'progress.log').open('w') as logfile:
            sys.stdout = Tee(original_stdout, logfile)
            previous_ids = None
            for kind in cfg['models']:
                log(f'Loading {kind}; mode={mode}, device={device}')
                loader = smoke_fixture if args.smoke else load_trained_model
                field, adata, stamp = loader(kind, cfg, device)
                ordered_ids = (adata.obs_names.tolist(), adata.var_names.tolist())
                if previous_ids is not None and ordered_ids != previous_ids:
                    raise ValueError('D16/D20 cell/gene identities or ordering differ')
                previous_ids = ordered_ids
                if args.check_only:
                    basis, base = shared_matrices(field)
                    checks = []
                    for t in cfg['timesteps']:
                        probe = adata[:4]
                        a = extract_coefficients(field, probe, t, 4)
                        checks.append(verify_reconstruction(field, probe, a, basis, base, t))
                    write_json(output / f'{kind}_preflight.json', dict(provenance=stamp, checks=checks))
                else:
                    analyze_model(kind, field, adata, stamp, cfg, output)
                manifest['completed_models'].append(kind)
                write_json(output / 'manifest.json', manifest)
                del field, adata
            manifest['status'] = 'complete'
            log(f'Complete: {output}')
    except KeyboardInterrupt:
        manifest.update(status='interrupted', error='KeyboardInterrupt')
        (output / 'error.txt').write_text(traceback.format_exc())
        raise
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        (output / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        sys.stdout = original_stdout
        write_json(output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
