"""Three-way START_X diagnostics using the 8/30 metric definitions."""
from __future__ import annotations

import importlib
import numpy as np
import pandas as pd
import torch

legacy = importlib.import_module('work.20260830.analysis.metrics')
PAIRS = {'cell_target': ('cell', 'target'), 'ode_target': ('ode', 'target'),
         'ode_cell': ('ode', 'cell')}
LABELS = {'cell': 'CellUNet', 'ode': 'ODE', 'target': 'target x0'}
EPS = 1e-12


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return dict(mean=float(finite.mean()) if len(finite) else np.nan,
                std=float(finite.std()) if len(finite) else np.nan,
                median=float(np.median(finite)) if len(finite) else np.nan,
                q25=float(np.quantile(finite, .25)) if len(finite) else np.nan,
                q75=float(np.quantile(finite, .75)) if len(finite) else np.nan,
                valid_count=len(finite), undefined_count=len(values)-len(finite))


def pair_values(left, right):
    """Rows are cells, columns are genes; undefined angles/ratios remain NaN."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError('Expected matching (cells, genes) arrays')
    a, b = left.double(), right.double()
    an, bn = a.norm(dim=1), b.norm(dim=1)
    cosine = legacy.safe_row_cosine(a, b, EPS)
    cosine[an * bn <= EPS] = torch.nan
    pearson = legacy.safe_row_pearson(a, b, EPS)
    denom = (a-a.mean(1, keepdim=True)).norm(dim=1) * (b-b.mean(1, keepdim=True)).norm(dim=1)
    pearson[denom <= EPS] = torch.nan
    ratio = an / bn.clamp_min(EPS)
    ratio[bn <= EPS] = torch.nan
    return dict(cosine=cosine, pearson=pearson, mse=(a-b).square().mean(1), norm_ratio=ratio)


def evaluate(model, diffusion, x0, timesteps, *, batch_size, seed, device, metadata):
    from guided_diffusion.gaussian_diffusion import ModelMeanType
    if diffusion.model_mean_type != ModelMeanType.START_X:
        raise ValueError('These diagnostics require START_X: target is clean x0')
    times = sorted(set(timesteps))
    if not times or times[0] < 0 or times[-1] >= diffusion.num_timesteps:
        raise ValueError('Diffusion timesteps out of range')
    x0 = torch.as_tensor(x0, dtype=torch.float32, device='cpu')
    if batch_size < 1 or x0.ndim != 2 or not len(x0) or not torch.isfinite(x0).all():
        raise ValueError('Invalid cells or batch size')
    pair_rows, norm_rows = [], []
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            for t_value in times:
                generator = torch.Generator(device='cpu').manual_seed(seed + t_value * 1_000_003)
                # Generate before batching: identical noise across conditions/checkpoints/batch sizes.
                noise = torch.randn(x0.shape, generator=generator, dtype=torch.float64)
                collected = {p: {k: [] for k in ('cosine', 'pearson', 'mse', 'norm_ratio')} for p in PAIRS}
                correlations = {p: legacy.RunningPearson() for p in PAIRS}
                norms = {k: [] for k in LABELS}
                for start in range(0, len(x0), batch_size):
                    clean = x0[start:start+batch_size].to(device)
                    t = torch.full((len(clean),), t_value, dtype=torch.long, device=device)
                    xt = diffusion.q_sample(clean, t, noise=noise[start:start+batch_size].to(device))
                    scaled_t = diffusion._scale_timesteps(t).unsqueeze(1)
                    outputs = dict(cell=model.ml_model(xt, scaled_t),
                                   ode=model.ode_model(xt, scaled_t), target=clean)
                    for name, value in outputs.items():
                        if value.shape != clean.shape or not torch.isfinite(value).all():
                            raise ValueError(f'Invalid {name} output at diffusion timestep {t_value}')
                        norms[name].extend(value.double().norm(dim=1).cpu().tolist())
                    for pair, (left, right) in PAIRS.items():
                        a, b = outputs[left], outputs[right]
                        correlations[pair].update(a, b)
                        for metric, value in pair_values(a, b).items():
                            collected[pair][metric].extend(value.cpu().tolist())
                base = dict(metadata, diffusion_timestep=t_value, analyzed_cells=len(x0), target='x_start')
                for name, values in norms.items():
                    norm_rows.append(dict(base, branch=name, **stats(values)))
                for pair, values in collected.items():
                    row = dict(base, pair=pair, numerator=PAIRS[pair][0], denominator=PAIRS[pair][1])
                    for metric, array in values.items():
                        row.update({metric+'_'+k: v for k, v in stats(array).items()})
                    corr = correlations[pair]
                    vx = max(corr.sum_x2-corr.sum_x**2/corr.n, 0.)
                    vy = max(corr.sum_y2-corr.sum_y**2/corr.n, 0.)
                    row['pearson_global'] = corr.value(EPS) if (vx*vy)**.5 > EPS else np.nan
                    pair_rows.append(row)
    finally:
        for module, mode in modes:
            module.training = mode
    return pd.DataFrame(pair_rows), pd.DataFrame(norm_rows)
