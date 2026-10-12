"""Read every committed JSONL loss row, retaining raw and weighted semantics."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

RAW = ('diffusion_loss', 'ode_offmask_base_raw', 'cell_ode_consistency_raw_20260830')
WEIGHTED = ('diffusion_loss', 'ode_regularization_final_weighted',
            'cell_ode_consistency_final_weighted_20260830')
NAMES = ('Reconstruction', 'GRN off-mask', 'CellUNet-ODE consistency')


def smooth(frame, window):
    """Trailing optimizer-step window, including sparse logs without stretching it."""
    if window < 1:
        raise ValueError('rolling window must be positive')
    indexed = frame.copy()
    indexed.index = pd.to_timedelta(frame['step'].to_numpy(), unit='s')
    columns = list(dict.fromkeys((*RAW, *WEIGHTED, 'total_loss')))
    roll = indexed[columns].rolling(pd.Timedelta(seconds=window), min_periods=1)
    mean, std = roll.mean(), roll.std(ddof=0)
    mean.index = frame.index
    std.index = frame.index
    mean.insert(0, 'step', frame['step'])
    std.insert(0, 'step', frame['step'])
    return mean, std


def fractions(history, rolling_mean):
    out = pd.DataFrame({'step': history['step']})
    for suffix, frame in (('', history), ('_rolling', rolling_mean)):
        denominator = frame[list(WEIGHTED)].sum(axis=1)
        denominator = denominator.where(denominator > 0, np.nan)
        for name, column in zip(('reconstruction', 'grn', 'consistency'), WEIGHTED):
            out[name+suffix] = frame[column] / denominator
        out['total'+suffix] = denominator
    return out


def load(run, window=100):
    rows = []
    paths = sorted(p for p in (Path(run)/'logs').glob('loss_*.jsonl') if '.rolled_back.' not in p.name)
    if not paths:
        raise FileNotFoundError(f'No loss JSONL files in {Path(run)/"logs"}')
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{number}: incomplete/invalid log row') from error
            row['loss_source'] = str(path.resolve())
            rows.append(row)
    history = pd.DataFrame(rows)
    required = list(dict.fromkeys(('step', *RAW, *WEIGHTED, 'total_loss')))
    missing = set(required)-set(history)
    if missing:
        raise ValueError(f'Missing loss columns: {sorted(missing)}')
    history[required] = history[required].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(history[required].to_numpy()).all():
        raise ValueError('Nonfinite loss values')
    if (history['step'] < 1).any() or (history['step'] % 1 != 0).any():
        raise ValueError('Invalid optimizer step')
    # Resume archives are excluded; conflicting overlapping live segments are ambiguous.
    duplicates = history[history.duplicated('step', keep=False)]
    for step, group in duplicates.groupby('step'):
        if not np.allclose(group[required].to_numpy(), group[required].iloc[0].to_numpy(), rtol=1e-10, atol=1e-12):
            raise ValueError(f'Conflicting live loss rows at optimizer step {step}')
    history = history.sort_values('step').drop_duplicates('step').reset_index(drop=True)
    if (history[list(dict.fromkeys((*RAW, *WEIGHTED, 'total_loss')))] < 0).any().any():
        raise ValueError('Negative MSE/regularization loss')
    if not np.allclose(history[list(WEIGHTED)].sum(axis=1), history['total_loss'], rtol=1e-5, atol=1e-7):
        raise ValueError('Saved total_loss does not equal the three weighted contributions')
    mean, std = smooth(history, window)
    return history, mean, std, fractions(history, mean)
