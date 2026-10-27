"""Read-only original D20 loader and a fixed-timestep ODE callable."""
from pathlib import Path
import importlib
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def as_batch(x, dimension):
    x = np.asarray(x, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != dimension or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError(f'Expected finite (D,) or nonempty (N,D), D={dimension}')
    return x, single


def evaluate_velocity(velocity, x, batch_size=256):
    result = np.empty_like(x, dtype=np.float64)
    for start in range(0, len(x), batch_size):
        block = x[start:start + batch_size]
        v = np.asarray(velocity(block), dtype=np.float64)
        if v.shape != block.shape or not np.isfinite(v).all():
            raise ValueError('Velocity must return finite values with the input shape')
        result[start:start + len(block)] = v
    return result


class ODEVelocity:
    def __init__(self, field, feature_ids, provenance, timestep=49, batch_size=256):
        self.field = field
        self.feature_ids = list(feature_ids)
        self.timestep = int(timestep)
        self.batch_size = batch_size
        self.provenance = dict(provenance, diffusion_timestep=self.timestep, branch='ODE only')

    def __call__(self, x):
        import torch
        x, single = as_batch(x, len(self.feature_ids))
        if (x < 0).any():
            raise ValueError('D20 requires nonnegative, linearly normalized expression')
        device = next(self.field.parameters()).device
        result = np.empty_like(x)
        with torch.inference_mode():
            for start in range(0, len(x), self.batch_size):
                block = torch.as_tensor(x[start:start+self.batch_size], dtype=torch.float32, device=device)
                t = torch.full((len(block),), self.timestep, dtype=torch.long, device=device)
                result[start:start+len(block)] = self.field(block, t).cpu().double().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite ODE output')
        return result[0] if single else result


def load_d20(config, device='cpu'):
    """Only the existing 10/27 loader builds/restores the ODE; no CellUNet or training."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Existing load_model imports its sibling artifacts by an absolute name.
    sibling = str(ROOT / 'work/20261027_matsum_lora_analysis')
    sys.path.insert(0, sibling)
    try:
        loader = importlib.import_module('work.20261027_matsum_lora_analysis.load_model')
        field, adata, stamp = loader.load_trained_model('lora', config, device)
    finally:
        sys.path.remove(sibling)
    if adata.n_vars != 1024:
        raise ValueError('D20 expects the ordered 1024 training HVGs')
    velocity = ODEVelocity(field, adata.var_names, stamp, config.get('timestep', 49))
    return velocity, adata
