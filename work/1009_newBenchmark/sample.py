"""Original 1000-step ancestral p_sample, Gaussian start, EMA weights."""
from common import *
import anndata as ad
import numpy as np
import pandas as pd
import torch


def sample(c, *, checkpoint=None, target_device='cpu', count=None):
    run = paths(c); cfg = c['training']; count = cfg['num_samples'] if count is None else count
    require(count > 0, 'sample count must be positive')
    model, ck, a, path = load_checkpoint(c, checkpoint, device(target_device))
    out = run / 'samples/generated.h5ad'
    stamp = dict(checkpoint_sha256=sha256(path), count=count, weights='ema',
                 seed=cfg['seed'], gene_ids=ck['gene_ids'], steps=1000, nw=0.5,
                 clip_denoised=False, batch_size=cfg['sample_batch_size'])
    if out.exists():
        old = json.loads((run / 'samples/metadata.json').read_text())
        require(all(old.get(k) == v for k, v in stamp.items()), 'existing samples have different provenance')
        require(old['output_sha256'] == sha256(out), 'samples changed')
        return out
    SOURCE.seed_all(cfg['seed']); d = diffusion(c); dev = next(model.parameters()).device
    samples = np.empty((count, a.n_vars), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, count, cfg['sample_batch_size']):
            n = min(cfg['sample_batch_size'], count-start)
            x = torch.randn(n, a.n_vars, device=dev)
            for t_index in reversed(range(1000)):
                t = torch.full((n,), t_index, device=dev, dtype=torch.long)
                x = d.p_sample(model, x, t, clip_denoised=False)['sample']
                SOURCE.finite('ancestral state', x)
            samples[start:start+n] = x.cpu().float().numpy()
            print(f'Generated {start+n}/{count}', flush=True)
    result = ad.AnnData(X=samples, obs=pd.DataFrame(index=[f'generated_{i}' for i in range(count)]), var=a.var.copy())
    result.uns['sampling'] = dict(method='ancestral', prediction='x_start', checkpoint=str(path), weights='ema')
    write_h5ad(result, out)
    write_json(run / 'samples/metadata.json', {**stamp, 'output_sha256': sha256(out)})
    return out


def main():
    p = parser(__doc__); device_arg(p); p.add_argument('--checkpoint')
    p.add_argument('--count', type=int, help='Smoke override; production default 3000')
    args = p.parse_args()
    print(sample(config(args.config), checkpoint=args.checkpoint, target_device=args.device, count=args.count))

if __name__ == '__main__':
    cli(main)
