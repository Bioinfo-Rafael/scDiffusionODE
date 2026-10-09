"""Export CellUNet predictions/displacements, never fabricate RNA velocity."""
from common import *
import anndata as ad
import numpy as np
import torch


def export(c, *, checkpoint=None, target_device='cpu'):
    run = paths(c)
    model, ck, a, path = load_checkpoint(c, checkpoint, device(target_device))
    d = diffusion(c); dev = next(model.parameters()).device; t_index = c['field_timestep']
    require(isinstance(t_index, int) and 0 <= t_index < 1000, 'field timestep must be 0..999')
    out = run / 'predictions/cellunet_field.h5ad'
    stamp = dict(checkpoint_sha256=sha256(path), training_sha256=ck['data_sha256'],
                 timestep=t_index, seed=c['training']['seed'], batch_size=c['inference_batch_size'],
                 definition='x_start_prediction_and_reconstruction_displacement', biological_velocity=False)
    if out.exists():
        old = json.loads((run / 'predictions/metadata.json').read_text())
        require(all(old.get(k) == v for k, v in stamp.items()), 'field already exists with different provenance')
        require(old['output_sha256'] == sha256(out), 'field file changed')
        return out
    SOURCE.seed_all(c['training']['seed'])
    # Memmaps avoid retaining multiple full dense cell-by-gene matrices in RAM.
    raw_path = run / 'predictions/x_start.npy'; delta_path = run / 'predictions/displacement.npy'
    raw = np.lib.format.open_memmap(raw_path, mode='w+', dtype='float32', shape=a.shape)
    delta = np.lib.format.open_memmap(delta_path, mode='w+', dtype='float32', shape=a.shape)
    with torch.no_grad():
        for start in range(0, a.n_obs, c['inference_batch_size']):
            stop = min(a.n_obs, start+c['inference_batch_size'])
            x = torch.from_numpy(dense_rows(a.X[start:stop])).to(dev)
            t = torch.full((len(x),), t_index, device=dev, dtype=torch.long)
            noise = torch.randn_like(x, dtype=torch.float64)
            noisy = d.q_sample(x, t, noise=noise)
            prediction = model(noisy, d._scale_timesteps(t).unsqueeze(1))
            raw[start:stop] = SOURCE.finite('x_start', prediction).cpu().float().numpy()
            delta[start:stop] = SOURCE.finite('displacement', prediction-x).cpu().float().numpy()
    raw.flush(); delta.flush()
    result = ad.AnnData(X=None, obs=a.obs.copy(), var=a.var.copy())
    result.layers['cellunet_x_start'] = np.asarray(raw)
    result.layers['reconstruction_displacement'] = np.asarray(delta)
    result.uns['cellunet_field'] = {**stamp, 'checkpoint': str(path), 'weights': 'ema',
        'input_policy': 'q_sample(linear_spliced_X,t,fixed_seed_float64_noise)',
        'displacement': 'CellUNet(q_sample(X,t),t) - X; no time division',
        'training_n_cells': a.n_obs, 'expression_scale': SCALE,
        'time_semantics': 'diffusion index only; no biological time/direction'}
    write_h5ad(result, out)
    write_json(run / 'predictions/metadata.json', {**stamp, 'output_sha256': sha256(out),
        'gene_ids_sha256': digest_ids(a.var_names), 'cell_ids_sha256': digest_ids(a.obs_names)})
    return out


def main():
    p = parser(__doc__); device_arg(p); p.add_argument('--checkpoint')
    args = p.parse_args()
    print(export(config(args.config), checkpoint=args.checkpoint, target_device=args.device))

if __name__ == '__main__':
    cli(main)
