"""Export the assumed velocity V(x) = CellUNet(x, diffusion_t=49)."""
from common import *
import anndata as ad
import numpy as np
import torch


def export(c, *, checkpoint=None, target_device='cpu'):
    run = paths(c)
    model, ck, a, path = load_checkpoint(c, checkpoint, device(target_device))
    d = diffusion(c); dev = next(model.parameters()).device; t_index = c['field_timestep']
    require(isinstance(t_index, int) and 0 <= t_index < 1000, 'field timestep must be 0..999')
    stem = f'cellunet_direct_t{t_index}'
    out = run / 'predictions' / f'{stem}.h5ad'
    metadata_path = run / 'predictions' / f'{stem}_metadata.json'
    stamp = dict(checkpoint_sha256=sha256(path), training_sha256=ck['data_sha256'],
                 timestep=t_index, batch_size=c['inference_batch_size'],
                 gene_ids_sha256=digest_ids(a.var_names), cell_ids_sha256=digest_ids(a.obs_names),
                 definition='ds_dt := CellUNet(clean_X, diffusion_t)',
                 input_policy='clean normalized spliced X; no q_sample, no subtraction',
                 model_time_direction='forward_by_experimental_assumption', weights='ema')
    if out.exists():
        old = json.loads(metadata_path.read_text())
        require(all(old.get(k) == v for k, v in stamp.items()), 'field already exists with different provenance')
        require(old['output_sha256'] == sha256(out), 'field file changed')
        return out
    # Keep the previous noisy displacement experiment untouched on disk.
    field_path = run / 'predictions' / f'{stem}.npy'
    require(not field_path.exists(), f'incomplete field array already exists: {field_path}')
    field = np.lib.format.open_memmap(field_path, mode='w+', dtype='float32', shape=a.shape)
    with torch.no_grad():
        for start in range(0, a.n_obs, c['inference_batch_size']):
            stop = min(a.n_obs, start+c['inference_batch_size'])
            x = torch.from_numpy(dense_rows(a.X[start:stop])).to(dev)
            t = torch.full((len(x),), t_index, device=dev, dtype=torch.long)
            prediction = model(x, d._scale_timesteps(t).unsqueeze(1))
            field[start:stop] = SOURCE.finite('direct CellUNet output', prediction).cpu().float().numpy()
    field.flush()
    result = ad.AnnData(X=None, obs=a.obs.copy(), var=a.var.copy())
    result.layers['velocity'] = np.asarray(field)
    result.uns['benchmark_velocity'] = {
        'definition': 'ds_dt', 'expression_scale': SCALE, 'time_direction': 'forward',
        'training_n_cells': a.n_obs, 'time_unit': 'model time (arbitrary unit)',
        'inference_description': f'Experimental assumption: ds/dt := CellUNet(clean normalized spliced X, diffusion_t={t_index}); EMA; no subtraction or q_sample',
        'checkpoint': str(path.resolve()),
    }
    result.uns['cellunet_field'] = {**stamp, 'checkpoint': str(path), 'weights': 'ema',
        'training_n_cells': a.n_obs, 'expression_scale': SCALE,
        'time_semantics': 'diffusion timestep is conditioning; ds/dt is the experimental model-time definition'}
    write_h5ad(result, out)
    write_json(metadata_path, {**stamp, 'output_sha256': sha256(out)})
    return out


def main():
    p = parser(__doc__); device_arg(p); p.add_argument('--checkpoint')
    args = p.parse_args()
    print(export(config(args.config), checkpoint=args.checkpoint, target_device=args.device))

if __name__ == '__main__':
    cli(main)
