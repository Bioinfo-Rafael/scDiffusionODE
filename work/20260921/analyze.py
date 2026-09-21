#!/usr/bin/env python3
"""Read-only native DDPM diagnostics for the canonical x0-prediction Stage1 EMA."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import model_io


def integer_list(value):
    return sorted(set(int(x) for x in value.split(',')), reverse=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', help='Stage1 campaign name or absolute campaign directory; ambiguous discovery fails')
    p.add_argument('--stage1-root', type=Path, default=model_io.STAGE1)
    p.add_argument('--data-path', help='Relocated original h5ad; SHA256 must match checkpoint')
    p.add_argument('--dry-run', action='store_true', help='Only discover and validate checkpoint/config; no dataset loading or inference')
    p.add_argument('--output', type=Path, help='New output directory (must not already exist)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--sample-count', type=int, default=3000)
    p.add_argument('--sampling-batch', type=int, default=50)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--pca-components', type=int, default=50)
    p.add_argument('--local-knn', type=int, default=50)
    p.add_argument('--tangent-dimensions', type=integer_list, default=[5, 10, 20])
    p.add_argument('--mode-knn', type=int, default=15)
    p.add_argument('--mixing-knn', type=int, default=30)
    p.add_argument('--swd-projections', type=int, default=256)
    p.add_argument('--near-zero-threshold', type=float, default=0.05)
    p.add_argument('--geometry-timesteps', type=integer_list, default=[0, 1, 5, 10, 20, 50, 100, 200])
    p.add_argument('--snapshot-timesteps', type=integer_list, default=[999, 900, 700, 500, 300, 200, 100, 50, 20, 10, 5, 1, 0])
    p.add_argument('--cpu-threads', type=int, default=4)
    return p


def write_csv(path, rows, columns=None):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample(model, diffusion, real, pca, args, output, timesteps, max_state_norm=1e6):
    import numpy as np
    import torch
    n, genes = args.sample_count, real.shape[1]
    shape = (len(timesteps), n, genes)
    arrays = {}
    def allocate(name, shape, dtype='float32'):
        a = np.lib.format.open_memmap(output / (name + '.npy'), mode='w+', dtype=dtype, shape=shape)
        arrays[name] = a
        return a
    states = allocate('reverse_state', shape)
    predictions = allocate('pred_xstart', shape)
    final = allocate('final_generated', (n, genes))
    projected = {key: allocate(key+'_pca', (len(timesteps), n, args.pca_components), 'float64')
                 for key in ('state', 'score', 'model_drift', 'noise', 'total')}
    component = np.asarray(pca.components_, dtype=np.float64)
    mean = np.asarray(pca.mean_, dtype=np.float64)
    slots = {t: i for i, t in enumerate(timesteps)}
    with torch.no_grad():
        for start in range(0, n, args.sampling_batch):
            stop = min(start + args.sampling_batch, n)
            sl = slice(start, stop)
            x = torch.randn((stop-start, genes), device=args.device)
            for timestep in reversed(range(diffusion.num_timesteps)):
                t = torch.full((len(x),), timestep, device=args.device, dtype=torch.long)
                # Native sampler calls p_mean_variance internally. Extra direct call at
                # snapshots only: deterministic eval mode; no RNG or sampler modification.
                stats = diffusion.p_mean_variance(model, x, t, clip_denoised=False) if timestep in slots else None
                out = diffusion.p_sample(model, x, t, clip_denoised=False)
                if not torch.isfinite(out['sample']).all() or not torch.isfinite(out['pred_xstart']).all():
                    raise FloatingPointError(f'Nonfinite sampler output at t={timestep}, batch={start}')
                if out['sample'].double().norm(dim=1).max().item() > max_state_norm:
                    raise FloatingPointError(f'Stage1 state-norm limit exceeded at t={timestep}')
                if timestep in slots:
                    assert torch.equal(stats['pred_xstart'], out['pred_xstart'])
                    # Confirm native START_X means raw CellUNet output (no clipping/conversion).
                    raw = model(x, t.unsqueeze(1))
                    assert torch.allclose(raw, out['pred_xstart'], rtol=1e-5, atol=1e-6)
                    j = slots[timestep]
                    before = x.cpu().numpy().astype(np.float64)
                    pred = out['pred_xstart'].cpu().numpy().astype(np.float64)
                    after = out['sample'].cpu().numpy().astype(np.float64)
                    mu = stats['mean'].cpu().numpy().astype(np.float64)
                    alpha = diffusion.alphas_cumprod[timestep]
                    # Tweedie START_X score in GENE space first; only then project C*s.
                    score = (np.sqrt(alpha)*pred - before) / (1-alpha)
                    states[j, sl], predictions[j, sl] = before, pred
                    projected['state'][j, sl] = (before-mean) @ component.T
                    for key, value in dict(score=score, model_drift=mu-before,
                                           noise=after-mu, total=after-before).items():
                        projected[key][j, sl] = value @ component.T
                    np.testing.assert_allclose(projected['total'][j, sl],
                        projected['model_drift'][j, sl]+projected['noise'][j, sl], rtol=1e-8, atol=1e-10)
                    if timestep == 0:
                        assert torch.equal(out['sample'], stats['mean'])
                x = out['sample']
            final[sl] = x.cpu().numpy()
            for array in arrays.values():
                array.flush()
            print(f'SAMPLING {stop}/{n} trajectories; completed all 1000 native DDPM steps', flush=True)
    return states, predictions, final, projected


def compute(model, diffusion, meta, args, output, run_meta):
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.decomposition import PCA
    from geometry import LocalGeometry
    from diagnostics import directions, exposure, mode_occupancy, mixing, make_figures
    real, obs, data_info = model_io.load_real(meta, args.data_path)
    run_meta.update(data_info)
    if args.pca_components > min(len(real)-1, real.shape[1]):
        raise ValueError('PCA dimension exceeds data rank bound; reduce --pca-components')
    if args.local_knn >= len(real) or args.mode_knn > len(real):
        raise ValueError('Not enough real cells for requested kNN sizes')
    if args.mixing_knn >= 2*min(len(real), args.sample_count):
        raise ValueError('Not enough balanced real/generated cells for mixing k')
    print(f'DATA verified SHA256 + gene order: {real.shape}; fitting real-only PCA', flush=True)
    pca = PCA(n_components=args.pca_components, whiten=False, svd_solver='randomized', random_state=args.seed)
    pca.fit(real)
    component = np.asarray(pca.components_, dtype=np.float64)
    mean = np.asarray(pca.mean_, dtype=np.float64)
    # Avoid a full float64 copy of the gene matrix; transform in batches.
    def transform(matrix):
        result = np.empty((len(matrix), args.pca_components), dtype=np.float64)
        for start in range(0, len(matrix), args.sampling_batch):
            sl = slice(start, start+args.sampling_batch)
            result[sl] = (np.asarray(matrix[sl], dtype=np.float64)-mean) @ component.T
        return result
    real_z = transform(real)
    np.save(output / 'real_pca.npy', real_z)
    np.savez(output / 'pca_basis.npz', components=component, mean=mean,
             explained_variance=pca.explained_variance_, explained_variance_ratio=pca.explained_variance_ratio_)
    model_io.write_json(output / 'pca_metadata.json', dict(n_components=args.pca_components,
        fit_population='all real cells only', n_real=len(real), n_genes=real.shape[1],
        whiten=False, solver='randomized', seed=args.seed, coordinate='z=C(x-mu); row convention (x-mu)@C.T',
        scaled_coordinate='z_scaled=a*z+(a-1)*(C@mu), a=sqrt(alpha_bar_t)',
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        total_explained_fraction=float(pca.explained_variance_ratio_.sum())))
    labels = {c: obs[c].astype('string').fillna('<missing>').astype(str).to_numpy()
              for c in ('Superclass', 'celltype') if c in obs.columns}
    run_meta['annotations_used'] = list(labels)
    run_meta['annotations_missing'] = [c for c in ('Superclass', 'celltype') if c not in labels]
    pd.DataFrame({'real_id': np.arange(len(real)), 'obs_name': obs.index.astype(str), **labels}).to_csv(output/'real_cells.csv', index=False)
    timesteps = sorted(set(args.snapshot_timesteps) | set(args.geometry_timesteps), reverse=True)
    run_meta['saved_timesteps'] = timesteps
    run_meta['array_semantics'] = {'reverse_state.npy': '[snapshot, trajectory, gene], INPUT state at t',
        'pred_xstart.npy': '[snapshot, trajectory, gene], clean prediction at the SAME input state',
        'final_generated.npy': '[trajectory, gene], output AFTER t=0 update; diagnostic timestep=-1',
        '*_pca.npy': 'fixed real PCA coordinates or projected vectors; timesteps match saved_timesteps'}
    run_meta['status'] = 'sampling'
    model_io.write_json(output / 'metadata.json', run_meta)
    _, predictions, final, projected = sample(model, diffusion, real, pca, args, output, timesteps,
                                              meta['effective_config'].get('max_state_norm', 1e6))
    geometry = LocalGeometry(real_z, component @ mean, args.local_knn, args.tangent_dimensions)
    ids = np.arange(args.sample_count)
    with (output / 'tangent_normal.csv').open('w', newline='') as handle:
        writer = None
        for j, t in enumerate(timesteps):
            if t not in args.geometry_timesteps:
                continue
            vectors = {key: projected[key][j] for key in ('score', 'model_drift', 'noise', 'total')}
            rows = geometry.decompose(projected['state'][j], vectors,
                                      np.sqrt(diffusion.alphas_cumprod[t]), t, ids)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            print(f'GEOMETRY t={t} cached_anchors={len(geometry.cache)}', flush=True)
    geometry_table = pd.read_csv(output / 'tangent_normal.csv')
    geometry_table[geometry_table.t > 200].to_csv(output / 'tangent_normal_high_noise_reference.csv', index=False)
    run_meta['cached_anchor_count'] = len(geometry.cache)
    projections = directions(args.pca_components, args.swd_projections, args.seed+3)
    np.save(output / 'swd_directions.npy', projections)
    exposure_rows, occupancy_rows, mode_summaries, assignment_rows = [], [], [], []
    mixing_rows, mixing_points = [], []
    def clean_diagnostics(z, t):
        rows, summaries, assignments = mode_occupancy(real_z, z, labels, t, args.mode_knn)
        occupancy_rows.extend(rows)
        mode_summaries.extend(summaries)
        assignment_rows.extend(assignments)
        mix, subset, fractions = mixing(real_z, z, args.mixing_knn, args.seed+4, args.near_zero_threshold)
        mixing_rows.append(dict(timestep=t, **mix))
        mixing_points.extend(dict(timestep=t, trajectory_id=int(i), real_neighbor_fraction=float(f))
                             for i, f in zip(subset, fractions))
    for j, t in enumerate(timesteps):
        # Same t state MARGINALS. Independent CPU RNG never perturbs reverse sampling.
        q_samples = []
        for repeat in range(2):
            forward_seed = args.seed + 10000 + 2*t + repeat
            rng = np.random.default_rng(forward_seed)
            real_ids = rng.choice(len(real), args.sample_count, replace=True)
            generator = torch.Generator(device='cpu').manual_seed(forward_seed)
            qz = np.empty_like(projected['state'][j])
            with torch.no_grad():
                for start in range(0, args.sample_count, args.sampling_batch):
                    sl = slice(start, min(start+args.sampling_batch, args.sample_count))
                    x0 = torch.from_numpy(real[real_ids[sl]])
                    times = torch.full((len(x0),), t, dtype=torch.long)
                    noise = torch.randn(x0.shape, generator=generator)
                    qt = diffusion.q_sample(x0, times, noise=noise)
                    qz[sl] = transform(qt.numpy())
            np.save(output / f'forward_real_ids_t{t}_{repeat}.npy', real_ids)
            np.save(output / f'forward_pca_t{t}_{repeat}.npy', qz)
            q_samples.append(qz)
        values = exposure(q_samples[0], projected['state'][j], projections)
        baseline = exposure(q_samples[0], q_samples[1], projections)
        exposure_rows.append(dict(t=t, n=args.sample_count, **values,
                                  swd_forward_baseline=baseline['swd']))
        pred_z = transform(predictions[j])
        np.save(output / f'pred_xstart_pca_t{t}.npy', pred_z)
        clean_diagnostics(pred_z, t)
        print(f'DIAGNOSTICS t={t} SWD={values["swd"]:.6g} forward_baseline={baseline["swd"]:.6g}', flush=True)
    final_z = transform(final)
    np.save(output / 'final_generated_pca.npy', final_z)
    clean_diagnostics(final_z, -1)
    write_csv(output / 'exposure.csv', exposure_rows)
    write_csv(output / 'mode_occupancy.csv', occupancy_rows,
              ['annotation', 'label', 'real_fraction', 'generated_fraction', 'difference', 'timestep'])
    write_csv(output / 'mode_summary.csv', mode_summaries,
              ['annotation', 'timestep', 'total_variation_distance', 'jensen_shannon_divergence'])
    write_csv(output / 'mode_assignments.csv', assignment_rows,
              ['annotation', 'timestep', 'trajectory_id', 'label', 'vote_fraction'])
    write_csv(output / 'mixing.csv', mixing_rows)
    write_csv(output / 'mixing_points.csv', mixing_points)
    print('FIGURES generating eleven overview figures', flush=True)
    make_figures(output)


def main(argv=None):
    args = parser().parse_args(argv)
    payload, provenance = model_io.discover(args.campaign, args.stage1_root)
    if args.dry_run:
        print(json.dumps(provenance, indent=2, ensure_ascii=False))
        print('DRY_RUN_OK: checkpoint/campaign validated; dataset not read; no inference or updates')
        return
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    for key in ('sample_count', 'sampling_batch', 'pca_components', 'local_knn', 'mode_knn',
                'mixing_knn', 'swd_projections', 'cpu_threads'):
        if getattr(args, key) < 1:
            raise ValueError(f'--{key.replace("_", "-")} must be positive')
    if args.sample_count < 2 or not 0 <= args.near_zero_threshold <= 1:
        raise ValueError('Need at least two samples and a near-zero threshold in [0,1]')
    if not all(0 <= t <= 999 for t in args.geometry_timesteps + args.snapshot_timesteps):
        raise ValueError('Timesteps must be in 0..999')
    if not any(t <= 200 for t in args.geometry_timesteps):
        raise ValueError('Include at least one low-noise geometry timestep for main figures')
    if not (1 <= min(args.tangent_dimensions) <= max(args.tangent_dimensions)
            <= min(args.local_knn-1, args.pca_components)):
        raise ValueError('Invalid tangent dimensions for requested local k/PCA')
    meta = payload['metadata']
    args.seed = args.seed if args.seed is not None else meta['effective_config'].get('seed', meta.get('seed', 1234))
    if args.seed is None:
        args.seed = 1234
    if not 0 <= args.seed < 2**32-12000:
        raise ValueError('Seed must be in [0, 2**32-12000)')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    model, diffusion = model_io.restore(payload, args.device)
    del payload
    output = (args.output or HERE / 'results' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])).resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f'RUN_DIRECTORY={output}', flush=True)
    run_meta = dict(status='initializing', created_utc=datetime.now(timezone.utc).isoformat(),
        run_directory=str(output), cli={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        seed=args.seed, effective_config=meta['effective_config'], checkpoint=provenance['checkpoint'],
        checkpoint_sha256=provenance['checkpoint_sha256_before'], gene_order_hash=meta['gene_order_hash'],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        analysis_source_sha256={p.name: model_io.file_hash(p) for p in HERE.glob('*.py')},
        versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'torch', 'anndata', 'pandas', 'matplotlib')},
        sampling_verification=dict(model_mean_type=diffusion.model_mean_type.name,
            model_var_type=diffusion.model_var_type.name, loss_type=diffusion.loss_type.name,
            native_ancestral_ddpm=True, steps=diffusion.num_timesteps, noise_schedule='linear',
            timestep_respacing='', clip_denoised=False, nw=0.5, cond_fn=None,
            parameter_updates=False, beta_sha256=__import__('hashlib').sha256(diffusion.betas.tobytes()).hexdigest()),
        forward_rng_policy='independent CPU streams: seed+10000+2*t+repeat (repeat=0,1)',
        swd_definition='mean absolute sorted projection difference (sliced W1); fixed unit directions seed+3',
        mixing_rng_policy='balanced subsets with fixed seed+4 at every snapshot; identity self-exclusion',
        model_eval=True, torch_no_grad=True)
    model_io.write_json(output / 'checkpoint_provenance.json', provenance)
    model_io.write_json(output / 'metadata.json', run_meta)
    try:
        with threadpool_limits(limits=args.cpu_threads), torch.no_grad():
            compute(model, diffusion, meta, args, output, run_meta)
        run_meta['status'] = 'completed'
    except BaseException as exc:
        run_meta.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        provenance['checkpoint_sha256_after'] = model_io.file_hash(provenance['checkpoint'])
        provenance['model_hash_after'] = model_io.state_hash(model)
        unchanged = (provenance['checkpoint_sha256_before'] == provenance['checkpoint_sha256_after']
                     and provenance['model_hash_before'] == provenance['model_hash_after'])
        provenance['unchanged_assertion_passed'] = unchanged
        provenance['all_parameters_frozen'] = all(not p.requires_grad for p in model.parameters())
        provenance['all_modules_eval'] = all(not m.training for m in model.modules())
        if not unchanged or not provenance['all_parameters_frozen'] or not provenance['all_modules_eval']:
            run_meta['status'] = 'failed_integrity_check'
        run_meta['finished_utc'] = datetime.now(timezone.utc).isoformat()
        model_io.write_json(output / 'checkpoint_provenance.json', provenance)
        model_io.write_json(output / 'metadata.json', run_meta)
        assert unchanged, 'Checkpoint or CellUNet parameters/buffers changed during analysis'
        assert provenance['all_parameters_frozen'] and provenance['all_modules_eval']
    print(f'COMPLETED {output}; checkpoint and model hashes unchanged', flush=True)


if __name__ == '__main__':
    main()
