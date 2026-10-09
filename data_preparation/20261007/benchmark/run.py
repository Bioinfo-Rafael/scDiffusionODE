#!/usr/bin/env python3
"""Common entry point. A training Python launches the isolated evaluation Python."""
from __future__ import annotations
import argparse
import contextlib
import json
import pickle
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
import config as C

# No scientific packages are imported in the caller's training environment.
if __name__ == '__main__':
    C.launch_isolated(__file__)

import anndata as ad
import numpy as np
import pandas as pd
import scvelo as scv
from scipy import sparse
from common import (annotations, check_environment, check_vendor, digest_ids, environment, finite, ids,
                    read_genes, reproducible, require, sha256, write_json)


def align_prediction(ref, pred, genes, velocity_key, time_key=None, *,
                     expected_full=C.FULL_CELLS, expected_ery=C.ERYTHROID_CELLS):
    require(ref.n_obs == expected_ery, f'evaluation must contain {expected_ery} cells')
    annotations(ref)
    meta = ref.uns.get('benchmark', {})
    require(meta.get('protocol') == C.PROTOCOL, 'wrong/missing prepared reference protocol')
    require(meta.get('expression_scale') == C.EXPRESSION_SCALE, 'reference expression scale mismatch')
    for key, value in [('seed', C.SEED), ('n_pcs', C.N_PCS), ('n_neighbors', C.N_NEIGHBORS)]:
        require(meta.get(key) == value, f'reference {key} differs from fixed protocol')
    source_ids = pd.Index(meta.get('source_cell_ids', []))
    ids(source_ids, 'source cell manifest')
    require(len(source_ids) == expected_full, 'source cell manifest count mismatch')
    require(set(ref.obs_names) <= set(source_ids), 'reference cells absent from source manifest')
    require(digest_ids(ref.var_names) == meta.get('source_gene_ids_sha256'), 'reference gene manifest changed')
    ids(pred.obs_names, 'prediction cell'); ids(pred.var_names, 'prediction gene'); ids(genes, 'expected gene')
    require(set(pred.obs_names) in (set(ref.obs_names), set(source_ids)),
            'prediction cell IDs must exactly match full source or complete erythroid reference')
    require(set(genes) <= set(ref.var_names), 'gene manifest contains unknown reference genes')
    require(set(pred.var_names) == set(genes), 'prediction gene IDs differ from expected gene manifest')
    require(velocity_key in pred.layers, f'missing velocity layer {velocity_key}')
    finite(pred.layers[velocity_key], 'prediction velocity')
    require(pred.layers[velocity_key].shape == pred.shape, 'velocity shape mismatch')
    contract = dict(pred.uns.get('benchmark_velocity', {}))
    require(contract.get('definition') == 'ds_dt', 'velocity contract must declare ds_dt; denoiser x0 is not ds_dt')
    require(contract.get('expression_scale') == C.EXPRESSION_SCALE, 'unsupported velocity expression scale')
    require(contract.get('time_direction') == 'forward', 'velocity time direction must be forward')
    require(contract.get('training_n_cells') == expected_full, 'model must be trained on full cells')
    for key in ('time_unit', 'inference_description', 'checkpoint'):
        require(isinstance(contract.get(key), str) and contract[key].strip(), f'velocity contract needs {key}')
    if time_key is None:
        candidates = [k for k in ('latent_time', 'model_time') if k in pred.obs]
        require(len(candidates) <= 1, 'multiple time columns: specify --time-key')
        time_key = candidates[0] if candidates else None
    if time_key is not None:
        require(time_key in pred.obs, f'missing model time column {time_key}')
        require(pd.api.types.is_numeric_dtype(pred.obs[time_key]), 'model time must be numeric')
        finite(pred.obs[time_key].to_numpy(), 'model time')
    # Labels from model files are optional; if present they must agree with the reference.
    matched = pred[ref.obs_names, genes]
    for key in ('celltype', 'stage'):
        if key in pred.obs:
            require(not matched.obs[key].isna().any(), f'prediction {key}: missing')
            require(np.array_equal(matched.obs[key].astype(str), ref.obs[key].astype(str)),
                    f'prediction {key} conflicts with reference')
    # No intersection/zero filling, gene-symbol conversion, barcode rewriting or rescaling.
    out = ad.AnnData(X=ref[:, genes].X.copy(), obs=ref.obs[['celltype', 'stage']].copy(),
                     var=pd.DataFrame(index=genes.copy()))
    out.obs['stage_day'] = annotations(ref).to_numpy()
    for key in ('spliced', 'unspliced'):
        require(key in ref.layers, f'reference {key}: missing')
        finite(ref.layers[key], f'reference {key}')
        out.layers[key] = ref[:, genes].layers[key].copy()
    velocity = matched.layers[velocity_key].astype(np.float32)
    finite(velocity, 'float32 velocity')  # catches overflow during official float32 conversion
    out.layers['candidate_velocity'] = velocity.copy()
    for key in ('X_pca', 'X_umap'):
        require(key in ref.obsm, f'reference {key}: missing')
        finite(ref.obsm[key], key)
        out.obsm[key] = ref.obsm[key].copy()
    require(out.obsm['X_pca'].shape == (expected_ery, C.N_PCS), 'PCA shape mismatch')
    require(out.obsm['X_umap'].shape == (expected_ery, 2), 'UMAP shape mismatch')
    for key in ('distances', 'connectivities'):
        require(key in ref.obsp, f'reference {key}: missing')
        finite(ref.obsp[key], key)
        out.obsp[key] = ref.obsp[key].copy()
    require('neighbors' in ref.uns, 'reference neighbors: missing')
    out.uns['neighbors'] = dict(ref.uns['neighbors'])
    out.uns['neighbors'].pop('indices', None)  # always let VeloEV compute official metric neighbors
    if time_key is not None:
        out.obs['candidate_time'] = matched.obs[time_key].to_numpy(dtype=float)
        require(out.obs.candidate_time.nunique() > 1, 'constant model time: TSC undefined')
    return out, {'time_kind': 'model_latent_time' if time_key else 'velocity_pseudotime',
                 'input_time_key': time_key, 'velocity_contract': contract,
                 'prediction_n_cells': pred.n_obs, 'evaluation_n_cells': out.n_obs,
                 'evaluation_n_genes': out.n_vars, 'gene_ids_sha256': digest_ids(genes),
                 'cell_ids_sha256': digest_ids(out.obs_names)}


def official_evaluation(adata, result_path):
    """Use unchanged upstream postprocess and metric functions, always k_fold=0."""
    check_vendor()
    from veloev.evaluation.evaluation import single_metric, calculate_cbdir
    reproducible()
    # Guard against scVelo's heuristic triggering a second library normalization.
    from scvelo.preprocessing.utils import not_yet_normalized
    require(not any(not_yet_normalized(adata.layers[k]) for k in ('spliced', 'unspliced')),
            'scVelo moments would renormalize this input; refusing to change velocity scale')
    scv.pp.moments(adata, n_neighbors=C.N_NEIGHBORS, n_pcs=C.N_PCS)
    finite(adata.layers['Ms'], 'Ms')
    # Only Ms and spliced are used downstream; avoid unnecessary dense Mu/U copies.
    del adata.layers['Mu']; del adata.layers['unspliced']
    processed = result_path / 'processed'
    processed.mkdir()
    adata.write_h5ad(processed / 'adata_run_candidate_full.h5ad', compression='gzip')
    # VPT.compute_eigen calls ARPACK without v0. A fresh process also resets its
    # internal RNG, which np.random.seed alone does not reset. No upstream patch.
    command = '''import sys, random
import numpy as np
sys.path.insert(0, sys.argv[1])
from veloev.postprocessing.postprocess import postprocess
random.seed(int(sys.argv[3])); np.random.seed(int(sys.argv[3]))
postprocess(methods=['candidate'], task='directional_temporal', k_fold=0,
            cluster_key='celltype', time_key='stage_day', basis='umap',
            result_path=sys.argv[2], n_jobs=1)
'''
    completed = subprocess.run([str(C.EVAL_PYTHON), '-c', command, str(C.VENDOR),
                                str(result_path.resolve()), str(C.SEED)],
                               env=C.subprocess_env(), capture_output=True, text=True)
    print(completed.stdout, end='')
    print(completed.stderr, end='', file=sys.stderr)
    completed.check_returncode()
    output = result_path / 'postprocess' / 'candidate_full.pkl'
    require(output.is_file(), 'official postprocess produced no output')
    # This pickle was just created by upstream in this run, never supplied by a caller.
    with output.open('rb') as f:
        post = pickle.load(f)
    n = adata.n_obs
    for key, shape in [('velocity_mat', adata.shape), ('velocity_emb', (n, 2)), ('exp_emb', (n, 2))]:
        finite(post.get(key), f'postprocess {key}')
        require(post[key].shape == shape, f'postprocess {key}: shape mismatch')
    finite(post.get('velocity_graph'), 'velocity graph')
    neighbors = post.get('neighbor_indices')
    require(isinstance(neighbors, np.ndarray) and neighbors.shape == (n, C.N_NEIGHBORS),
            'postprocess neighbor shape mismatch')
    require(np.issubdtype(neighbors.dtype, np.integer) and (neighbors >= 0).all()
            and (neighbors < n).all(), 'invalid neighbor indices')
    for key in ('cell_label', 'time_label'):
        require(isinstance(post.get(key), pd.Series) and post[key].index.equals(adata.obs_names),
                f'postprocess {key}: cell order mismatch')
    chosen_time = post.get('method_time')
    if chosen_time is None:
        chosen_time = post.get('pseudo_time')
    require(isinstance(chosen_time, pd.Series) and chosen_time.index.equals(adata.obs_names),
            'official time missing or misaligned')
    finite(chosen_time.to_numpy(), 'official inferred time')
    require(chosen_time.nunique() > 1, 'constant inferred time: TSC undefined')
    times = sorted(adata.obs.stage_day.unique().tolist())
    time_transitions = list(zip(times[:-1], times[1:]))
    # Record scores for each transition by invoking the same official function separately.
    evl = result_path / 'evaluation'
    evl.mkdir()
    transition_rows = []
    for transition in C.CELL_TYPE_TRANSITIONS:
        frame = calculate_cbdir([transition], result_path / 'postprocess', evl,
                                ['candidate'], k_fold=0, save=False)
        value = float(frame.iloc[0, 1])
        transition_rows.append({'source': transition[0], 'target': transition[1], 'CBDir': value})
    pd.DataFrame(transition_rows).to_csv(result_path / 'cbdir_transitions.csv', index=False)
    require(all(np.isfinite(r['CBDir']) for r in transition_rows),
            'one or more transitions have no evaluable boundary: inspect cbdir_transitions.csv')
    scores = {}
    for metric in ('cbdir', 'icvcoh', 'cto', 'tsc'):
        frame = single_metric(metric=metric, result_path=result_path, methods=['candidate'],
                              k_fold=0, cell_type_transitions=C.CELL_TYPE_TRANSITIONS,
                              time_transitions=time_transitions)
        value = float(frame.iloc[0, 1])
        require(np.isfinite(value), f'official {metric}: nonfinite result')
        scores[metric] = value
    pd.DataFrame({'cell_id': adata.obs_names, 'stage': adata.obs.stage.astype(str),
                  'stage_day': adata.obs.stage_day.to_numpy(),
                  'inferred_time': chosen_time.to_numpy()}).to_csv(result_path / 'cell_times.csv', index=False)
    return scores, time_transitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--prediction', required=True, type=Path)
    parser.add_argument('--method', required=True)
    parser.add_argument('--reference', type=Path, default=C.DATA / 'erythroid.h5ad')
    parser.add_argument('--velocity-key', default='velocity')
    parser.add_argument('--time-key', help='model latent time; default auto-detect latent_time/model_time')
    parser.add_argument('--genes', type=Path, help='one model gene ID per line; default all reference genes')
    parser.add_argument('--official-hvg-genes', type=Path, help='optional verified official gene list, for comparison only')
    args = parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.method) is not None, 'invalid method directory name')
    result = C.DATA / 'results' / args.method
    result.mkdir(parents=True, exist_ok=False)
    print(f'Evaluation log: {result / "run.log"}', flush=True)
    with (result / 'run.log').open('x', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            metadata = {'status': 'running', 'method': args.method, 'protocol': C.PROTOCOL,
                        'started_at_utc': datetime.now(timezone.utc).isoformat(),
                        'command': sys.argv, 'environment': environment(), 'k_fold': 0,
                        'seed': C.SEED, 'n_jobs': 1, 'basis': 'umap',
                        'postprocess_fresh_process': True,
                        'n_pcs': C.N_PCS, 'n_neighbors': C.N_NEIGHBORS,
                        'velocity_graph': {'sqrt_transform': False, 'xkey': 'Ms'},
                        'root_policy': 'scVelo automatic terminal_states(random_state=0); no stage supervision',
                        'cell_type_transitions': C.CELL_TYPE_TRANSITIONS}
            try:
                check_environment()
                metadata['implementation_sha256'] = {p.name: sha256(p) for p in
                    [C.HERE / name for name in ('config.py', 'common.py', 'prepare.py', 'run.py', 'requirements.lock')]}
                metadata['veloev_commit'] = check_vendor()
                metadata['inputs'] = {k: {'path': str(p.resolve()), 'sha256': sha256(p)} for k, p in
                                      [('prediction', args.prediction), ('reference', args.reference)]}
                ref, pred = ad.read_h5ad(args.reference), ad.read_h5ad(args.prediction)
                genes = read_genes(args.genes) if args.genes else ref.var_names.copy()
                aligned, details = align_prediction(ref, pred, genes, args.velocity_key, args.time_key)
                metadata.update(details)
                metadata['reference_provenance'] = {k: v for k, v in ref.uns['benchmark'].items()
                                                    if k != 'source_cell_ids'}
                metadata['official_hvg_comparison'] = {'status': 'not_verified'}
                if args.official_hvg_genes:
                    official_genes = read_genes(args.official_hvg_genes)
                    metadata['official_hvg_comparison'] = {
                        'manifest': str(args.official_hvg_genes.resolve()),
                        'sha256': sha256(args.official_hvg_genes),
                        'status': 'same_set' if set(genes) == set(official_genes) else 'different_set',
                        'official_n_genes': len(official_genes), 'overlap': len(set(genes) & set(official_genes))}
                metadata['comparison_differences'] = [
                    'full 89267-cell training once, erythroid 9815-cell evaluation, no three-fold retraining',
                    'independent spliced/unspliced total normalization to 1e4 before lineage subset',
                    'geometry HVGs from normalized subset; no raw min_shared_counts=20 filtering',
                    'model gene manifest used without intersection with official HVGs',
                    'PCA seed explicitly 1234; automatic root inference with scVelo defaults',
                ]
                if details['time_kind'] == 'model_latent_time':
                    metadata['root_policy'] = 'not applicable: model latent time supplied'
                (result / 'gene_ids.txt').write_text('\n'.join(genes) + '\n')
                (result / 'geometry_gene_ids.txt').write_text(
                    '\n'.join(ref.var_names[ref.var.benchmark_geometry_hvg]) + '\n')
                del ref, pred
                write_json(result / 'metadata.json', metadata)
                scores, time_transitions = official_evaluation(aligned, result)
                metadata['time_transitions'] = time_transitions
                # Recheck both immutable input artifacts after the evaluation.
                for item in metadata['inputs'].values():
                    require(sha256(item['path']) == item['sha256'], 'input file changed during evaluation')
                pd.DataFrame([{'method': args.method, **scores}]).to_csv(result / 'metrics.csv', index=False)
                metadata['status'] = 'complete'
                metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
                write_json(result / 'metadata.json', metadata)
                print(json.dumps(scores))
            except BaseException as exc:
                metadata.update(status='failed', error=str(exc))
                metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
                write_json(result / 'metadata.json', metadata)
                traceback.print_exc()
                raise
    print(f'Saved evaluation: {result / "metrics.csv"}')


if __name__ == '__main__':
    main()
