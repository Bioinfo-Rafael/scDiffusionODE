"""Three-fold data preparation, strict artifact alignment and score aggregation."""
from __future__ import annotations
import argparse
import contextlib
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import config as C
import upstream
from common import (annotations, check_environment, check_vendor, digest_ids, environment,
                    finite, ids, read_genes, reproducible, require, sha256, write_json)

METRICS = ('cbdir', 'icvcoh', 'cto', 'tsc')


def split_reference(reference):
    ids(reference.obs_names, 'erythroid cell')
    annotations(reference)
    require(reference.obs.celltype.astype(str).value_counts().min() >= C.N_FOLDS,
            'each celltype needs at least three cells')
    # Calls the original test_idx implementation; do not sort input IDs before splitting.
    return upstream.utils().split_anndata_stratified(
        reference, n_splits=C.N_FOLDS, cluster_key='celltype', random_state=C.SPLIT_SEED)


def raw_reference(source, expected_ery=C.ERYTHROID_CELLS):
    require(source.n_obs == expected_ery, f'official raw input must contain {expected_ery} erythroid cells')
    ids(source.obs_names, 'raw cell'); ids(source.var_names, 'raw gene')
    annotations(source)
    require(not source.uns.get('mouse_gastrulation_preprocessing', {}).get('normalization'),
            'raw input declares prior normalization')
    # Checking integer counts cannot prove provenance; the input hash and origin are also recorded.
    for name, matrix in [('X', source.X), *[(k, source.layers.get(k)) for k in ('spliced', 'unspliced')]]:
        require(matrix is not None, f'raw {name}: missing')
        finite(matrix, f'raw {name}')
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        require(np.all(values >= 0) and np.all(values == np.floor(values)),
                f'raw {name}: expected nonnegative integer counts, not normalized expression')
    # Original normalization can consult obs initial_size_*; preserve raw-source
    # annotations exactly instead of dropping library-size provenance.
    ref = ad.AnnData(X=source.X.copy(), obs=source.obs.copy(), var=source.var.copy())
    for key in ('spliced', 'unspliced'):
        # Original notebook calls .toarray() on raw layers. Sparse conversion changes no values.
        ref.layers[key] = sparse.csr_matrix(source.layers[key])
    ref.obs['stage_day'] = annotations(ref)
    return ref


def prepare_folds(input_path, output, profile, *, expected_full=C.FULL_CELLS,
                  expected_ery=C.ERYTHROID_CELLS):
    from prepare import make_reference, build_geometry
    require(profile in ('normalized', 'official-raw'), 'unknown preprocessing profile')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', protocol=C.FOLD_PROTOCOL, profile=profile,
                    started_at_utc=datetime.now(timezone.utc).isoformat(),
                    benchmark_commit=C.BENCHMARK_COMMIT, environment=environment(),
                    split=dict(n_splits=3, shuffle=True, random_state=42, subset='test_idx',
                               stratify='celltype', input_order='preserved'),
                    parameters=dict(seed=C.SEED, n_pcs=C.N_PCS, n_neighbors=C.N_NEIGHBORS,
                                    n_top_genes=C.N_HVG, min_shared_counts=20 if profile == 'official-raw' else None,
                                    recipe='upstream notebook cell 3 unchanged' if profile == 'official-raw'
                                           else 'log1p -> seurat HVG -> arpack PCA -> euclidean neighbors -> UMAP',
                                    umap_random_state=C.SEED, umap_min_dist=0.5, umap_spread=1.0,
                                    pca_random_state='scvelo moments / scanpy defaults' if profile == 'official-raw' else C.SEED),
                    paper_input_and_fold_equivalence='not_verified_against_paper_artifacts', folds=[])
    with (output / 'prepare.log').open('x', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            manifest['upstream_sources'] = upstream.verify_sources()
            manifest['input'] = dict(path=str(Path(input_path).resolve()), sha256=sha256(input_path))
            source = ad.read_h5ad(input_path)
            ref = (make_reference(source, expected_full=expected_full, expected_ery=expected_ery)
                   if profile == 'normalized' else raw_reference(source, expected_ery))
            source_ids = source.obs_names.tolist()
            del source
            erythroid_ids = ref.obs_names.tolist()
            manifest.update(source_cell_ids=source_ids, erythroid_cell_ids=erythroid_ids,
                            input_gene_ids_sha256=digest_ids(ref.var_names), n_cells=ref.n_obs)
            parts = split_reference(ref)
            del ref
            if profile == 'official-raw':
                reproducible()
                # Raw input -> original notebook computation, once per independent fold.
                # Original files are retained separately, without adapter metadata changes.
                upstream.preprocess(parts, output / 'official')
            for i, part in enumerate(parts):
                directory = output / f'fold_{i}'
                directory.mkdir()
                if profile == 'official-raw':
                    prepared = ad.read_h5ad(output / 'official' / 'processed' / f'adata_preprocessed_{i}.h5ad')
                    require(prepared.obs_names.equals(part.obs_names), 'official preprocessing changed cell IDs/order')
                    prepared.var['benchmark_geometry_hvg'] = True
                    prepared.uns['benchmark'] = {}
                    scale = C.RAW_EXPRESSION_SCALE
                else:
                    # Explicitly discard any geometry before fitting on this fold alone.
                    part.obsm.clear(); part.obsp.clear()
                    part.uns.pop('neighbors', None); part.uns.pop('umap', None)
                    prepared = build_geometry(part)
                    scale = C.EXPRESSION_SCALE
                prepared.uns['benchmark'].update(
                    protocol=C.FOLD_PROTOCOL, profile=profile, fold=i,
                    source_cell_ids=np.asarray(source_ids, dtype=str),
                    erythroid_cell_ids=np.asarray(erythroid_ids, dtype=str),
                    source_gene_ids_sha256=digest_ids(prepared.var_names),
                    seed=C.SEED, n_pcs=C.N_PCS, n_neighbors=C.N_NEIGHBORS,
                    expression_scale=scale)
                path = directory / 'reference.h5ad'
                prepared.write_h5ad(path, compression='gzip')
                hvg = prepared.var_names[prepared.var.benchmark_geometry_hvg].tolist()
                (directory / 'cell_ids.txt').write_text('\n'.join(prepared.obs_names) + '\n')
                (directory / 'gene_ids.txt').write_text('\n'.join(prepared.var_names) + '\n')
                (directory / 'hvg_ids.txt').write_text('\n'.join(hvg) + '\n')
                info = dict(fold=i, reference=str(path.relative_to(output)), sha256=sha256(path),
                            n_cells=prepared.n_obs, n_genes=prepared.n_vars,
                            cell_ids=prepared.obs_names.tolist(), hvg_ids=hvg,
                            celltype_counts=prepared.obs.celltype.value_counts().to_dict(),
                            geometry_fit='this_fold_only', parameters=manifest['parameters'])
                write_json(directory / 'metadata.json', info)
                manifest['folds'].append(info)
            require(sha256(input_path) == manifest['input']['sha256'], 'input changed during preparation')
            manifest['status'] = 'complete'
        except BaseException as exc:
            manifest.update(status='failed', error=str(exc))
            traceback.print_exc()
            raise
        finally:
            manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(output / 'manifest.json', manifest)
    return manifest


def validate_manifest(directory, expected_ery=C.ERYTHROID_CELLS):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    require(manifest.get('status') == 'complete' and manifest.get('protocol') == C.FOLD_PROTOCOL,
            'incomplete or incompatible fold reference')
    require(manifest['split'] == dict(n_splits=3, shuffle=True, random_state=42, subset='test_idx',
                                      stratify='celltype', input_order='preserved'), 'fold split parameters differ')
    require([f['fold'] for f in manifest['folds']] == list(range(C.N_FOLDS)), 'expected folds 0, 1, 2')
    ids(pd.Index(manifest['source_cell_ids']), 'source cell manifest')
    ids(pd.Index(manifest['erythroid_cell_ids']), 'erythroid cell manifest')
    all_ids = [cell for f in manifest['folds'] for cell in f['cell_ids']]
    ids(pd.Index(all_ids), 'fold union')
    require(len(all_ids) == expected_ery and set(all_ids) == set(manifest['erythroid_cell_ids']),
            'fold union does not equal complete erythroid lineage')
    require(set(all_ids) <= set(manifest['source_cell_ids']), 'fold cells absent from source manifest')
    for item in manifest['folds']:
        require(item['n_cells'] == len(item['cell_ids']), 'fold cell count mismatch')
        path = directory / item['reference']
        require(sha256(path) == item['sha256'], f'fold reference changed: {path}')
    return manifest


def aggregate(rows):
    frame = pd.DataFrame(rows)
    require(len(frame) == C.N_FOLDS and set(frame.fold) == set(range(C.N_FOLDS)),
            'aggregation requires exactly folds 0, 1, 2')
    finite(frame[list(METRICS)].to_numpy(), 'fold scores')
    return pd.DataFrame([dict(metric=k, mean=frame[k].mean(), std=frame[k].std(ddof=C.STD_DDOF),
                              ddof=C.STD_DDOF, n_folds=C.N_FOLDS) for k in METRICS])


def evaluate_folds(reference, predictions, result, *, method, genes=None, velocity_key='velocity',
                   time_key=None, time_kind=None, inference_protocol='full-trained',
                   expected_ery=C.ERYTHROID_CELLS):
    from run import align_prediction, official_evaluation
    require(inference_protocol in ('full-trained', 'per-fold'), 'invalid inference protocol')
    require(len(predictions) == (1 if inference_protocol == 'full-trained' else 3),
            'full-trained needs one prediction; per-fold needs three predictions')
    manifest = validate_manifest(reference, expected_ery)
    result = Path(result)
    result.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='running', method=method, protocol=C.FOLD_PROTOCOL,
                    started_at_utc=datetime.now(timezone.utc).isoformat(),
                    evaluation='3fold', n_folds=3, veloev_k_fold_per_isolated_fold=0,
                    inference_protocol=inference_protocol, training_repeated=inference_protocol == 'per-fold',
                    environment=environment(), benchmark_commit=C.BENCHMARK_COMMIT,
                    reference_manifest_sha256=sha256(Path(reference) / 'manifest.json'),
                    profile=manifest['profile'], std_ddof=C.STD_DDOF,
                    comparison_differences=(
                        ['independent S/U normalization to 1e4 before lineage extraction',
                         'no raw min_shared_counts filtering; normalized fold-local geometry HVGs',
                         'velocity graph uses declared model genes, which may differ from each fold HVGs']
                        if manifest['profile'] == 'normalized' else
                        ['raw-source official notebook computation; paper artifact identity not established',
                         'locked evaluation environment differs from official py310_pt212 inference environment',
                         'RNG seed explicitly set by adapter; upstream stochastic script only parses seed']),
                    split=manifest['split'], parameters=manifest['parameters'],
                    paper_input_and_fold_equivalence=manifest['paper_input_and_fold_equivalence'], folds=[])
    with (result / 'run.log').open('x', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            metadata['veloev_commit'] = check_vendor()
            metadata['upstream_sources'] = upstream.verify_sources()
            metadata['implementation_sha256'] = {p.name: sha256(p) for p in C.HERE.glob('*.py')}
            input_hashes = {str(Path(p).resolve()): sha256(p) for p in predictions}
            metadata['prediction_sha256'] = input_hashes
            if genes:
                metadata['gene_manifest'] = dict(path=str(Path(genes).resolve()), sha256=sha256(genes))
            metadata['cell_type_transitions'] = C.CELL_TYPE_TRANSITIONS
            rows = []
            for item in manifest['folds']:
                i = item['fold']
                folder = result / f'fold_{i}'
                folder.mkdir()
                detail = dict(fold=i, status='running', inference_protocol=inference_protocol,
                              reference=item, velocity_graph=dict(sqrt_transform=False, xkey='Ms'),
                              root_policy='scVelo automatic defaults; no stage supervision/sign correction',
                              n_jobs=1, seed=C.SEED)
                metadata['folds'].append(detail)
                with (folder / 'run.log').open('x', buffering=1) as fold_log, contextlib.redirect_stdout(fold_log), contextlib.redirect_stderr(fold_log):
                    try:
                        ref = ad.read_h5ad(Path(reference) / item['reference'])
                        require(ref.obs_names.tolist() == item['cell_ids'], 'fold cell order mismatch')
                        require(ref.uns['benchmark']['fold'] == i, 'reference fold metadata mismatch')
                        require(list(ref.uns['benchmark']['source_cell_ids']) == manifest['source_cell_ids'],
                                'reference source manifest differs')
                        require(list(ref.uns['benchmark']['erythroid_cell_ids']) == manifest['erythroid_cell_ids'],
                                'reference erythroid manifest differs')
                        pred = ad.read_h5ad(predictions[0 if inference_protocol == 'full-trained' else i])
                        selected = read_genes(genes) if genes else ref.var_names.copy()
                        aligned, alignment = align_prediction(
                            ref, pred, selected, velocity_key, time_key,
                            expected_full=len(manifest['source_cell_ids']), expected_ery=item['n_cells'],
                            inference_protocol=inference_protocol, time_kind=time_kind)
                        detail.update(alignment)
                        detail['hvg_comparison'] = dict(
                            status='same_set' if set(selected) == set(item['hvg_ids']) else 'different_set',
                            selected_n=len(selected), geometry_hvg_n=len(item['hvg_ids']),
                            overlap=len(set(selected) & set(item['hvg_ids'])))
                        if alignment['input_time_key']:
                            detail['root_policy'] = 'supplied time preserved exactly; no benchmark root selection'
                        for name, values in [('cell_ids', aligned.obs_names), ('gene_ids', selected),
                                             ('hvg_ids', item['hvg_ids'])]:
                            (folder / f'{name}.txt').write_text('\n'.join(values) + '\n')
                        del ref, pred
                        scores, time_pairs = official_evaluation(aligned, folder)
                        detail.update(status='complete', time_transitions=time_pairs)
                        row = dict(method=method, fold=i, n_cells=aligned.n_obs, n_genes=aligned.n_vars, **scores)
                        pd.DataFrame([row]).to_csv(folder / 'metrics.csv', index=False)
                        rows.append(row)
                        # Retain partial fold scores even when a later fold fails.
                        pd.DataFrame(rows).to_csv(result / 'metrics_per_fold.csv', index=False)
                        del aligned
                    except BaseException as exc:
                        detail.update(status='failed', error=str(exc))
                        traceback.print_exc()
                        raise
                    finally:
                        write_json(folder / 'metadata.json', detail)
                print(f'Fold {i} complete', flush=True)
            for path, checksum in input_hashes.items():
                require(sha256(path) == checksum, 'prediction changed during evaluation')
            if genes:
                require(sha256(genes) == metadata['gene_manifest']['sha256'], 'gene manifest changed during evaluation')
            require(sha256(Path(reference) / 'manifest.json') == metadata['reference_manifest_sha256'],
                    'reference manifest changed during evaluation')
            validate_manifest(reference, expected_ery)
            aggregate(rows).to_csv(result / 'metrics_summary.csv', index=False)
            metadata.update(status='complete', evaluation_n_cells=sum(r['n_cells'] for r in rows))
        except BaseException as exc:
            metadata.update(status='failed', error=str(exc))
            traceback.print_exc()
            raise
        finally:
            metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(result / 'metadata.json', metadata)
    return result


def evaluation_main(argv):
    parser = argparse.ArgumentParser(description='Three disjoint erythroid test folds; no automatic retraining.')
    parser.add_argument('--evaluation', choices=['3fold', 'full'], default='3fold',
                        help='default 3fold; full selects the legacy single-reference CLI')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prediction', type=Path, help='one full-trained prediction (full or complete erythroid)')
    group.add_argument('--fold-predictions', type=Path, nargs=3, metavar=('FOLD0', 'FOLD1', 'FOLD2'))
    parser.add_argument('--reference', type=Path, default=C.REFERENCE, help='prepared three-fold directory')
    parser.add_argument('--method', required=True)
    parser.add_argument('--genes', type=Path, help='explicit expected gene manifest; default all reference genes')
    parser.add_argument('--velocity-key', default='velocity')
    parser.add_argument('--time-key')
    parser.add_argument('--time-kind', choices=['model_latent_time', 'precomputed_velocity_pseudotime'])
    parser.add_argument('--output-dir', type=Path, help='new result directory; default data/benchmark/results/<method>')
    args = parser.parse_args(argv)
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.method) is not None, 'invalid method name')
    check_environment()
    result = evaluate_folds(args.reference, [args.prediction] if args.prediction else args.fold_predictions,
                           args.output_dir or C.DATA / 'results' / args.method, method=args.method,
                           genes=args.genes, velocity_key=args.velocity_key, time_key=args.time_key,
                           time_kind=args.time_kind,
                           inference_protocol='full-trained' if args.prediction else 'per-fold')
    print(f'Saved three-fold evaluation: {result}')
