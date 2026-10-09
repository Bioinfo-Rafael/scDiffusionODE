#!/usr/bin/env python3
"""Remote Linux only: Data 3 raw preprocessing -> official stochastic -> shared benchmark."""
from __future__ import annotations
import argparse
import contextlib
import json
import platform
import shutil
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCHMARK = ROOT / 'data_preparation' / '20261007' / 'benchmark'
sys.path.insert(0, str(BENCHMARK))
import config as C
if __name__ == '__main__':
    C.launch_isolated(__file__)

import anndata as ad
import numpy as np
import pandas as pd
import scvelo as scv
from scipy import sparse
import upstream
from folds import METRICS, evaluate_folds, prepare_folds, raw_reference
from common import (check_environment, check_vendor, digest_ids, environment, finite, ids,
                    reproducible, require, sha256, write_json)


def matrix_stats(matrix):
    finite(matrix, 'comparison matrix')
    noninteger = nonzero = 0
    for start in range(0, matrix.shape[0], 256):
        block = matrix[start:start + 256]
        values = block.data if sparse.issparse(block) else np.asarray(block).ravel()
        noninteger += int(np.count_nonzero(values != np.floor(values)))
        nonzero += int(np.count_nonzero(values))
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    return dict(nonzero_count=nonzero, noninteger_count=noninteger,
                row_sum_quantiles=np.quantile(totals, [0, .25, .5, .75, 1]).tolist(),
                independently_normalized_1e4=bool(np.all((totals == 0) | np.isclose(totals, 1e4, rtol=1e-5, atol=1e-3))))


def compare_inputs(raw, existing):
    """Exact IDs only; report mismatches instead of rewriting barcodes or gene symbols."""
    for a in (raw, existing):
        ids(a.obs_names, 'comparison cell'); ids(a.var_names, 'comparison gene')
    keep = existing.obs.celltype.isin(C.ERYTHROID_CELLTYPES)
    existing_cells = existing.obs_names[keep]
    cells = raw.obs_names[raw.obs_names.isin(existing_cells)]
    genes = raw.var_names[raw.var_names.isin(existing.var_names)]
    result = dict(raw_shape=list(raw.shape), existing_shape=list(existing.shape),
                  existing_erythroid_n=len(existing_cells), shared_cells=len(cells), shared_genes=len(genes),
                  same_erythroid_cell_set=set(raw.obs_names) == set(existing_cells),
                  same_erythroid_cell_order=raw.obs_names.equals(existing_cells),
                  same_gene_set=set(raw.var_names) == set(existing.var_names),
                  same_gene_order=raw.var_names.equals(existing.var_names),
                  raw_cell_order_sha256=digest_ids(raw.obs_names),
                  existing_cell_order_sha256=digest_ids(existing_cells),
                  raw_gene_order_sha256=digest_ids(raw.var_names),
                  existing_gene_order_sha256=digest_ids(existing.var_names),
                  existing_normalization=dict(existing.uns.get('mouse_gastrulation_preprocessing', {})),
                  comparison_scope='exact shared IDs; raw normalization denominator uses all raw genes', layers={})
    for key in ('spliced', 'unspliced'):
        require(key in raw.layers and key in existing.layers, f'comparison {key} missing')
        raw_mat, current = raw.layers[key], existing.layers[key]
        details = dict(raw=matrix_stats(raw_mat), existing=matrix_stats(current))
        if len(cells) and len(genes):
            raw_rows = raw.obs_names.get_indexer(cells)
            a = raw_mat[raw_rows][:, raw.var_names.get_indexer(genes)]
            b = current[existing.obs_names.get_indexer(cells)][:, existing.var_names.get_indexer(genes)]
            totals = np.asarray(raw_mat[raw_rows].sum(axis=1)).ravel()
            max_raw = max_normalized = 0.
            for start in range(0, len(cells), 128):
                aa = sparse.csr_matrix(a[start:start + 128]).toarray().astype(float)
                bb = sparse.csr_matrix(b[start:start + 128]).toarray().astype(float)
                scale = np.divide(1e4, totals[start:start + 128], out=np.zeros(len(aa)), where=totals[start:start + 128] != 0)
                max_raw = max(max_raw, float(np.max(np.abs(aa - bb))))
                max_normalized = max(max_normalized, float(np.max(np.abs(aa * scale[:, None] - bb))))
            details.update(shared_raw_max_abs_difference=max_raw,
                           shared_raw_normalized_1e4_max_abs_difference=max_normalized)
        result['layers'][key] = details
    result['annotation_comparison_status'] = (
        'compared_on_exact_shared_cells' if len(cells) else 'no_exact_shared_cells')
    result['annotation_matches'] = {
        key: (bool(raw.obs.loc[cells, key].astype(str).equals(existing.obs.loc[cells, key].astype(str)))
              if len(cells) else None)
        for key in ('celltype', 'stage')}
    result['paper_artifact_equivalence'] = 'not_verified; downloaded raw is not a paper fold manifest'
    return result


def adapt_prediction(original, reference, checkpoint):
    """Attach the interface contract; keep the official velocity, mask and time unchanged."""
    require(original.obs_names.equals(reference.obs_names), 'inferred cell IDs/order differ from prepared fold')
    require(original.var_names.equals(reference.var_names), 'inferred gene IDs/order differ from prepared fold')
    require('scvelo_stc_velocity' in original.layers, 'official stochastic velocity missing')
    finite(original.layers['scvelo_stc_velocity'], 'official stochastic velocity')
    for key in ('scvelo_stc_time', 'scvelo_stc_velocity_pseudotime'):
        require(key in original.obs, f'official {key} missing')
        finite(original.obs[key].to_numpy(), key)
    require(np.array_equal(original.obs.scvelo_stc_time, original.obs.scvelo_stc_velocity_pseudotime),
            'official time columns differ')
    pred = ad.AnnData(X=None, obs=original.obs.copy(), var=original.var.copy())
    pred.layers['scvelo_stc_velocity'] = original.layers['scvelo_stc_velocity'].copy()
    pred.uns['benchmark_velocity'] = dict(
        definition='ds_dt', expression_scale=C.RAW_EXPRESSION_SCALE, time_direction='forward',
        training_n_cells=original.n_obs, time_unit='scVelo stochastic arbitrary time',
        inference_description='unchanged official run_scvelo_st; per-fold raw preprocessing',
        checkpoint=str(checkpoint))
    return pred


def compare_scores(actual, expected):
    for frame in (actual, expected):
        require(len(frame) == 3 and set(frame.fold) == {0, 1, 2}, 'comparison requires folds 0, 1, 2')
        finite(frame[list(METRICS)].to_numpy(), 'comparison scores')
    rows = []
    actual, expected = actual.set_index('fold'), expected.set_index('fold')
    for metric in METRICS:
        for fold in range(3):
            value, paper = float(actual.loc[fold, metric]), float(expected.loc[fold, metric])
            error = abs(value - paper)
            rows.append(dict(metric=metric, fold=fold, reproduced=value, paper=paper,
                             difference=value - paper, absolute_error=error,
                             within_1e_5=error <= 1e-5, within_1e_4=error <= 1e-4,
                             input_preprocessing_fold_equivalence='not_verified_against_paper_artifacts'))
    return pd.DataFrame(rows)


def save_comparison(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    actual = pd.read_csv(result / 'metrics_per_fold.csv')
    expected = pd.read_csv(HERE / 'paper_expectations.csv')
    comparison = compare_scores(actual, expected)
    comparison.to_csv(result / 'comparison_with_paper.csv', index=False)
    for metric in METRICS:
        frame = comparison[comparison.metric == metric]
        fig, ax = plt.subplots(figsize=(5, 3.5), layout='constrained')
        ax.plot(frame.fold, frame.paper, 'o-', label='Official reported')
        ax.plot(frame.fold, frame.reproduced, 's--', label='Reproduced')
        ax.set(xticks=[0, 1, 2], xlabel='Fold', ylabel=metric.upper())
        ax.axhline(0, color='grey', linewidth=.5)
        ax.legend()
        fig.savefig(result / f'comparison_{metric}.png', dpi=180)
        plt.close(fig)
    return dict(all_within_1e_5=bool(comparison.within_1e_5.all()),
                all_within_1e_4=bool(comparison.within_1e_4.all()),
                complete_paper_reproduction='not established by numeric tolerance alone')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-input', type=Path, help='optional original Data 3 raw h5ad; default official scVelo loader')
    parser.add_argument('--existing-input', type=Path, default=C.SOURCE)
    parser.add_argument('--output', type=Path, default=HERE / 'runs' / 'scvelo_stochastic')
    args = parser.parse_args()
    require(platform.system() == 'Linux', 'real-data reproduction is authorized on remote Linux only')
    check_environment(); check_vendor(); upstream.verify_sources()
    require(args.existing_input.is_file(), 'existing MouseGastrulation.h5ad required for input comparison')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='running', benchmark_commit=C.BENCHMARK_COMMIT, veloev_commit=C.VELOEV_COMMIT,
                    environment=environment(), seed=C.SEED, split_seed=C.SPLIT_SEED,
                    official_seed_difference='upstream parses --seed but never seeds RNG; adapter seeds NumPy/Python',
                    code_sha256=sha256(__file__), paper_expectations_sha256=sha256(HERE / 'paper_expectations.csv'),
                    upstream_sources=upstream.verify_sources(), command=sys.argv,
                    input_preprocessing_fold_equivalence='not_verified_against_paper_artifacts')
    print(f'Reproduction log: {output / "run.log"}', flush=True)
    with (output / 'run.log').open('x', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            existing_hash = sha256(args.existing_input)
            if args.raw_input:
                raw_path = args.raw_input.resolve()
                raw = ad.read_h5ad(raw_path)
                origin = 'explicit --raw-input'
            else:
                raw_path = output / 'gastrulation_erythroid.h5ad'
                raw = scv.datasets.gastrulation_erythroid(file_path=raw_path)
                origin = 'scvelo.datasets.gastrulation_erythroid; figshare file 27686871'
                # Loader applies var_names_make_unique; capture exactly the returned IDs too.
                raw_path = output / 'loader_returned.h5ad'
                raw.write_h5ad(raw_path)
            raw_hash = sha256(raw_path)
            metadata.update(raw_input=dict(path=str(raw_path), sha256=raw_hash, origin=origin),
                            existing_input=dict(path=str(args.existing_input), sha256=existing_hash))
            existing = ad.read_h5ad(args.existing_input)
            write_json(output / 'input_comparison.json', compare_inputs(raw, existing))
            del existing
            raw_reference(raw)  # refuse normalized data, missing stages or unexpected IDs/counts
            del raw
            references = output / 'references'
            prepare_folds(raw_path, references, 'official-raw')
            official = references / 'official'
            function = upstream.stochastic_function(official)
            predictions = []
            reproducible()
            for i in range(3):
                folder = output / f'inference_fold_{i}'
                folder.mkdir()
                info = dict(status='running', fold=i, inference_protocol='per-fold',
                            function='run_scvelo_st', upstream_function_body_modified=False,
                            seed_policy='one seed before ordered loop over folds 0,1,2')
                with (folder / 'run.log').open('x', buffering=1) as fold_log, contextlib.redirect_stdout(fold_log), contextlib.redirect_stderr(fold_log):
                    try:
                        function(i)
                        original_path = official / 'processed' / f'adata_run_scvelo_stc_{i}.h5ad'
                        original = ad.read_h5ad(original_path)
                        reference = ad.read_h5ad(references / f'fold_{i}' / 'reference.h5ad')
                        pred = adapt_prediction(original, reference, original_path)
                        pred_path = folder / 'prediction.h5ad'
                        pred.write_h5ad(pred_path, compression='gzip')
                        predictions.append(pred_path)
                        info.update(status='complete', original=str(original_path), original_sha256=sha256(original_path),
                                    prediction_sha256=sha256(pred_path), n_cells=pred.n_obs, n_genes=pred.n_vars)
                        del original, reference, pred
                    except BaseException as exc:
                        info.update(status='failed', error=str(exc))
                        traceback.print_exc()
                        raise
                    finally:
                        write_json(folder / 'metadata.json', info)
            result = evaluate_folds(references, predictions, output / 'evaluation', method='scvelo_stc',
                                    velocity_key='scvelo_stc_velocity', time_key='scvelo_stc_time',
                                    time_kind='precomputed_velocity_pseudotime', inference_protocol='per-fold')
            for name in ('metrics_per_fold.csv', 'metrics_summary.csv'):
                shutil.copyfile(result / name, output / name)
            metadata['comparison'] = save_comparison(output)
            require(sha256(args.existing_input) == existing_hash and sha256(raw_path) == raw_hash,
                    'input changed during reproduction')
            metadata['status'] = 'complete'
        except BaseException as exc:
            metadata.update(status='failed', error=str(exc))
            traceback.print_exc()
            raise
        finally:
            write_json(output / 'metadata.json', metadata)
    print(f'Saved reproduction and comparison: {output}')


if __name__ == '__main__':
    main()
