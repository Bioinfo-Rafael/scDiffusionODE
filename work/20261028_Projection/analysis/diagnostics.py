"""Sampling, failure accounting, bounded per-cell caches and result tables."""
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from common.base import sha256
from common.levelset_projection import ProjectionError
from .surface_divergence import estimate_surface_divergence
from .gene_contributions import exact_gene_contributions

METHODS = ('hodge', 'cellrank')
METRICS = ('surface_divergence', 'jacobian_term', 'curvature_term', 'trace_standard_error',
           'additive_residual', 'surface_residual', 'displacement', 'distribution_distance',
           'gradient_norm', 'tangent_error')


def cell_classes():
    return list(importlib.import_module('data_preparation.20261007.benchmark.config').ERYTHROID_CELLTYPES)


def dense(x):
    return np.asarray(x.toarray() if hasattr(x, 'toarray') else x, dtype=float)


def json_value(value):
    if isinstance(value, np.ndarray): return [json_value(v) for v in value.tolist()]
    if isinstance(value, dict): return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_value(v) for v in value]
    if isinstance(value, np.generic): return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def save_npz_new(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, suffix='.npz')
    os.close(handle)
    try:
        np.savez_compressed(name, **arrays)
        os.link(name, path)  # Exclusive, atomic publication; never overwrite an existing cell result.
    finally:
        Path(name).unlink(missing_ok=True)


def stratified_sample(labels, per_class, seed, classes):
    if per_class < 0: raise ValueError('per_class must be >=0 (0 selects all target cells)')
    rng = np.random.default_rng(seed)
    chosen = []
    for label in classes:
        indices = np.flatnonzero(np.asarray(labels) == label)
        count = len(indices) if per_class == 0 else min(per_class, len(indices))
        chosen.extend(rng.choice(indices, count, replace=False).tolist())
    return np.array(sorted(chosen), dtype=int)


def scalar_selection(projector, target):
    values = np.concatenate([projector.scalar.values(dense(target.X[i:i+512])) for i in range(0, len(target), 512)])
    eligible_ids = set(projector.metadata['eligible_cells'])
    eligible = np.isfinite(values) & np.array([cell in eligible_ids for cell in target.obs_names])
    if projector.method == 'cellrank':
        source = projector.adata.obs['projection_cellrank_eligible'].reindex(target.obs_names)
        if source.isna().any(): raise ValueError('Missing CellRank eligibility annotations')
        eligible &= source.to_numpy(bool)
    if not eligible.any(): raise ValueError(f'No valid target scalars: {projector.method}')
    c = float(np.median(values[eligible]))
    selection = dict(c=c, target_count=len(target), eligible_count=int(eligible.sum()),
                     excluded_count=int((~eligible).sum()),
                     exclusion_reason='nonfinite scalar or outside original scalar eligible population',
                     quantiles=np.quantile(values[eligible], [0, .25, .5, .75, 1]).tolist(),
                     median_population='all eligible cells in the five target classes, before overview sampling')
    return values, eligible, selection


def analyze_overview(projector, derivatives, target, sample, values, eligible, c, cache,
                     *, probes=64, seed=0):
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    field = projector.get_projected_field(c)
    selected = set(sample.tolist())
    records, points = [], {}
    for i, cell in enumerate(target.obs_names):
        record = dict(cell_id=str(cell), celltype=str(target.obs.iloc[i]['celltype']), target_row=i,
                      scalar=float(values[i]), eligible=bool(eligible[i]), sampled=i in selected,
                      status='not_sampled', reason='', retraction_success=None,
                      **{metric: np.nan for metric in METRICS})
        if not eligible[i]:
            record.update(status='scalar_ineligible', reason='not in valid scalar population')
        elif i in selected:
            path = cache/f'cell_{i:07d}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    record = json.loads(str(saved['record_json']))
                    for metric in METRICS:
                        if record.get(metric) is None: record[metric] = np.nan
                    if record['cell_id'] != cell: raise ValueError('Cached cell order mismatch')
                    if record['status'] == 'ok': points[i] = saved['surface_point']
            else:
                x = dense(target.X[i:i+1])[0]
                y = np.empty(0); samples = {}
                try:
                    y = field.surface_points(x)  # Existing retraction; never differentiated.
                    diagnostics = dict(field.last_diagnostics)
                    tangent_velocity = field(y)  # Existing callable, already on its surface.
                    diagnostics['tangent_error'] = field.last_diagnostics['tangent_error']
                    for key in ('surface_residual', 'displacement', 'distribution_distance', 'gradient_norm', 'tangent_error'):
                        record[key] = float(diagnostics[key][0])
                    record['retraction_success'] = True
                    f = derivatives.value(y)
                    gradient = projector.scalar.gradient(y)[0]; n = gradient/np.linalg.norm(gradient)
                    if not np.allclose(tangent_velocity, f-n*(n @ f), rtol=2e-5, atol=1e-7):
                        raise ValueError('AD velocity and existing projected callable differ')
                    result, samples = estimate_surface_divergence(derivatives, projector.scalar, y,
                                                                 probes=probes, seed=seed+i)
                    record.update(result, status='ok')
                    points[i] = y
                except ProjectionError as exc:
                    reason = str(exc.diagnostics['status'][0])
                    record.update(status='surface_undefined' if 'vanishing_gradient' in reason else 'retraction_failed',
                                  reason=reason, retraction_success=False)
                    for key in ('surface_residual', 'displacement', 'distribution_distance', 'gradient_norm'):
                        record[key] = float(exc.diagnostics[key][0])
                except (RuntimeError, ValueError) as exc:
                    record.update(status='derivative_failed' if record['retraction_success'] else 'retraction_failed',
                                  reason=f'{type(exc).__name__}: {exc}',
                                  retraction_success=bool(record['retraction_success']))
                save_npz_new(path, record_json=np.array(json.dumps(json_value(record))), surface_point=y, **samples)
                print(f'{projector.method}: overview {len(records)+1}/{len(target)}: {record["status"]}', flush=True)
        records.append(record)
    return pd.DataFrame(records).set_index('cell_id'), points


def representative_indices(tables, classes, per_class, seed):
    if per_class < 1: raise ValueError('At least one representative per available class required')
    common = (tables['hodge']['status'].to_numpy() == 'ok') & (tables['cellrank']['status'].to_numpy() == 'ok')
    labels = tables['hodge']['celltype'].to_numpy()
    rng = np.random.default_rng(seed)
    result = {method: [] for method in METHODS}
    for label in classes:
        shared = np.flatnonzero(common & (labels == label))
        shared = rng.permutation(shared)[:per_class].tolist()
        for method in METHODS:
            own = np.flatnonzero((tables[method]['status'].to_numpy() == 'ok') & (labels == label))
            extra = np.array([i for i in own if i not in shared], dtype=int)
            result[method].extend(shared+rng.permutation(extra)[:max(0, per_class-len(shared))].tolist())
    return {method: np.array(rows, dtype=int) for method, rows in result.items()}


def analyze_genes(projector, derivatives, table, points, indices, feature_ids, cache):
    arrays = {key: [] for key in ('jacobian', 'curvature', 'total')}
    summaries, kept = [], []
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    for i in indices:
        cell = str(table.index[i]); path = cache/f'gene_{i:07d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                summary = json.loads(str(saved['summary_json']))
                data = {key: saved[key] for key in arrays} if summary['status'] == 'ok' else {}
        else:
            summary = dict(cell_id=cell, celltype=str(table.iloc[i]['celltype']), target_row=int(i))
            try:
                result = exact_gene_contributions(derivatives, projector.scalar, points[i])
                summary.update(result['summary'], status='ok')
                estimate = float(table.iloc[i]['surface_divergence'])
                error = estimate-summary['surface_divergence']
                se = float(table.iloc[i]['trace_standard_error'])
                summary.update(overview_minus_exact=error, overview_standard_error=se,
                               overview_error_in_se=error/se if se > 0 else None)
                data = {key: result[key] for key in arrays}
            except (RuntimeError, ValueError) as exc:
                summary.update(status='failed', reason=f'{type(exc).__name__}: {exc}'); data = {}
            save_npz_new(path, summary_json=np.array(json.dumps(json_value(summary))), **data)
            print(f'{projector.method}: exact gene contributions {cell}: {summary["status"]}', flush=True)
        summaries.append(summary)
        if summary['status'] == 'ok':
            for key in arrays: arrays[key].append(data[key])
            kept.append(i)
    return {key: np.array(value).reshape(-1, len(feature_ids)) for key, value in arrays.items()}, (pd.DataFrame(summaries) if summaries else pd.DataFrame(columns=['cell_id', 'celltype', 'target_row', 'status', 'reason'])), np.array(kept, dtype=int)


def method_summary(table, selection):
    attempted = table['sampled'] & table['eligible']
    valid = table[table['status'] == 'ok']
    success = int(table.loc[attempted, 'retraction_success'].eq(True).sum())
    return dict(**selection, sampled_count=int(table['sampled'].sum()), attempted_retractions=int(attempted.sum()),
                retraction_success_count=success, retraction_success_rate=success/int(attempted.sum()) if attempted.any() else None,
                divergence_success_count=len(valid), statuses=dict(Counter(table['status'])),
                failure_reasons=dict(Counter(table.loc[attempted & (table.status != 'ok'), 'reason'])),
                divergence_mean=float(valid.surface_divergence.mean()), divergence_median=float(valid.surface_divergence.median()),
                jacobian_mean=float(valid.jacobian_term.mean()), curvature_mean=float(valid.curvature_term.mean()),
                median_trace_se=float(valid.trace_standard_error.median()),
                scalar_bias_warning='Overview describes the selected successful sample, not all target cells')


def gene_rank_table(arrays, feature_ids):
    if not len(arrays['total']): return pd.DataFrame(columns=['gene_id', 'signed_mean', 'mean_absolute', 'rank'])
    frame = pd.DataFrame(dict(gene_id=feature_ids, signed_mean=arrays['total'].mean(0),
        mean_absolute=abs(arrays['total']).mean(0), jacobian_mean=arrays['jacobian'].mean(0),
        curvature_mean=arrays['curvature'].mean(0)))
    frame['rank'] = frame.mean_absolute.rank(ascending=False, method='min').astype(int)
    return frame.sort_values(['mean_absolute', 'gene_id'], ascending=[False, True])


def comparison_summary(tables, genes, representatives, feature_ids, top_n=20):
    paired = (tables['hodge'].status == 'ok') & (tables['cellrank'].status == 'ok')
    a = tables['hodge'].loc[paired, 'surface_divergence'].to_numpy()
    b = tables['cellrank'].loc[paired, 'surface_divergence'].to_numpy()
    result = dict(common_success_count=int(paired.sum()), paired_cell_ids=tables['hodge'].index[paired].tolist(),
                  divergence_comparison='same original cells, different method-specific surface points and c',
                  correlation_candidates=1, correlation_selection='pre-specified method pair; no scan over alternatives',
                  shared_mathematics='same original F, input cells and projection equations; not independent biological evidence')
    if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0:
        result.update(pearson=float(pearsonr(a, b).statistic), spearman=float(spearmanr(a, b).statistic))
    else:
        result.update(pearson=None, spearman=None)
    result['common_population_statistics'] = {method: tables[method].loc[paired, ['surface_divergence', 'jacobian_term', 'curvature_term']].agg(['mean', 'median']).to_dict() for method in METHODS}
    common_reps = sorted(set(representatives['hodge']).intersection(representatives['cellrank']))
    result.update(common_representative_count=len(common_reps), ranking_candidate_genes=len(feature_ids))
    if common_reps:
        means = {}
        for method in METHODS:
            rows = [list(representatives[method]).index(i) for i in common_reps]
            means[method] = abs(genes[method]['total'][rows]).mean(0)
        if np.std(means['hodge']) > 0 and np.std(means['cellrank']) > 0:
            result['gene_rank_spearman'] = float(spearmanr(means['hodge'], means['cellrank']).statistic)
        else: result['gene_rank_spearman'] = None
        k = min(top_n, len(feature_ids))
        top = {m: set(np.argsort(-means[m], kind='stable')[:k]) for m in METHODS}
        result.update(top_k=k, top_k_overlap_informative=k < len(feature_ids),
                      top_k_overlap=len(top['hodge'] & top['cellrank']),
                      top_k_jaccard=len(top['hodge'] & top['cellrank'])/len(top['hodge'] | top['cellrank']))
    return result
