"""Read model -> prepare full-population scalars -> median c -> retract -> derivatives -> save -> five figures."""
import argparse
import json
from pathlib import Path
import sys
import traceback
import importlib.metadata
import hashlib
import os
import numpy as np
import pandas as pd
import anndata as ad
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
for folder in (HERE, ROOT):
    if str(folder) not in sys.path: sys.path.insert(0, str(folder))
from common.velocity_adapter import load_d20
from common.base import data_identity, sha256
from Hodge.projector import HodgeProjector
from CellRank.projector import CellRankProjector
from CellRank.subprocess_bridge import check_backend
from analysis.surface_divergence import TorchDerivatives
from analysis.gene_contributions import grn_sensitivities
from analysis.diagnostics import (METHODS, METRICS, cell_classes, write_json, json_value, save_npz_new,
    stratified_sample, scalar_selection, analyze_overview, representative_indices, analyze_genes,
    method_summary, gene_rank_table, comparison_summary)


def prepare_scalar(cls, velocity, full, saved, output, config, signature, smoke, cellrank_python=None):
    """Reuse verified ALL-cell artifacts; explicitly recompute mismatched/subset populations."""
    destination = output/'scalars'/cls.method
    candidate = destination if (destination/'metadata.json').exists() else saved
    reason = 'No saved scalar artifacts available'
    if candidate and (candidate/'metadata.json').exists():
        metadata = json.loads((candidate/'metadata.json').read_text())
        same_population = metadata.get('expression_and_ids_sha256') == signature
        terminal_ok = True
        if cls.method == 'cellrank':
            expected = full.obs_names[full.obs['celltype'] == 'Erythroid3'].tolist()
            terminal_ok = (metadata.get('terminal_obs_key') == 'celltype' and
                           metadata.get('terminal_cells') == {'Erythroid3': expected})
        if same_population and terminal_ok:
            annotated = ad.read_h5ad(candidate/'annotated.h5ad')
            projector = cls.load_saved(candidate, velocity, annotated)
            return projector, dict(action='reused_full_population', path=str(candidate),
                                   scalar_population=full.n_obs, metadata_sha256=sha256(candidate/'metadata.json'))
        reason = 'Saved scalar population/terminal definition differs; explicitly recomputing on ALL current cells'
    print(f'{cls.method}: {reason}; population={full.n_obs}', flush=True)
    settings = dict(config['interpolation'])
    if smoke: settings.update(n_components=8, n_centers=96)
    projector = cls(velocity, full.copy(), interpolation=settings)
    if cls.method == 'hodge':
        projector.fit_scalar(k=12, rdim=4, seed=config['seed'])
    else:
        projector.fit_scalar(terminal_obs_key='celltype', terminal_values=['Erythroid3'],
                             n_neighbors=15 if smoke else 30, seed=config['seed'],
                             cellrank_python=cellrank_python, worker_directory=output/'cellrank_worker')
    projector.save(destination)
    return projector, dict(action='recomputed_full_population', reason=reason, path=str(destination),
                           scalar_population=full.n_obs, metadata_sha256=sha256(destination/'metadata.json'))


def optional_grn(config, velocity, rankings, points, representatives, derivatives, smoke):
    if smoke:
        return [], dict(status='not_applicable', reason='Synthetic fixture has no biological GRN')
    manifest = ROOT/Path(config['data_dir'])/'manifest.json'
    meta = json.loads(manifest.read_text()).get('grn', {})
    path = Path(meta.get('mapped_tsv', ''))
    if not path.is_file() or not meta.get('mapped_sha256') or sha256(path) != meta['mapped_sha256']:
        return [], dict(status='unverified', reason='No hash-verified training GRN candidate file')
    edges = pd.read_csv(path, sep='\t', dtype=str)
    if not {'from', 'to'} <= set(edges):
        return [], dict(status='unverified', reason='GRN needs from/to gene-ID columns')
    top = set()
    for method in METHODS: top.update(rankings[method].head(20).gene_id)
    incident = edges[edges['from'].isin(top) | edges['to'].isin(top)].drop_duplicates().sort_values(['from', 'to'])
    chosen = list(incident[['from', 'to']].itertuples(index=False, name=None))[:200]
    records = []
    for method in METHODS:
        result = grn_sensitivities(derivatives, [points[method][i] for i in representatives[method]],
                                  chosen, velocity.feature_ids, candidate_sha256=meta['mapped_sha256'])
        records.extend(dict(method=method, **row) for row in result)
    return records, dict(status='verified_candidates_only', source=str(path), sha256=meta['mapped_sha256'],
                          incident_candidate_count=len(incident), evaluated_candidate_count=len(chosen),
                          selection='sorted first 200 candidate edges incident to top20 genes per method',
                          interpretation='local J[target,source] sensitivity; not causal GRN strength')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(HERE/'config.json'))
    parser.add_argument('--output')
    parser.add_argument('--smoke', action='store_true', help='Explicit synthetic data; never an automatic fallback')
    parser.add_argument('--resume', action='store_true', help='Reuse matching scalar and per-cell derivative caches')
    parser.add_argument('--replot', type=Path, help='Regenerate the same five PNGs using saved tables/NPZ only')
    parser.add_argument('--hodge-dir', type=Path, default=HERE/'outputs/hodge')
    parser.add_argument('--cellrank-dir', type=Path, default=HERE/'outputs/cellrank')
    parser.add_argument('--cellrank-python', default=os.environ.get('CELLRANK_PYTHON'),
                        help='Separate CellRank Python executable; default CELLRANK_PYTHON. Parent retains D20/AD.')
    parser.add_argument('--samples-per-class', type=int, help='Default real:128, smoke:8; 0=all target cells')
    parser.add_argument('--representatives-per-class', type=int, help='Default real:2, smoke:1')
    parser.add_argument('--probes', type=int, help='Default real:64, smoke:32')
    parser.add_argument('--derivative-batch', type=int, default=16)
    parser.add_argument('--seed', type=int, default=28)
    parser.add_argument('--top-genes', type=int, default=20)
    parser.add_argument('--device', default='auto', help='auto selects CUDA when available, otherwise CPU')
    args = parser.parse_args()
    if args.replot:
        from analysis.plotting import plot_saved
        figures = plot_saved(args.replot, top_genes=args.top_genes)
        path = args.replot/'manifest.json'
        saved_manifest = json.loads(path.read_text())
        saved_manifest.setdefault('artifacts_sha256', {}).update({p.name: sha256(p) for p in figures+[args.replot/'FIGURES.md']})
        saved_manifest['last_replot'] = dict(top_genes=args.top_genes, plotting_source_sha256=sha256(HERE/'analysis/plotting.py'))
        write_json(path, saved_manifest)
        return
    args.samples_per_class = args.samples_per_class if args.samples_per_class is not None else (8 if args.smoke else 128)
    args.representatives_per_class = args.representatives_per_class if args.representatives_per_class is not None else (1 if args.smoke else 2)
    args.probes = args.probes if args.probes is not None else (32 if args.smoke else 64)
    if args.samples_per_class < 0 or args.representatives_per_class < 1 or args.probes < 2 or args.derivative_batch < 1:
        parser.error('Invalid sampling/probe/batch setting')
    if not 1 <= args.top_genes <= 30: parser.error('--top-genes must be 1..30')
    output = Path(args.output) if args.output else HERE/'outputs/divergence_analysis'/('smoke' if args.smoke else 'real')
    output = output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f'Refusing overwrite: {output}; use --resume or a new directory')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    previous_manifest_exists = manifest_path.exists()
    resume_identity_verified = False
    manifest = dict(status='running', stage='load_model_and_data', data_kind='synthetic' if args.smoke else 'real_D20',
                    completed_stages=[], request_date_JST='2026-10-28')
    if not previous_manifest_exists: write_json(manifest_path, manifest)
    try:
        manifest['stage'] = 'check_cellrank_backend'
        cellrank_backend = check_backend(args.cellrank_python)
        manifest.update(cellrank_backend=cellrank_backend, parent_python=sys.executable, parent_python_version=sys.version)
        manifest['stage'] = 'load_model_and_data'
        # 1. Load the unchanged model and ALL training cells (no five-class scalar refit).
        config = json.loads(Path(args.config).read_text()); config['seed'] = args.seed
        if config['weights'] != 'ema' or config.get('timestep', 49) != 49:
            raise ValueError('This analysis requires D20 EMA and fixed timestep 49')
        if args.smoke:
            from analysis.smoke_data import make_smoke
            velocity, full, derivatives = make_smoke(args.seed)
        else:
            device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
            velocity, full = load_d20(config, device)
            derivatives = TorchDerivatives.from_ode(velocity, args.derivative_batch)
        derivatives.batch_size = args.derivative_batch
        classes = cell_classes()
        if 'celltype' not in full.obs: raise ValueError('celltype annotation required')
        target_rows = np.flatnonzero(full.obs.celltype.isin(classes).to_numpy())
        if not len(target_rows): raise ValueError('None of the five benchmark classes found')
        missing = set(classes)-set(full.obs.iloc[target_rows].celltype.astype(str))
        if missing: raise ValueError(f'Missing target classes: {sorted(missing)}')
        target = full[target_rows].copy()
        signature = data_identity(full)
        numerical_files = [HERE/'common/scalar_extension.py', HERE/'common/velocity_adapter.py',
            HERE/'common/levelset_projection.py', HERE/'analysis/surface_divergence.py',
            HERE/'analysis/gene_contributions.py', HERE/'analysis/diagnostics.py', Path(__file__),
            HERE/'CellRank/projector.py', HERE/'CellRank/subprocess_bridge.py', HERE/'CellRank/worker.py', HERE/'CellRank/cellrank_core.py']
        if args.smoke: numerical_files.append(HERE/'analysis/smoke_data.py')
        identity = dict(input=signature, model=velocity.provenance, cellrank_backend=cellrank_backend,
            celltype_labels_sha256=hashlib.sha256(json.dumps(full.obs.celltype.astype(str).tolist()).encode()).hexdigest(),
            derivative_device=str(derivatives.device), derivative_dtype=str(derivatives.dtype),
            config=config, sample_per_class=args.samples_per_class, representatives_per_class=args.representatives_per_class,
            probes=args.probes, derivative_batch=args.derivative_batch,
            numerical_sources={str(p.relative_to(HERE)): sha256(p) for p in numerical_files})
        identity_path = output/'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Resume identity changed (input/model/settings/numerical source); use a new output directory')
        if not identity_path.exists(): write_json(identity_path, identity)
        resume_identity_verified = True
        manifest.update(identity=identity, full_scalar_population=full.n_obs, target_population=len(target),
                        target_cell_ids=target.obs_names.tolist(), feature_ids=target.var_names.tolist(),
                        class_counts=target.obs.celltype.value_counts().to_dict(),
                        packages={p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scipy', 'anndata', 'pandas', 'scikit-learn', 'matplotlib')},
                        device=str(derivatives.device), dtype=str(derivatives.dtype))
        manifest['completed_stages'].append('load_model_and_data')
        # 2. Reuse existing official Hodge/CellRank projectors, full-population artifacts preferred.
        projectors, scalar_sources = {}, {}
        for cls, saved in ((HodgeProjector, args.hodge_dir), (CellRankProjector, args.cellrank_dir)):
            manifest['stage'] = 'prepare_scalar_'+cls.method; write_json(manifest_path, manifest)
            projectors[cls.method], scalar_sources[cls.method] = prepare_scalar(cls, velocity, full, saved,
                                                                            output, config, signature, args.smoke, args.cellrank_python)
            manifest['completed_stages'].append(manifest['stage'])
        scalar_identity = {m: scalar_sources[m]['metadata_sha256'] for m in METHODS}
        scalar_identity_path = output/'scalar_identity.json'
        if scalar_identity_path.exists() and json.loads(scalar_identity_path.read_text()) != scalar_identity:
            raise ValueError('Saved scalar changed since cached derivative calculation')
        if not scalar_identity_path.exists(): write_json(scalar_identity_path, scalar_identity)
        # 3. Medians use ALL eligible cells in the five classes, not the overview sample.
        selections, values, eligible = {}, {}, {}
        for method in METHODS:
            values[method], eligible[method], selections[method] = scalar_selection(projectors[method], target)
        sample = stratified_sample(target.obs.celltype.astype(str), args.samples_per_class, args.seed, classes)
        manifest.update(scalar_sources=scalar_sources, selections=selections,
                        scalar_interpolation={m: projectors[m].scalar.metadata() for m in METHODS},
                        sampled_cell_ids=target.obs_names[sample].tolist())
        # 4–5. Existing retraction at c, then surface derivatives at y. Failures stay in the tables.
        tables, points = {}, {}
        for method in METHODS:
            manifest['stage'] = 'overview_'+method; write_json(manifest_path, manifest)
            tables[method], points[method] = analyze_overview(projectors[method], derivatives, target, sample,
                values[method], eligible[method], selections[method]['c'], output/'cache'/method,
                probes=args.probes, seed=args.seed)
            tables[method].to_csv(output/f'{method}_overview.csv')
            manifest['completed_stages'].append(manifest['stage'])
        # 6. Exact streamed gene-coordinate contributions at representative successful points.
        chosen = representative_indices(tables, classes, args.representatives_per_class, args.seed)
        genes, summaries, representatives, rankings = {}, {}, {}, {}
        for method in METHODS:
            manifest['stage'] = 'gene_contributions_'+method; write_json(manifest_path, manifest)
            genes[method], summaries[method], representatives[method] = analyze_genes(projectors[method], derivatives,
                tables[method], points[method], chosen[method], target.var_names.tolist(), output/'cache'/method)
            rankings[method] = gene_rank_table(genes[method], target.var_names.tolist())
            manifest['completed_stages'].append(manifest['stage'])
        manifest['stage'] = 'optional_grn'; write_json(manifest_path, manifest)
        grn_rows, grn_report = optional_grn(config, velocity, rankings, points, representatives, derivatives, args.smoke)
        # 7. Save new annotated target AnnData, tables, contributions and numerical provenance.
        manifest['stage'] = 'save_results'; write_json(manifest_path, manifest)
        output_data = target.copy()
        for method in METHODS:
            prefix = f'projection_{method}_'
            table = tables[method]
            for key in METRICS:
                output_data.obs[prefix+key] = pd.to_numeric(table[key], errors='coerce').to_numpy()
            output_data.obs[prefix+'retraction_success'] = table.retraction_success.map({True: 1., False: 0.}).to_numpy()
            output_data.obs[prefix+'analysis_status'] = table.status.to_numpy(str)
            output_data.obs[prefix+'failure_reason'] = table.reason.fillna('').to_numpy(str)
            output_data.obs[prefix+'scalar'] = values[method]
            output_data.obs[prefix+'scalar_eligible'] = eligible[method]
            output_data.obs[prefix+'sampled'] = table.sampled.to_numpy(bool)
            summaries[method].to_csv(output/f'{method}_representatives.csv', index=False)
            rankings[method].to_csv(output/f'{method}_gene_ranking.csv', index=False)
            path = output/f'{method}_gene_contributions.npz'
            if not path.exists():
                save_npz_new(path, **genes[method], cell_ids=target.obs_names[representatives[method]].to_numpy(str),
                             celltypes=target.obs.iloc[representatives[method]].celltype.to_numpy(str),
                             gene_ids=target.var_names.to_numpy(str), target_rows=representatives[method])
        manifest.update(results={m: method_summary(tables[m], selections[m]) for m in METHODS},
                        comparison=comparison_summary(tables, genes, representatives, target.var_names.tolist()),
                        representatives={m: target.obs_names[representatives[m]].tolist() for m in METHODS}, grn=grn_report,
                        gene_failures={m: summaries[m][summaries[m].status != 'ok'].to_dict('records') if len(summaries[m]) else [] for m in METHODS},
                        umap_semantics='original input cell UMAP positions; surface points y have NOT been embedded',
                        units='arbitrary model-rate units; no calibrated biological time',
                        status_meanings={'not_sampled': 'not calculated', 'scalar_ineligible': 'excluded from median and derivatives',
                            'surface_undefined': 'vanishing gradient', 'retraction_failed': 'no accepted surface point',
                            'derivative_failed': 'surface reached but derivative calculation failed', 'ok': 'overview estimate available'})
        output_data.obs['projection_source_row'] = target_rows
        output_data.uns['surface_divergence_analysis'] = {'metadata_json': json.dumps(json_value(manifest), ensure_ascii=False)}
        output_data.write_h5ad(output/'annotated_divergence.h5ad')
        pd.DataFrame(grn_rows).to_csv(output/'grn_sensitivities.csv', index=False)
        # Numeric group summaries avoid extracting values from plots.
        class_rows = []
        for method in METHODS:
            for label in classes:
                subset = tables[method][(tables[method].celltype == label) & (tables[method].status == 'ok')]
                class_rows.append(dict(method=method, celltype=label, n=len(subset),
                    **{f'{term}_{stat}': float(getattr(subset[term], stat)())
                       for term in ('surface_divergence', 'jacobian_term', 'curvature_term') for stat in ('mean', 'median')}))
        pd.DataFrame(class_rows).to_csv(output/'class_summary.csv', index=False)
        manifest['completed_stages'].append('save_results')
        # 8. Exactly five consolidated PNGs; replot reads only the saved files.
        manifest['stage'] = 'plot_five_figures'; write_json(manifest_path, manifest)
        from analysis.plotting import plot_saved
        plot_saved(output, top_genes=args.top_genes)
        manifest['completed_stages'].append('plot_five_figures')
        failed = any((tables[m].sampled & tables[m].eligible & (tables[m].status != 'ok')).any()
                     or len(manifest['gene_failures'][m]) for m in METHODS)
        manifest.update(status='completed_with_cell_failures' if failed else 'completed', stage='complete')
        manifest['artifacts_sha256'] = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != 'manifest.json'}
        write_json(manifest_path, manifest)
        print(json.dumps(json_value(manifest['results']), indent=2), flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        failure_path = output/'failed_resume.json' if previous_manifest_exists and not resume_identity_verified else manifest_path
        write_json(failure_path, manifest)
        raise


if __name__ == '__main__':
    main()
