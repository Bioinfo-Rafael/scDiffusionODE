"""Exactly five PNG overviews, reconstructed from saved results without model/Jacobian calls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from .diagnostics import METHODS

FIGURES = [
    ('figure1_analysis_overview.png', 'Analysis overview: population, scalars, median surfaces, success and divergence'),
    ('figure2_surface_divergence.png', 'Surface divergence on original-cell UMAP and class distributions'),
    ('figure3_divergence_decomposition.png', 'Jacobian and curvature terms, class summaries and residuals'),
    ('figure4_gene_contributions.png', 'Exact representative-cell gene-coordinate contributions'),
    ('figure5_method_comparison.png', 'Matched-cell comparison, gene ranking agreement and numerical reliability'),
]
COLORS = {'hodge': '#2c6eac', 'cellrank': '#d26939'}
TERMS = {'surface_divergence': ('$D_S$', '#222222'), 'jacobian_term': ('$D_J$', '#3978b8'),
         'curvature_term': ('$D_C$', '#d98528')}
SHORT = ['BP1', 'BP2', 'Ery1', 'Ery2', 'Ery3']
FOOTER = ('BP1: Blood progenitors 1   BP2: Blood progenitors 2   Ery1: Erythroid1   Ery2: Erythroid2   Ery3: Erythroid3\n'
          'Map coordinates are ORIGINAL input-cell UMAP, not embeddings of surface points y. '
          'Divergence/contribution: arbitrary model-rate units; no biological time calibration.')


def note(ax, text):
    ax.text(.5, .5, text, ha='center', va='center', transform=ax.transAxes, wrap=True, fontsize=10)
    ax.set_axis_off()


def finish(fig, output, index, kind, title):
    fig.suptitle(f'{"SYNTHETIC VALIDATION — " if kind == "synthetic" else "D20 EMA, t=49 — "}{title}', fontsize=17)
    fig.text(.02, .015, FOOTER, va='bottom', fontsize=9)
    fig.subplots_adjust(left=.07, right=.97, top=.91, bottom=.10, wspace=.38, hspace=.48)
    fig.savefig(output/FIGURES[index][0], dpi=160, facecolor='white')
    plt.close(fig)


def scatter_map(fig, ax, xy, values, title, *, norm=None, cmap='viridis', mask=None, units='scalar (a.u.)'):
    ax.set_title(title, fontsize=12)
    if xy is None:
        note(ax, 'No saved original-cell UMAP\nNo embedding was recomputed')
        return
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c='#dddddd', rasterized=True)
    valid = np.isfinite(values)
    if mask is not None: valid &= mask
    if valid.any():
        art = ax.scatter(xy[valid, 0], xy[valid, 1], c=np.asarray(values)[valid], s=12,
                         cmap=cmap, norm=norm, linewidths=0, rasterized=True)
        fig.colorbar(art, ax=ax, shrink=.75, label=units)
    ax.set_xlabel('Original UMAP 1'); ax.set_ylabel('Original UMAP 2')
    ax.text(.02, .02, f'colored n={int(valid.sum())}; gray=unavailable/not sampled', transform=ax.transAxes, fontsize=8)


def class_distribution(ax, table, classes, *, metric='surface_divergence'):
    for i, label in enumerate(classes):
        values = table.loc[(table.celltype == label) & (table.status == 'ok'), metric].dropna().to_numpy()
        if len(values):
            ax.boxplot([values], positions=[i], widths=.55, showfliers=False, manage_ticks=False)
            ax.scatter(np.full(len(values), i), values, s=8, alpha=.35)
        ax.text(i, .98, f'n={len(values)}', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)
    ax.axhline(0, c='gray', lw=.8)
    ax.set_xticks(range(5), SHORT); ax.set_xlabel('Cell class'); ax.set_ylabel('Surface divergence (model-rate units)')


def plot_saved(directory, top_genes=20):
    output = Path(directory)
    manifest = json.loads((output/'manifest.json').read_text())
    a = ad.read_h5ad(output/'annotated_divergence.h5ad')
    classes = list(manifest['class_counts'])
    # Follow the explicit benchmark order rather than frequency-sorted class_counts.
    classes = ['Blood progenitors 1', 'Blood progenitors 2', 'Erythroid1', 'Erythroid2', 'Erythroid3']
    labels = a.obs.celltype.astype(str).to_numpy()
    xy_key = next((key for key in ('X_umap_source', 'X_umap') if key in a.obsm), None)
    xy = np.asarray(a.obsm[xy_key])[:, :2] if xy_key else None
    tables = {m: pd.read_csv(output/f'{m}_overview.csv', index_col=0).reindex(a.obs_names) for m in METHODS}
    genes = {}
    for method in METHODS:
        with np.load(output/f'{method}_gene_contributions.npz', allow_pickle=False) as data:
            genes[method] = dict(data)
    existing = [p for p in output.iterdir() if p.suffix.lower() in ('.png', '.pdf', '.svg')]
    if any(p.name not in {f[0] for f in FIGURES} for p in existing):
        raise ValueError('Unexpected figure artifacts: keep this run to the five declared PNGs')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    kind = manifest['data_kind']
    populations = f'target N={len(a)}; overview sampled={int(tables["hodge"].sampled.sum())}'
    valid = {m: tables[m][tables[m].status == 'ok'] for m in METHODS}
    all_ds = np.concatenate([v.surface_divergence.dropna().to_numpy() for v in valid.values()])
    vmax = max(float(np.max(abs(all_ds))) if len(all_ds) else 0, 1e-9)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # 1. Broad population overview, before any gene-specific detail.
    fig, ax = plt.subplots(3, 3, figsize=(19, 14))
    if xy is None: note(ax[0, 0], 'Original UMAP absent')
    else:
        palette = plt.get_cmap('tab10')
        for i, label in enumerate(classes):
            rows = labels == label
            ax[0, 0].scatter(xy[rows, 0], xy[rows, 1], s=9, color=palette(i), label=f'{label} (n={rows.sum()})')
        ax[0, 0].legend(fontsize=8, loc='best')
        ax[0, 0].set_xlabel('Original UMAP 1'); ax[0, 0].set_ylabel('Original UMAP 2')
        ax[0, 0].set_title('Five analysis classes')
    for j, method in enumerate(METHODS):
        table = tables[method]; select = manifest['selections'][method]
        scatter_map(fig, ax[0, j+1], xy, table.scalar.to_numpy(),
                    f'{method}: continuous scalar; eligible n={select["eligible_count"]}',
                    mask=table.eligible.to_numpy(bool), units='U (a.u.)' if method == 'hodge' else 'tau (dimensionless)')
        values = table.loc[table.eligible, 'scalar'].to_numpy()
        ax[1, j].hist(values, bins=25, color=COLORS[method], alpha=.8, label=f'valid n={len(values)}')
        ax[1, j].axvline(select['c'], color='black', ls='--', label=f'median c={select["c"]:.5g}')
        ax[1, j].set_title(f'{method}: all eligible target cells')
        ax[1, j].set_xlabel('Continuous scalar (a.u.; CellRank dimensionless)'); ax[1, j].set_ylabel('Cells'); ax[1, j].legend(fontsize=9)
        if len(valid[method]):
            ax[2, 0].hist(valid[method].surface_divergence, bins=20, histtype='step', linewidth=2,
                           color=COLORS[method], label=f'{method} n={len(valid[method])}')
    rates = [manifest['results'][m]['retraction_success_rate'] or 0 for m in METHODS]
    ax[1, 2].bar(METHODS, rates, color=[COLORS[m] for m in METHODS]); ax[1, 2].set_ylim(0, 1.12)
    for i, method in enumerate(METHODS):
        r = manifest['results'][method]
        ax[1, 2].text(i, rates[i]+.03, f'{r["retraction_success_count"]}/{r["attempted_retractions"]}', ha='center')
    ax[1, 2].set_title('Retraction success / attempted eligible sample'); ax[1, 2].set_ylabel('Fraction')
    ax[2, 0].axvline(0, c='gray', lw=.8); ax[2, 0].legend(fontsize=9)
    ax[2, 0].set_title('Surface divergence: successful sampled cells')
    ax[2, 0].set_xlabel('Surface divergence (model-rate units)'); ax[2, 0].set_ylabel('Cells')
    text = ['Same initial-cell sample for both methods.', 'Each c = median over ALL eligible target cells.',
            'Scalar fit population: '+str(manifest.get('scalar_population', manifest.get('full_scalar_population'))),
            'CellRank terminal: all Erythroid3 cells.',
            'Overview: JVP Hutchinson trace + analytic curvature.',
            'Detailed genes: exact streamed coordinate derivatives.']
    note(ax[2, 1], '\n'.join(text))
    note(ax[2, 2], f'{populations}\nCommon divergence successes: {manifest["comparison"]["common_success_count"]}\n'
         'Failure cells are retained in CSV/h5ad.\nSuccess-conditioned summaries are not population-wide conclusions.\n'
         + ('All cells/genes are artificial test data.' if kind == 'synthetic' else 'No biological causal interpretation.'))
    finish(fig, output, 0, kind, 'Figure 1 | Analysis overview — '+populations)

    # 2. Common zero-centered scale, method-specific surface points placed on ORIGINAL UMAP.
    fig, ax = plt.subplots(2, 3, figsize=(19, 11))
    for row, method in enumerate(METHODS):
        table = tables[method]
        scatter_map(fig, ax[row, 0], xy, table.surface_divergence.to_numpy(),
                    f'{method}: D_S at y; original x locations', norm=norm, cmap='RdBu_r', units='model-rate units')
        class_distribution(ax[row, 1], table, classes)
        ax[row, 1].set_title(f'{method}: success-conditioned distributions')
        heights, counts = [], []
        for label in classes:
            attempted = table[(table.celltype == label) & table.sampled & table.eligible]
            counts.append(len(attempted)); heights.append(float((attempted.status != 'ok').mean()) if len(attempted) else np.nan)
        ax[row, 2].bar(SHORT, heights, color=COLORS[method]); ax[row, 2].set_ylim(0, 1.1)
        for i, count in enumerate(counts): ax[row, 2].text(i, .98, f'n={count}', ha='center', fontsize=8)
        ax[row, 2].set_title(f'{method}: retraction/derivative failure fraction')
        ax[row, 2].set_ylabel('Failed / attempted eligible sample'); ax[row, 2].set_xlabel('Cell class')
    finish(fig, output, 1, kind, 'Figure 2 | Surface divergence — shared symmetric color scale; '+populations)

    # 3. Decomposition, including a direct directional-derivative residual (not only arithmetic sums).
    fig, ax = plt.subplots(2, 3, figsize=(20, 11))
    for row, method in enumerate(METHODS):
        table = valid[method]
        for k, (term, (label, color)) in enumerate(TERMS.items()):
            if len(table): ax[row, 0].hist(table[term], bins=22, histtype='step', color=color, lw=1.8, label=label)
            for i, celltype in enumerate(classes):
                values = table.loc[table.celltype == celltype, term]
                if len(values):
                    ax[row, 1].scatter(i+(k-1)*.17, values.mean(), color=color, marker='o', label=f'{label} mean' if i == 0 else None)
                    ax[row, 1].scatter(i+(k-1)*.17, values.median(), facecolors='none', edgecolors=color, marker='s', label=f'{label} median' if i == 0 else None)
        ax[row, 0].axvline(0, c='gray', lw=.8); ax[row, 0].legend(fontsize=9)
        ax[row, 0].set_title(f'{method}: component distributions (n={len(table)})')
        ax[row, 0].set_xlabel('Component (model-rate units)'); ax[row, 0].set_ylabel('Cells')
        ax[row, 1].set_xticks(range(5), SHORT); ax[row, 1].axhline(0, c='gray', lw=.8)
        ax[row, 1].set_title(f'{method}: per-class mean / median'); ax[row, 1].set_ylabel('Component (model-rate units)')
        ax[row, 1].legend(fontsize=7, ncol=2)
        if len(table):
            residual = np.maximum(table.additive_residual.to_numpy(), 1e-18)
            ax[row, 2].scatter(np.arange(len(residual)), residual, s=12, color=COLORS[method], label='directional identity residual')
            ax[row, 2].set_yscale('log'); ax[row, 2].legend(fontsize=8)
        ax[row, 2].set_title(f'{method}: max per-probe |direct - J - C|')
        ax[row, 2].set_xlabel('Successful sampled cell index'); ax[row, 2].set_ylabel('Absolute error (model-rate units)')
    finish(fig, output, 2, kind, 'Figure 3 | D_S = D_J + D_C — stochastic J, analytic curvature of fitted scalar')

    # 4. Exact gene coordinates only, with signed and absolute summaries explicitly separated.
    fig, ax = plt.subplots(2, 3, figsize=(22, 17))
    for row, method in enumerate(METHODS):
        data = genes[method]
        if not len(data['total']):
            for axis in ax[row]: note(axis, f'{method}: no successful representative cells')
            continue
        signed, absolute = data['total'].mean(0), abs(data['total']).mean(0)
        order = np.argsort(-absolute, kind='stable')[:min(top_genes, len(signed))]
        names = data['gene_ids'][order]; y = np.arange(len(order))
        ax[row, 0].barh(y, signed[order], color=np.where(signed[order] < 0, '#387cb0', '#c65c43'))
        ax[row, 1].barh(y, absolute[order], color=COLORS[method])
        for column in (0, 1):
            ax[row, column].set_yticks(y, names, fontsize=8); ax[row, column].invert_yaxis()
            ax[row, column].set_xlabel('Gene-coordinate contribution (model-rate units)')
        ax[row, 0].axvline(0, color='gray', lw=.8)
        ax[row, 0].set_title(f'{method}: signed mean; exact reps n={len(data["total"])}')
        ax[row, 1].set_title(f'{method}: mean absolute contribution; top {len(order)}')
        heat = np.full((len(order), len(classes)), np.nan)
        for i, celltype in enumerate(classes):
            mask = data['celltypes'] == celltype
            if mask.any(): heat[:, i] = data['total'][mask][:, order].mean(0)
        cap = max(float(np.nanmax(abs(heat))), 1e-12)
        art = ax[row, 2].imshow(heat, aspect='auto', cmap='RdBu_r', vmin=-cap, vmax=cap)
        ax[row, 2].set_yticks(y, names, fontsize=8); ax[row, 2].set_xticks(range(5), SHORT)
        ax[row, 2].set_title(f'{method}: class mean on exact reps only')
        fig.colorbar(art, ax=ax[row, 2], shrink=.6, label='Signed contribution (model-rate units)')
    finish(fig, output, 3, kind, 'Figure 4 | Gene-coordinate contributions — coordinate dependent, not causal gene effects')

    # 5. Same original-cell IDs, and matched representative IDs for ranking comparisons.
    fig, ax = plt.subplots(2, 2, figsize=(18, 13))
    comp = manifest['comparison']; common = (tables['hodge'].status == 'ok') & (tables['cellrank'].status == 'ok')
    if common.any():
        ax[0, 0].scatter(tables['hodge'].loc[common, 'surface_divergence'], tables['cellrank'].loc[common, 'surface_divergence'], s=20, alpha=.7)
    ax[0, 0].axhline(0, color='gray', lw=.8); ax[0, 0].axvline(0, color='gray', lw=.8)
    ax[0, 0].set_xlabel('Hodge D_S (model-rate units)'); ax[0, 0].set_ylabel('CellRank D_S (model-rate units)')
    def fmt(value): return 'undefined' if value is None else f'{value:.3g}'
    ax[0, 0].set_title(f'Common original cells n={comp["common_success_count"]}; different y\nPearson={fmt(comp.get("pearson"))}, Spearman={fmt(comp.get("spearman"))}')
    shared_ids = sorted(set(genes['hodge']['cell_ids']) & set(genes['cellrank']['cell_ids']))
    if shared_ids:
        means = {}
        for method in METHODS:
            rows = [list(genes[method]['cell_ids']).index(cell) for cell in shared_ids]
            means[method] = abs(genes[method]['total'][rows]).mean(0)
        from scipy.stats import rankdata
        ax[0, 1].scatter(rankdata(-means['hodge']), rankdata(-means['cellrank']), s=12, alpha=.5)
        ax[0, 1].set_xlabel('Hodge mean-|contribution| rank'); ax[0, 1].set_ylabel('CellRank mean-|contribution| rank')
    else: note(ax[0, 1], 'No shared exact representative cells; ranking comparison omitted')
    ax[0, 1].set_title(f'Ranking over {len(a.var_names)} gene coordinates; matched reps n={len(shared_ids)}\n'
                       f'Spearman={fmt(comp.get("gene_rank_spearman"))}; top-k overlap={comp.get("top_k_overlap", "NA")}/{comp.get("top_k", "NA")}'
                       + (' (all genes: trivial)' if comp.get('top_k') == len(a.var_names) else ''))
    grn = manifest['grn']
    if grn['status'] == 'verified_candidates_only' and (output/'grn_sensitivities.csv').stat().st_size > 1:
        frame = pd.read_csv(output/'grn_sensitivities.csv').sort_values('mean_absolute_sensitivity', ascending=False).head(10)
        rows = [[r.method, f'{r.source} -> {r.target}', f'{r.mean_sensitivity:.3g}', f'{r.mean_absolute_sensitivity:.3g}'] for r in frame.itertuples()]
        ax[1, 0].axis('off')
        table_art = ax[1, 0].table(cellText=rows, colLabels=['Method', 'Candidate edge', 'Mean J', 'Mean |J|'], loc='center', cellLoc='left')
        table_art.auto_set_font_size(False); table_art.set_fontsize(8); table_art.scale(1, 1.7)
        ax[1, 0].set_title('Verified GRN candidates: local sensitivities, not causal strengths')
    else:
        for method in METHODS:
            reps = pd.read_csv(output/f'{method}_representatives.csv')
            if len(reps) and 'overview_minus_exact' in reps:
                reps = reps[reps.status == 'ok']
                ax[1, 0].errorbar(np.arange(len(reps)), reps.overview_minus_exact,
                    yerr=2*reps.overview_standard_error, fmt='o', capsize=3, label=method, alpha=.75)
        ax[1, 0].axhline(0, color='gray', lw=.8); ax[1, 0].legend(fontsize=9)
        ax[1, 0].set_xlabel('Exact representative index'); ax[1, 0].set_ylabel('Overview - exact D_S (model-rate units)')
        ax[1, 0].set_title('GRN unavailable: numerical validation instead\nError bars = 2 x probe SE (not interpolation uncertainty)')
    lines = ['Precision / interpretation audit']
    for method in METHODS:
        r = manifest['results'][method]
        rmse = manifest['scalar_interpolation'][method]['metrics']['validation_nrmse']
        residual = float(valid[method].additive_residual.max()) if len(valid[method]) else np.nan
        lines.extend([f'{method}: retraction {r["retraction_success_count"]}/{r["attempted_retractions"]}; divergence n={r["divergence_success_count"]}',
                      f'  scalar holdout NRMSE={rmse:.3g}; directional residual max={residual:.2g}'])
    lines.extend(['Overview trace SE describes probe uncertainty only.', 'Curvature is of the fitted continuous scalar.',
                  'Correlation pair was pre-specified (1 candidate pair).',
                  'Same F/cells/formula: shared mathematics can induce agreement.',
                  'Exact gene reps and overview samples have different precision.',
                  'Numerical convergence does not establish biological restoring force.'])
    note(ax[1, 1], '\n'.join(lines))
    finish(fig, output, 4, kind, 'Figure 5 | Matched comparisons and reliability — '+populations)
    if len(list(output.glob('figure*.png'))) != 5: raise RuntimeError('Expected exactly five figures')
    (output/'FIGURES.md').write_text('Start with Figure 1. '+('SYNTHETIC validation only; no real-cell conclusions.\n\n' if kind == 'synthetic' else '\n\n')+
        '\n'.join(f'{i+1}. [{name}]({name}) — {caption}' for i, (name, caption) in enumerate(FIGURES))+
        '\n\nOnly PNG is produced: five completed figures, not five per method or class.\n')
    return [output/name for name, _ in FIGURES]
