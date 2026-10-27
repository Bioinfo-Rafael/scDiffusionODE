"""Cell-level coefficients: signed/absolute means, variance, histograms, embeddings and clusters."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from artifacts import groups, write_json, log
from numerics import coefficient_moments, normalized_coefficients, sample_by_type
import plots


def coefficient_tables_and_plots(a, obs, output, cfg):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    labels = obs.celltype.astype(str).to_numpy()
    normalized, total = normalized_coefficients(a)
    table = obs.copy()
    for k in range(8):
        table[f'a{k + 1}'] = a[:, k]
        table[f'abs_fraction{k + 1}'] = np.abs(normalized[:, k])
    table['coefficient_L1'] = total
    table.to_csv(output / 'cell_coefficients.csv.gz', index_label='cell_id')
    rows, mean_signed, mean_abs, fractions, type_names = [], [], [], [], []
    signed_edges = [np.histogram_bin_edges(a[:, k], bins=cfg['histogram_bins']) for k in range(8)]
    for slug, label, indices in groups(labels):
        b = a[indices]; moments = coefficient_moments(b)
        folder = output / 'groups' / slug; folder.mkdir(parents=True, exist_ok=True)
        for k in range(8):
            rows.append(dict(group=label, scope='all' if slug == 'all' else 'celltype', n=len(indices), component=k + 1,
                             mean=moments['mean'][k], mean_absolute=moments['mean_abs'][k],
                             variance=moments['variance'][k], std=moments['std'][k],
                             positive_fraction=np.mean(b[:, k] > 0), negative_fraction=np.mean(b[:, k] < 0),
                             zero_fraction=np.mean(b[:, k] == 0)))
        plots.hist_panels(b.T, folder / 'signed_histograms.png', f'{label}: signed a; n={len(b)}',
                          reference=None if slug == 'all' else a.T, edges=signed_edges)
        if slug != 'all':
            type_names.append(label); mean_signed.append(moments['mean']); mean_abs.append(moments['mean_abs'])
            nonzero = total[indices] > 0
            fractions.append(np.abs(normalized[indices][nonzero]).mean(axis=0) if nonzero.any() else np.full(8, np.nan))
    pd.DataFrame(rows).to_csv(output / 'coefficient_statistics.csv', index=False)
    for name, values, title in [
        ('signed_mean', mean_signed, 'Mean signed coefficients (positive / negative stacks)'),
        ('mean_absolute', mean_abs, plots.WARNING),
        ('mean_cell_composition', fractions, '⚠ WARNING: mean of cell-wise |a| / L1; sign discarded')]:
        values = np.asarray(values)
        pd.DataFrame(values, index=type_names, columns=[f'a{k + 1}' for k in range(8)]).to_csv(output / f'{name}.csv')
        plots.stacked(values, type_names, output / f'{name}.png', title)
    selected = sample_by_type(labels, cfg['scatter_cells_per_type'], cfg['seed'])
    pd.DataFrame({'cell_id': obs.index[selected]}).to_csv(output / 'scatter_cell_ids.csv', index=False)
    plots.pair_scatter(a[selected], labels[selected], output / 'pairs_balanced', cfg['seed'])
    write_json(output / 'normalization.json', dict(zero_L1_cells=int(np.sum(total == 0)),
               rule='zero-L1 vectors excluded from composition averages; embeddings use signed raw a',
               variance_ddof=0, scatter_policy='up to configured count per celltype; all cells used for statistics'))


def embeddings_and_clusters(a, obs, output, cfg):
    """Unsupervised operations never use celltype to fit or choose k; labels assess results."""
    import umap
    output = Path(output)
    folder = output / 'signed'; folder.mkdir(parents=True, exist_ok=True)
    x = a; annotations = obs
    if len(x) < 4 or np.max(np.ptp(x, axis=0)) == 0:
        write_json(folder / 'skipped.json', {'reason': 'Too few cells or constant representation'})
        return
    log(f'PCA / UMAP / clustering: signed, n={len(x)}')
    pca = PCA(n_components=min(8, len(x)), svd_solver='full').fit(x)
    coordinates = pca.transform(x)
    pd.DataFrame(pca.components_.T, index=[f'a{k + 1}' for k in range(8)]).to_csv(folder / 'pca_loadings.csv')
    pd.DataFrame({'explained_variance_ratio': pca.explained_variance_ratio_}).to_csv(folder / 'pca_variance.csv', index=False)
    embedding = umap.UMAP(n_neighbors=min(cfg['umap_neighbors'], len(x) - 1), min_dist=cfg['umap_min_dist'],
                          metric='euclidean', random_state=cfg['seed'], n_jobs=1).fit_transform(x)
    frame = annotations.copy()
    for j in range(coordinates.shape[1]): frame[f'PC{j + 1}'] = coordinates[:, j]
    frame['UMAP1'], frame['UMAP2'] = embedding.T
    for label in ('celltype',):
        if label in annotations:
            plots.scatter(coordinates[:, :2], annotations[label].astype(str), folder / f'pca_{label}.png',
                          f'signed: PCA, color={label}; PC1/2={pca.explained_variance_ratio_[:2].sum():.1%}')
            plots.scatter(embedding, annotations[label].astype(str), folder / f'umap_{label}.png', f'signed: UMAP, color={label}')
    scores = []
    labels = annotations.celltype.astype(str).to_numpy()
    for count in cfg['cluster_counts']:
        if count >= len(x) or count < 2:
            continue
        first = None
        for seed in cfg['cluster_seeds']:
            cluster = KMeans(n_clusters=count, random_state=seed, n_init=10).fit_predict(x)
            if first is None: first = cluster
            # Random subset silhouette uses original feature coordinates, never UMAP.
            sample = np.random.default_rng(cfg['seed']).choice(len(x), min(len(x), cfg['silhouette_sample_size']), replace=False)
            unique = len(np.unique(cluster[sample]))
            silhouette = silhouette_score(x[sample], cluster[sample]) if 1 < unique < len(sample) else np.nan
            scores.append(dict(k=count, seed=seed, observed_clusters=len(np.unique(cluster)),
                               ARI_celltype=adjusted_rand_score(labels, cluster),
                               AMI_celltype=adjusted_mutual_info_score(labels, cluster),
                               ARI_first_seed=adjusted_rand_score(first, cluster), silhouette=silhouette))
            frame[f'cluster_k{count}_seed{seed}'] = cluster
        counts = pd.crosstab(pd.Series(first, name='cluster'), pd.Series(labels, name='celltype'))
        for suffix, table in [('counts', counts)]:
            table.to_csv(folder / f'cluster_k{count}_{suffix}.csv')
            plots.heatmap(table.to_numpy(), folder / f'cluster_k{count}_{suffix}.png', f'k={count}: {suffix}',
                          labels=(table.index, table.columns), signed=False)
    frame.to_csv(folder / 'coordinates_and_clusters.csv.gz', index_label='cell_id')
    pd.DataFrame(scores).to_csv(folder / 'clustering_metrics.csv', index=False)
    write_json(folder / 'definition.json', dict(centered_PCA=True, standardized_components=False,
               clustering_space='original eight-dimensional representation', cluster_selection='all predeclared k; no label-based selection',
               n=len(x), exclusions=0))


def run_coefficient_analysis(a, adata, output, cfg):
    coefficient_tables_and_plots(a, adata.obs, output, cfg)
    embeddings_and_clusters(a, adata.obs, Path(output) / 'embeddings', cfg)
