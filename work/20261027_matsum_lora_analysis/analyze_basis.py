"""Eight shared matrices: entry distributions, graph spectra and pairwise comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import write_json, log
from numerics import relative_std, gene_strength
from spectral import spectrum, spectral_comparisons, matrix_cosines
import plots


def run_basis_analysis(basis, base, genes, output, cfg):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'shared_matrices.npz', B=basis, W0=base, gene_ids=np.asarray(genes, dtype=str))
    limit = max(np.abs(basis).max(), 1e-15)
    importance = np.abs(basis).sum(axis=(0, 1)) + np.abs(basis).sum(axis=(0, 2))
    selected = np.sort(np.argsort(-importance, kind='stable')[:cfg['heatmap_top_genes']])
    pd.DataFrame({'index': selected, 'gene_id': np.asarray(genes)[selected]}).to_csv(output / 'zoom_gene_ids.csv', index=False)
    plots.matrix_panels(basis, output / 'basis_heatmaps_full.png', 'Shared B_k (target rows, source columns)', limit)
    plots.matrix_panels(basis[:, selected][:, :, selected], output / 'basis_heatmaps_zoom.png',
                        'Shared B_k; same selected genes/order for all eight', limit, np.asarray(genes)[selected])
    plots.heatmap(base, output / 'W0_heatmap.png', 'Shared W0 (zero for MatSum)')
    mean, std = basis.mean(axis=0), basis.std(axis=0)
    ratio, valid, threshold = relative_std(mean, std, cfg['near_zero_mean_relative'], cfg['near_zero_mean_absolute'])
    np.savez_compressed(output / 'across_eight_basis_statistics.npz', mean=mean, std=std, std_over_mean=ratio,
                        valid_ratio=valid, gene_ids=np.asarray(genes, dtype=str))
    for name, value in [('mean', mean), ('std', std), ('std_over_mean', ratio), ('std_over_abs_mean', np.abs(ratio))]:
        plots.histogram(value, output / f'across_basis_{name}_histogram.png', f'Across 8 B_k: {name}; d*d entries', cfg['histogram_bins'])
    write_json(output / 'across_basis_definition.json', dict(axis='eight shared matrices, not cells', ddof=0,
               entries=int(mean.size), diagonal_included=True, ratio_threshold=threshold,
               undefined_ratio_entries=int((~valid).sum()), warning='Basis coordinate statistics depend on basis parameterization'))
    rows, spectra = [], []
    for k, matrix in enumerate(basis):
        log(f'Shared basis B{k + 1}: Laplacian eigendecomposition')
        item = spectrum(matrix, cfg['spectral_dimensions']); spectra.append(item)
        np.savez_compressed(output / f'B{k + 1}_graph.npz', adjacency=item['adjacency'], degree=item['degree'],
                            ordinary_eigenvalues=item['ordinary']['eigenvalues'], normalized_eigenvalues=item['normalized']['eigenvalues'],
                            ordinary_low_eigenvectors=item['ordinary']['eigenvectors'], normalized_low_eigenvectors=item['normalized']['eigenvectors'])
        incoming, outgoing = gene_strength(matrix)
        pd.DataFrame({'gene_id': genes, 'incoming': incoming, 'outgoing': outgoing,
                      'incoming_rank': pd.Series(incoming).rank(ascending=False, method='min'),
                      'outgoing_rank': pd.Series(outgoing).rank(ascending=False, method='min')}).sort_values('outgoing', ascending=False).to_csv(output / f'B{k + 1}_gene_ranking.csv', index=False)
        for variant in ('ordinary', 'normalized'):
            info = item[variant]
            rows.append(dict(component=k + 1, laplacian=variant, frobenius_norm=float(np.linalg.norm(matrix)),
                             cutoff_gap=info['cutoff_gap'], cutoff_degenerate=info['cutoff_degenerate'],
                             zero_modes=info['zero_modes'], zero_mode_tolerance=info['zero_mode_tolerance'],
                             isolated_nodes=info['isolated_nodes']))
    pd.DataFrame(rows).to_csv(output / 'spectral_diagnostics.csv', index=False)
    labels = [f'B{k + 1}' for k in range(8)]
    for variant in ('ordinary', 'normalized'):
        values = np.stack([item[variant]['eigenvalues'] for item in spectra])
        vectors = np.stack([item[variant]['eigenvectors'] for item in spectra])
        plots.spectra(values, output / f'{variant}_spectrum_full.png', f'{variant} Laplacian spectrum')
        plots.spectra(values[:, :cfg['spectral_dimensions']], output / f'{variant}_spectrum_low.png', f'{variant}: low eigenvalues')
        comparisons = spectral_comparisons(values, vectors, cfg['spectral_dimensions'])
        for name, matrix in comparisons.items():
            pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output / f'{variant}_{name}.csv')
            plots.heatmap(matrix, output / f'{variant}_{name}.png', f'{variant}: {name}', labels=(labels, labels), signed=False)
    cosine, distance = matrix_cosines(basis)
    for name, matrix in [('signed_matrix_cosine_similarity', cosine), ('signed_matrix_cosine_distance', distance)]:
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output / f'{name}.csv')
        plots.heatmap(matrix, output / f'{name}.png', name, labels=(labels, labels), signed=name.endswith('similarity'))
    return selected
