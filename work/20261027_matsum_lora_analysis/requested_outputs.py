"""Explicit requested figure scope; paths relative to one model directory."""
import re
from pathlib import PurePosixPath

BASIS_FIGURES = {
    'basis_heatmaps_full.png', 'across_basis_std_histogram.png',
    'across_basis_std_over_mean_histogram.png', 'ordinary_spectrum_full.png',
    'ordinary_low_eigenvalue_rmsd.png', 'ordinary_eigenvector_product_frobenius.png',
    'signed_matrix_cosine_distance.png',
}


def requested_figure(relative):
    parts = PurePosixPath(relative).parts
    if len(parts) == 2 and parts[0] == 'basis':
        return parts[1] in BASIS_FIGURES
    if len(parts) < 3 or not re.fullmatch(r't\d+', parts[0]):
        return False
    path = '/'.join(parts[1:])
    patterns = [
        r'coefficients/(signed_mean|mean_absolute|mean_cell_composition)\.png',
        r'coefficients/groups/[^/]+/signed_histograms\.png',
        r'coefficients/pairs_balanced/a[1-8]_a[1-8]\.png',
        r'coefficients/embeddings/signed/(pca_celltype|umap_celltype|cluster_k\d+_counts)\.png',
        r'cell_matrices/groups/[^/]+/(mean_C_full|C_std_histogram|C_std_over_mean_histogram)\.png',
        r'cell_matrices/sampled_graphs/cell_\d+_C_ordinary\.png',
    ]
    return any(re.fullmatch(pattern, path) for pattern in patterns)


def obsolete_data(relative):
    """Only known former generated artifacts; unknown/user files are preserved."""
    path = PurePosixPath(relative)
    name = path.name
    if 'embeddings' in path.parts and any(p in path.parts for p in ('signed_L1', 'absolute_L1')):
        return name in {'coordinates_and_clusters.csv.gz', 'clustering_metrics.csv', 'definition.json',
                        'pca_loadings.csv', 'pca_variance.csv', 'skipped.json'} or bool(re.fullmatch(r'cluster_k\d+_(counts|within_cluster|within_celltype)\.csv', name))
    if path.parts[0] == 'basis':
        return (name == 'zoom_gene_ids.csv' or
                bool(re.fullmatch(r'(ordinary|normalized)_(low_eigenvalue_rmsd|eigenvector_product_frobenius|eigenspace_similarity|eigenspace_distance)\.csv', name))
                and name not in {'ordinary_low_eigenvalue_rmsd.csv', 'ordinary_eigenvector_product_frobenius.csv'} or
                name == 'signed_matrix_cosine_similarity.csv')
    if re.fullmatch(r'cluster_k\d+_(within_cluster|within_celltype)\.csv', name):
        return True
    return name in {
        'composition_of_mean_absolute.csv', 'variance_decomposition.csv', 'variance_decomposition_within_stage.csv',
        'cell_effects.csv.gz', 'mean_effects_by_celltype.csv', 'W_variance_decomposition.npz',
        'ordinary_spectra.npz', 'normalized_spectra.npz',
    }
