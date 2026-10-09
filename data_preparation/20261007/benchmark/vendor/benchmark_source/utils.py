import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import pairwise_distances

def split_anndata_stratified(adata, n_splits=5, cluster_key='clusters', random_state=42):
    """
    Split AnnData object into n_splits stratified folds while maintaining cell type proportions.
    
    Parameters:
    adata: AnnData object with celltype information in adata.obs['celltype']
    n_splits: Number of folds (default: 5)
    random_state: Random seed for reproducibility
    
    Returns:
    List of n_splits AnnData objects
    """
    # Get cell type labels
    labels = adata.obs[cluster_key].values
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Store split AnnData objects
    split_adatas = []
    
    # Get indices for each fold
    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        # Create new AnnData object for this fold
        sub_adata = adata[test_idx].copy()
        split_adatas.append(sub_adata)
    
    return split_adatas

def _get_indices_distances_from_dense_matrix(D, n_neighbors: int):
    sample_range = np.arange(D.shape[0])[:, None]
    indices = np.argpartition(D, n_neighbors - 1, axis=1)[:, :n_neighbors]
    indices = indices[sample_range, np.argsort(D[sample_range, indices])]
    distances = D[sample_range, indices]
    return indices, distances

def fill_in_neighbors_indices(adata):
    X = adata.obsm['X_pca']
    _distances = pairwise_distances(X, metric='euclidean')
    knn_indices, _ = _get_indices_distances_from_dense_matrix(_distances, n_neighbors=30)
    adata.uns["neighbors"]["indices"] = knn_indices

