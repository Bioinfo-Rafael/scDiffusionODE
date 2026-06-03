import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch
import sys
sys.path.append('..')
from VAE.VAE_model import VAE
from sklearn.preprocessing import LabelEncoder

def custom_collate_fn(batch):
    """Custom collate function to handle data type conversion."""
    data = [item[0] for item in batch]
    dicts = [item[1] for item in batch]
    
    # Convert to float32 tensor
    data_tensor = torch.stack([torch.tensor(d, dtype=torch.float32) for d in data])
    
    # Handle dict items
    if dicts and dicts[0]:  # if there are any dict items
        combined_dict = {}
        for key in dicts[0].keys():
            values = [d[key] for d in dicts]
            if key == "y":
                # Handle scalar values properly
                if isinstance(values[0], np.ndarray):
                    if values[0].ndim == 0:  # scalar array
                        values = [int(v.item()) for v in values]
                    else:
                        values = [int(v) for v in values]
                elif isinstance(values[0], (int, np.integer)):
                    values = [int(v) for v in values]
                combined_dict[key] = torch.tensor(values, dtype=torch.long)
            else:
                combined_dict[key] = torch.tensor(values)
    else:
        combined_dict = {}
    
    return data_tensor, combined_dict

def stabilize(expression_matrix):
    ''' Use Anscombes approximation to variance stabilize Negative Binomial data
    See https://f1000research.com/posters/4-1041 for motivation.
    Assumes columns are samples, and rows are genes
    '''
    from scipy import optimize
    phi_hat, _ = optimize.curve_fit(lambda mu, phi: mu + phi * mu ** 2, expression_matrix.mean(1), expression_matrix.var(1))

    return np.log(expression_matrix + 1. / (2 * phi_hat[0]))

def load_VAE(vae_path, num_gene, hidden_dim):
    # Detect device automatically
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    autoencoder = VAE(
        num_genes=num_gene,
        device=device,
        seed=0,
        loss_ae='mse',
        hidden_dim=hidden_dim,
        decoder_activation='ReLU',
    )
    
    # Load state dict with proper device mapping
    if device == 'cpu':
        state_dict = torch.load(vae_path, map_location='cpu')
    else:
        state_dict = torch.load(vae_path)
    
    autoencoder.load_state_dict(state_dict)
    return autoencoder


def load_data(
    *,
    data_dir,
    batch_size,
    vae_path=None,
    deterministic=False,
    train_vae=False,
    hidden_dim=128,
    preprocess=True
):
    """
    For a dataset, create a generator over (cells, kwargs) pairs.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param vae_path: the path to save autoencoder / read autoencoder checkpoint.
    :param deterministic: if True, yield results in a deterministic order.
    :param train_vae: train the autoencoder or use the autoencoder.
    :param hidden_dim: the dimensions of latent space. If use pretrained weight, set 128
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    # Auto-detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    adata = sc.read_h5ad(data_dir)

    if preprocess:    
    # preporcess the data. modify this part if use your own dataset. the gene expression must first norm1e4 then log1p
    # Skip gene filtering to maintain consistent gene count between VAE training and diffusion training
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.filter_cells(adata, min_genes=10)
        adata.var_names_make_unique()

            # Check if data is already preprocessed (avoid double normalization)
        if '_preprocessed_' not in data_dir:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        else:
            print("Data appears to be already preprocessed, skipping normalization")

    # if generate ood data, left this as the ood data
    # selected_cells = (adata.obs['organ'] != 'mammary') | (adata.obs['celltype'] != 'B cell')  
    # adata = adata[selected_cells, :]  

    classes = adata.obs['celltype'].values
    label_encoder = LabelEncoder()
    labels = classes
    label_encoder.fit(labels)
    classes = label_encoder.transform(labels)



    # Handle both sparse and dense matrices
    if hasattr(adata.X, 'toarray'):
        cell_data = adata.X.toarray()
    else:
        cell_data = adata.X

    # turn the gene expression into latent space. use this if training the diffusion backbone.
    if not train_vae:
        num_gene = cell_data.shape[1]
        autoencoder = load_VAE(vae_path, num_gene, hidden_dim)
        
        # Detect device and convert data accordingly
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cell_tensor = torch.tensor(cell_data, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            cell_data = autoencoder(cell_tensor, return_latent=True)
            cell_data = cell_data.cpu().detach().numpy().astype(np.float32)
    
    dataset = CellDataset(
        cell_data,
        classes
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, 
            drop_last=True, collate_fn=custom_collate_fn
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, 
            drop_last=True, collate_fn=custom_collate_fn
        )
    while True:
        yield from loader


class CellDataset(Dataset):
    def __init__(
        self,
        cell_data,
        class_name
    ):
        super().__init__()
        self.data = cell_data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx].astype(np.float32)  # Ensure float32 type
        out_dict = {}
        if self.class_name is not None:
            out_dict["y"] = np.array(self.class_name[idx], dtype=np.int64)
        return arr, out_dict

