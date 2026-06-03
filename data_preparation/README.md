# データ準備

このディレクトリには、scDiffusionの訓練用データを準備するためのファイルが含まれています。

## ファイル

- `data_preparation.ipynb`: Pancreasデータセットを準備するJupyter Notebook

## 使用方法

### Jupyter Notebookで実行
```bash
# scDiffusionディレクトリで実行
conda activate scdiffusion
jupyter notebook data_preparation/data_preparation.ipynb
```

### コマンドラインで実行
```bash
# scDiffusionディレクトリで実行
conda activate scdiffusion
cd data_preparation
python -c "
import scvelo
import numpy as np
import scipy.sparse
import os
import scanpy as sc

print('Downloading pancreas dataset...')
adata = scvelo.datasets.pancreas(file_path='data/Pancreas/endocrinogenesis_day15.h5ad')

print(f'Original data shape: {adata.shape}')
print('Processing data for scDiffusion compatibility...')

# Column rename
adata.obs = adata.obs.rename(columns={'clusters': 'celltype'})

# Data type conversion
if scipy.sparse.issparse(adata.X):
    adata.X = adata.X.astype(np.float64)
else:
    adata.X = np.ascontiguousarray(adata.X, dtype=np.float64)

print(f'Final data shape: {adata.shape}')
print(f'Cell types: {len(adata.obs.celltype.unique())}')

# Save
adata.write('adata_pancreas.h5ad')
print('Data saved successfully!')

# Test load
test_adata = sc.read_h5ad('adata_pancreas.h5ad')
print(f'Test read - Shape: {test_adata.shape}')
print('✓ Data preparation complete!')
"
```

## 出力

実行後、以下のファイルが生成されます：

- `adata_pancreas.h5ad`: scDiffusion用に処理されたpancreasデータセット
  - 遺伝子数: 27,998
  - 細胞型数: 8
  - データ形式: float64, PyTorch互換
  - ファイルサイズ: 約40-50MB

## 次のステップ

データ準備完了後：

1. VAE訓練を実行
2. Diffusion backbone訓練を実行

または `train.sh` で全体パイプラインを実行
