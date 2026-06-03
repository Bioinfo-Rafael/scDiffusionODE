#!/usr/bin/env python3
"""
scDiffusion用 Pancreasデータ準備スクリプト
コマンドラインから直接実行可能
"""

import scvelo
import numpy as np
import scipy.sparse
import os
import scanpy as sc

def main():
    print("=== scDiffusion Data Preparation ===")
    print("Preparing pancreas dataset for scDiffusion...")
    
    # データのダウンロードと読み込み
    print("\n1. Downloading pancreas dataset...")
    adata = scvelo.datasets.pancreas(file_path='data/Pancreas/endocrinogenesis_day15.h5ad')
    print(f"   Original data shape: {adata.shape}")
    print(f"   Original data type: {type(adata.X)}")
    print(f"   Original data dtype: {adata.X.dtype}")
    
    # 基本情報表示
    print(f"\n2. Dataset Information:")
    print(f"   Number of cells: {adata.n_obs}")
    print(f"   Number of genes: {adata.n_vars}")
    print(f"   Cell types: {list(adata.obs.clusters.unique())}")
    print(f"   Number of cell types: {len(adata.obs.clusters.unique())}")
    
    # データの前処理
    print("\n3. Processing data for scDiffusion compatibility...")
    
    # カラム名を標準化
    adata.obs = adata.obs.rename(columns={"clusters": "celltype"})
    print("   ✓ Renamed 'clusters' column to 'celltype'")
    
    # データ型を統一
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.astype(np.float64)
        print("   ✓ Converted sparse matrix to float64")
    else:
        adata.X = np.ascontiguousarray(adata.X, dtype=np.float64)
        print("   ✓ Converted dense matrix to contiguous float64")
    
    # 最終確認
    print(f"\n4. Final Data Format:")
    print(f"   Data shape: {adata.shape}")
    print(f"   Data type: {type(adata.X)}")
    print(f"   Data dtype: {adata.X.dtype}")
    print(f"   Is sparse: {scipy.sparse.issparse(adata.X)}")
    print(f"   Cell types: {list(adata.obs.celltype.unique())}")
    print(f"   Number of cell types: {len(adata.obs.celltype.unique())}")
    
    # toarray()のテスト
    if scipy.sparse.issparse(adata.X):
        test_array = adata.X.toarray()
        print(f"   toarray() result dtype: {test_array.dtype}")
        print(f"   toarray() result contiguous: {test_array.flags.c_contiguous}")
        del test_array  # メモリ解放
    
    # データの保存
    output_file = "adata_pancreas.h5ad"
    print(f"\n5. Saving processed data to {output_file}...")
    adata.write(output_file)
    
    # ファイルサイズの確認
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"   ✓ Data saved successfully!")
    print(f"   File size: {file_size:.1f} MB")
    
    # 保存されたデータの読み込みテスト
    print(f"\n6. Testing saved data...")
    test_adata = sc.read_h5ad(output_file)
    print(f"   Test read - Shape: {test_adata.shape}")
    print(f"   Test read - Cell types: {len(test_adata.obs.celltype.unique())}")
    print(f"   Test read - Data type: {test_adata.X.dtype}")
    print("   ✓ Data loading test passed!")
    
    print(f"\n=== Data Preparation Complete! ===")
    print(f"Generated file: {output_file}")
    print(f"- Genes: {adata.n_vars:,}")
    print(f"- Cells: {adata.n_obs:,}")
    print(f"- Cell types: {len(adata.obs.celltype.unique())}")
    print(f"- Data format: {adata.X.dtype}, PyTorch compatible")
    print(f"\nNext steps:")
    print(f"1. Run VAE training: python VAE/VAE_train.py")
    print(f"2. Run diffusion training: python cell_train.py")
    print(f"3. Or run complete pipeline: ./train.sh")

if __name__ == "__main__":
    main()
