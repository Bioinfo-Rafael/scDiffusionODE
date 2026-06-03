#!/usr/bin/env python3
"""
PBMC68k データからAnnDataオブジェクトを作成し、h5adファイルとして保存するスクリプト

このスクリプトは以下のファイルを読み込み、AnnDataオブジェクトを作成します：
- hg19/barcodes.tsv: セルのバーコード
- hg19/genes.tsv: 遺伝子ID と遺伝子名
- hg19/matrix.mtx: スパース行列（遺伝子発現データ）
- ../68k_pbmc_barcodes_annotation.tsv: セルタイプアノテーション
"""

import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from scipy import sparse
from scipy.io import mmread
import os

def create_anndata_from_pbmc68k():
    """
    PBMC68k データからAnnDataオブジェクトを作成する
    """
    print("PBMC68k データからAnnDataオブジェクトを作成しています...")
    
    # ファイルパスの設定
    current_dir = os.getcwd()
    barcodes_file = os.path.join(current_dir, 'barcodes.tsv')
    genes_file = os.path.join(current_dir, 'genes.tsv')
    matrix_file = os.path.join(current_dir, 'matrix.mtx')
    metadata_file = os.path.join(current_dir, '..', '68k_pbmc_barcodes_annotation.tsv')
    
    # ファイルの存在確認
    for file_path in [barcodes_file, genes_file, matrix_file, metadata_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    
    print("ファイルを読み込んでいます...")
    
    # 1. バーコード（細胞ID）の読み込み
    barcodes = pd.read_csv(barcodes_file, header=None, names=['barcode'])
    print(f"バーコード数: {len(barcodes)}")
    
    # 2. 遺伝子情報の読み込み
    genes = pd.read_csv(genes_file, header=None, names=['gene_id', 'gene_name'], sep='\t')
    print(f"遺伝子数: {len(genes)}")
    
    # 3. 発現行列（Matrix Market形式）の読み込み
    print("発現行列を読み込んでいます（時間がかかる場合があります）...")
    matrix = mmread(matrix_file)
    
    # 行列の転置（scRNA-seqデータでは通常、行が遺伝子、列が細胞）
    # Matrix Market形式では遺伝子 x 細胞の形式で保存されているため、
    # AnnDataの標準形式（細胞 x 遺伝子）に転置する
    matrix_csc = matrix.tocsc().T
    print(f"発現行列の形状: {matrix_csc.shape} (細胞 x 遺伝子)")
    
    # 4. メタデータの読み込み
    metadata = pd.read_csv(metadata_file, sep='\t')
    print(f"メタデータの行数: {len(metadata)}")
    print(f"メタデータの列: {metadata.columns.tolist()}")
    
    # 5. バーコードをインデックスに設定してメタデータをマージ
    barcodes_df = barcodes.set_index('barcode')
    metadata_df = metadata.set_index('barcodes')
    
    # バーコードの一致を確認
    common_barcodes = barcodes_df.index.intersection(metadata_df.index)
    print(f"共通のバーコード数: {len(common_barcodes)}")
    
    # 共通のバーコードのデータのみを使用
    barcodes_filtered = barcodes_df.loc[common_barcodes]
    metadata_filtered = metadata_df.loc[common_barcodes]
    
    # 行列も対応する細胞のみに絞り込み
    barcode_indices = [i for i, barcode in enumerate(barcodes['barcode']) if barcode in common_barcodes]
    matrix_filtered = matrix_csc[barcode_indices, :]
    
    print(f"フィルタリング後の行列の形状: {matrix_filtered.shape}")
    
    # 6. AnnDataオブジェクトの作成
    print("AnnDataオブジェクトを作成しています...")
    
    # 遺伝子名に重複がある場合は、gene_idを使用
    gene_names = genes['gene_name'].values
    gene_ids = genes['gene_id'].values
    
    # 遺伝子名の重複をチェック
    if len(set(gene_names)) != len(gene_names):
        print("遺伝子名に重複があります。gene_idを使用します。")
        var_names = gene_ids
        var_df = pd.DataFrame(index=gene_ids)
        var_df['gene_id'] = gene_ids
        var_df['gene_name'] = gene_names
    else:
        var_names = gene_names
        var_df = pd.DataFrame(index=gene_names)
        var_df['gene_id'] = gene_ids
        var_df['gene_name'] = gene_names
    
    # AnnDataオブジェクトの作成
    adata = ad.AnnData(
        X=matrix_filtered,
        obs=metadata_filtered,
        var=var_df
    )
    
    # 基本的な情報を追加
    adata.obs_names = common_barcodes
    adata.var_names = var_names
    
    print(f"AnnDataオブジェクトが作成されました:")
    print(f"  - 細胞数: {adata.n_obs}")
    print(f"  - 遺伝子数: {adata.n_vars}")
    print(f"  - 観測データ（obs）の列: {adata.obs.columns.tolist()}")
    print(f"  - 変数データ（var）の列: {adata.var.columns.tolist()}")
    
    # 7. h5adファイルとして保存
    output_path = os.path.join(current_dir, '..', '..', '..', '..', 'pbmc68k.h5ad')
    print(f"h5adファイルとして保存しています: {output_path}")
    
    adata.write_h5ad(output_path)
    print("保存が完了しました!")
    
    return adata

if __name__ == "__main__":
    try:
        adata = create_anndata_from_pbmc68k()
        print("\nデータの概要:")
        print(adata)
        
        # 細胞タイプの分布を表示
        if 'celltype' in adata.obs.columns:
            print("\n細胞タイプの分布:")
            print(adata.obs['celltype'].value_counts())
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
