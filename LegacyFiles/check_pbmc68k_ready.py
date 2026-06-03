#!/usr/bin/env python3
"""
テスト用スクリプト - PBMC68k classifier sampling準備
"""

import os
import sys

# まず必要なファイルとディレクトリの存在確認
def check_requirements():
    print("=== PBMC68k Classifier Sample 実行前チェック ===")
    
    # 必要なファイル
    required_files = {
        'Model (backbone)': 'output/checkpoint/backbone/pbmc68k/model010000.pt',
        'Classifier': 'output/checkpoint/classifier/pbmc68k_classifier/model009999.pt', 
        'VAE': 'output/checkpoint/AE/pbmc68k/model_seed=0_step=9999.pt',
        'Data': 'data_preparation/pbmc68k.h5ad',
        'Main script': 'classifier_sample.py'
    }
    
    # 必要なディレクトリ
    required_dirs = {
        'Output base': 'output/',
        'Sample output': 'output/simulated_samples/'
    }
    
    print("\n1. ファイル存在確認:")
    all_files_exist = True
    for name, path in required_files.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"   {status} {name}: {path}")
        if not exists:
            all_files_exist = False
    
    print("\n2. ディレクトリ存在確認:")
    all_dirs_exist = True
    for name, path in required_dirs.items():
        exists = os.path.isdir(path)
        status = "✓" if exists else "✗"
        print(f"   {status} {name}: {path}")
        if not exists:
            all_dirs_exist = False
    
    print("\n3. Python環境確認:")
    try:
        import torch
        print(f"   ✓ PyTorch: {torch.__version__}")
        print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("   ✗ PyTorch not available")
        return False
    
    try:
        import scanpy as sc
        print(f"   ✓ Scanpy available")
    except ImportError:
        print("   ✗ Scanpy not available")
        return False
    
    print("\n4. データセット詳細:")
    try:
        import scanpy as sc
        adata = sc.read_h5ad('data_preparation/pbmc68k.h5ad')
        print(f"   ✓ データ形状: {adata.shape}")
        print(f"   ✓ 細胞タイプ数: {len(adata.obs.celltype.unique())}")
        print(f"   ✓ 遺伝子数: {adata.n_vars}")
        
        cell_types = list(adata.obs.celltype.unique())
        print(f"   ✓ 細胞タイプ: {cell_types}")
        
    except Exception as e:
        print(f"   ✗ データ読み込みエラー: {e}")
        return False
    
    print(f"\n5. 実行準備状況:")
    if all_files_exist and all_dirs_exist:
        print("   ✓ 全ての必要ファイルが存在します")
        print("   ✓ Classifier sampling実行準備完了")
        return True
    else:
        print("   ✗ 一部のファイルまたはディレクトリが不足しています")
        return False

if __name__ == "__main__":
    ready = check_requirements()
    
    if ready:
        print("\n=== 実行推奨パラメータ ===")
        print("細胞タイプ数: 11")
        print("推奨サンプル数/タイプ: 500-1000")
        print("推奨バッチサイズ: 250-500")
        print("分類器スケール: 2.0")
        print("フィルタリング: 有効")
        
        print("\n実行準備が整いました！")
        print("次のコマンドでclassifier samplingを開始できます:")
        print("python pbmc68k_classifier_sample.py")
    else:
        print("\n実行前に不足しているファイルを確認してください。")
