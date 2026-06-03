#!/usr/bin/env python3
"""
PBMC68k用 Classifier Sample実行スクリプト
11種類の細胞タイプ全てに対してconditional samplingを実行

【使用履歴】
- 2025-07-28 06:13:08: Type 0, 1, 2のサンプリングを実行 (フィルタリング有効)
- ログ: output/pbmc68k_classifier_sampling_20250728_061308.log
"""

import sys
import os
import datetime
sys.path.append(os.getcwd())

from classifier_sample import main
import argparse

def pbmc68k_conditional_sampling():
    """PBMC68k用 conditional sampling実行"""
    
    # Parameters
    num_class = 11  # Number of cell types
    num_samples_per_type = 500  # 各細胞タイプ500サンプル（テストで時間効率を確認したため）
    batch_size = 250
    
    log_file = f"output/pbmc68k_classifier_sampling_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    print(f"=== PBMC68k Classifier Sampling ===")
    print(f"Target cell types: {num_class}")
    print(f"Samples per type: {num_samples_per_type}")
    print(f"Total samples: {num_class * num_samples_per_type}")
    print(f"Log file: {log_file}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs('output/simulated_samples/pbmc68k/conditional', exist_ok=True)
    
    # Log initial setup
    with open(log_file, 'w') as f:
        f.write(f"PBMC68k Classifier Sampling Log\n")
        f.write(f"Start time: {datetime.datetime.now()}\n")
        f.write(f"Number of cell types: {num_class}\n")
        f.write(f"Samples per type: {num_samples_per_type}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write("=" * 80 + "\n\n")
    
    successful_types = []
    failed_types = []
    
    for cell_type_idx in range(num_class):
        print(f"\n--- Sampling cell type {cell_type_idx} ---")
        print(f"Target samples: {num_samples_per_type}")
        
        # Override sys.argv to simulate command line arguments
        sys.argv = [
            'pbmc68k_classifier_sample.py',
            '--num_samples', str(num_samples_per_type),
            '--batch_size', str(batch_size),
            '--model_path', 'output/checkpoint/backbone/pbmc68k/model010000.pt',
            '--classifier_path', 'output/checkpoint/classifier/pbmc68k_classifier/model009999.pt',
            '--sample_dir', f'output/simulated_samples/pbmc68k/conditional/type_{cell_type_idx}_',
            '--num_class', '11',
            '--classifier_scale', '2',
            '--ae_dir', 'output/checkpoint/AE/pbmc68k/model_seed=0_step=9999.pt',
            '--num_gene', '32738',
            '--filter', 'True'
        ]
        
        start_time = datetime.datetime.now()
        
        try:
            main(cell_type=[cell_type_idx])
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            
            print(f"✓ Cell type {cell_type_idx} sampling completed")
            print(f"  Duration: {duration}")
            
            successful_types.append(cell_type_idx)
            
            # Log success
            with open(log_file, 'a') as f:
                f.write(f"SUCCESS: Type {cell_type_idx}\n")
                f.write(f"  Start: {start_time}\n")
                f.write(f"  End: {end_time}\n")
                f.write(f"  Duration: {duration}\n\n")
                
        except Exception as e:
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            
            print(f"✗ Error in cell type {cell_type_idx}: {e}")
            print(f"  Duration: {duration}")
            
            failed_types.append((cell_type_idx, str(e)))
            
            # Log failure
            with open(log_file, 'a') as f:
                f.write(f"FAILED: Type {cell_type_idx}\n")
                f.write(f"  Start: {start_time}\n")
                f.write(f"  End: {end_time}\n")
                f.write(f"  Duration: {duration}\n")
                f.write(f"  Error: {e}\n\n")
            
            continue
    
    # Final summary
    print("\n" + "=" * 50)
    print("SAMPLING SUMMARY")
    print("=" * 50)
    print(f"Total cell types: {num_class}")
    print(f"Successful: {len(successful_types)}")
    print(f"Failed: {len(failed_types)}")
    
    if successful_types:
        print(f"\nSuccessful types:")
        for idx in successful_types:
            print(f"  {idx}")
    
    if failed_types:
        print(f"\nFailed types:")
        for idx, error in failed_types:
            print(f"  {idx}: {error}")
    
    # Final log entry
    with open(log_file, 'a') as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"End time: {datetime.datetime.now()}\n")
        f.write(f"Total cell types: {num_class}\n")
        f.write(f"Successful: {len(successful_types)}\n")
        f.write(f"Failed: {len(failed_types)}\n")
        if successful_types:
            f.write("Successful types:\n")
            for idx in successful_types:
                f.write(f"  {idx}\n")
        if failed_types:
            f.write("Failed types:\n")
            for idx, error in failed_types:
                f.write(f"  {idx}: {error}\n")
    
    print(f"\nLog saved to: {log_file}")

if __name__ == "__main__":
    pbmc68k_conditional_sampling()
