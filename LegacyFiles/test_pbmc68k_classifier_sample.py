#!/usr/bin/env python3
"""
PBMC68k用 Classifier Sample テスト実行スクリプト
1つの細胞タイプのみで動作確認
"""

import sys
import os
import datetime
sys.path.append(os.getcwd())

from classifier_sample import main
import argparse

def test_pbmc68k_sampling():
    """PBMC68k用 テスト実行（細胞タイプ0のみ）"""
    
    # Test parameters
    test_cell_type = 0  # CD8+ Cytotoxic T
    num_samples = 100   # テスト用に少数
    batch_size = 50
    
    print(f"=== PBMC68k Classifier Sampling TEST ===")
    print(f"Test cell type: {test_cell_type}")
    print(f"Test samples: {num_samples}")
    print(f"Batch size: {batch_size}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs('output/simulated_samples/pbmc68k/conditional', exist_ok=True)
    
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time}")
    
    # Override sys.argv to simulate command line arguments
    sys.argv = [
        'test_pbmc68k_classifier_sample.py',
        '--num_samples', str(num_samples),
        '--batch_size', str(batch_size),
        '--model_path', 'output/checkpoint/backbone/pbmc68k/model010000.pt',
        '--classifier_path', 'output/checkpoint/classifier/pbmc68k_classifier/model009999.pt',
        '--sample_dir', f'output/simulated_samples/pbmc68k/conditional/test_type_{test_cell_type}_',
        '--num_class', '11',
        '--classifier_scale', '2',
        '--ae_dir', 'output/checkpoint/AE/pbmc68k/model_seed=0_step=9999.pt',
        '--num_gene', '32738',
        '--filter', 'True'
    ]
    
    try:
        print("Starting classifier sampling...")
        main(cell_type=[test_cell_type])
        
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        
        print(f"\n✓ Test sampling completed successfully!")
        print(f"Duration: {duration}")
        print(f"Output should be saved to: output/simulated_samples/pbmc68k/conditional/test_type_{test_cell_type}_0")
        
        # Check if output file exists
        output_file = f"output/simulated_samples/pbmc68k/conditional/test_type_{test_cell_type}_0.npz"
        if os.path.exists(output_file):
            print(f"✓ Output file exists: {output_file}")
            
            # Check file size
            file_size = os.path.getsize(output_file) / (1024*1024)  # MB
            print(f"✓ Output file size: {file_size:.2f} MB")
        else:
            print(f"✗ Output file not found: {output_file}")
            
        return True
        
    except Exception as e:
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        
        print(f"\n✗ Test sampling failed!")
        print(f"Duration: {duration}")
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        
        return False

if __name__ == "__main__":
    success = test_pbmc68k_sampling()
    
    if success:
        print("\n=== Test completed successfully! ===")
        print("Ready to run full classifier sampling with all 11 cell types.")
        print("Command: python pbmc68k_classifier_sample.py")
    else:
        print("\n=== Test failed ===")
        print("Please check the error messages above and fix issues before running full sampling.")
