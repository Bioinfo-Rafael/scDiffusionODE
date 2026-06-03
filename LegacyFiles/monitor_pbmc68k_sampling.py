#!/usr/bin/env python3
"""
PBMC68k Classifier Sampling 進行状況監視スクリプト
"""

import time
import os
import re
import datetime

def monitor_sampling_progress():
    """Sampling進行状況を監視"""
    
    cell_types = [
        'CD8+ Cytotoxic T',
        'CD8+/CD45RA+ Naive Cytotoxic', 
        'CD4+/CD45RO+ Memory',
        'CD19+ B',
        'CD4+/CD25 T Reg',
        'CD56+ NK',
        'CD4+ T Helper2',
        'CD4+/CD45RA+/CD25- Naive T',
        'CD34+',
        'Dendritic',
        'CD14+ Monocyte'
    ]
    
    log_file = "pbmc68k_sampling_output.log"
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    print("=== PBMC68k Classifier Sampling Progress Monitor ===")
    print(f"Monitoring log file: {log_file}")
    print(f"Total cell types: {len(cell_types)}")
    print("=" * 60)
    
    completed_types = []
    current_type = None
    last_size = 0
    
    while True:
        try:
            # Check if process is still running
            import subprocess
            result = subprocess.run(['pgrep', '-f', 'pbmc68k_classifier_sample.py'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                print("\nProcess completed or not running.")
                break
            
            # Read log file
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Count completed types
            completed_pattern = r'✓ Cell type (\d+) \(([^)]+)\) sampling completed'
            completed_matches = re.findall(completed_pattern, content)
            
            # Find current type
            current_pattern = r'--- Sampling cell type (\d+): ([^-]+) ---'
            current_matches = re.findall(current_pattern, content)
            
            # Get latest progress
            step_pattern = r'step\s+(\d+)'
            step_matches = re.findall(step_pattern, content)
            
            created_pattern = r'created (\d+) samples'
            created_matches = re.findall(created_pattern, content)
            
            # Display progress
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n[{current_time}] Progress Update:")
            print(f"Completed types: {len(completed_matches)}/11")
            
            if completed_matches:
                print("Completed:")
                for type_idx, type_name in completed_matches:
                    print(f"  ✓ Type {type_idx}: {type_name}")
            
            if current_matches:
                current_type_idx, current_type_name = current_matches[-1]
                print(f"\nCurrently processing:")
                print(f"  → Type {current_type_idx}: {current_type_name}")
                
                if step_matches:
                    latest_step = step_matches[-1]
                    print(f"  → Latest step: {latest_step}")
                
                if created_matches:
                    latest_created = created_matches[-1]
                    print(f"  → Latest samples created: {latest_created}")
            
            # Check for completion
            if len(completed_matches) >= 11:
                print("\n🎉 All cell types completed!")
                break
            
            print("-" * 40)
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"Error during monitoring: {e}")
            time.sleep(10)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        completed_matches = re.findall(completed_pattern, content)
        print(f"Total completed types: {len(completed_matches)}/11")
        
        # Check output files
        print("\nGenerated files:")
        output_dir = "output/simulated_samples/pbmc68k/conditional/"
        if os.path.exists(output_dir):
            for i in range(11):
                output_file = f"{output_dir}type_{i}_0.npz"
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file) / (1024*1024)  # MB
                    print(f"  ✓ Type {i}: {output_file} ({file_size:.2f} MB)")
                else:
                    print(f"  ✗ Type {i}: {output_file} (not found)")
        
    except Exception as e:
        print(f"Error reading final summary: {e}")

if __name__ == "__main__":
    monitor_sampling_progress()
