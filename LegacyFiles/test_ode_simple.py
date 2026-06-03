#!/usr/bin/env python3
"""
Simple test script to verify ODE module imports and basic functionality
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

print("Testing ODE module import...")

try:
    from ODE.ode_analysis import GeneODE, ODE_ML_Hybrid
    print("✓ Successfully imported GeneODE and ODE_ML_Hybrid")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Create a small test dataset
print("\nCreating test data...")
gene_list = ['GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5']
n_genes = len(gene_list)
n_cells = 10

# Create test expression data
test_data = torch.randn(n_cells, n_genes)
print(f"✓ Created test data: {test_data.shape}")

# Create a simple TSV file for gene regulatory network
print("\nCreating test TSV file...")
test_edges = pd.DataFrame({
    'from': ['GENE1', 'GENE2', 'GENE3'],
    'to': ['GENE2', 'GENE3', 'GENE4']
})
tsv_path = "test_edges.tsv"
test_edges.to_csv(tsv_path, sep='\t', index=False)
print(f"✓ Created test TSV file: {tsv_path}")

# Test GeneODE initialization
print("\nTesting GeneODE initialization...")
try:
    ode_model = GeneODE(
        gene_list=gene_list,
        edge_tsv_path=tsv_path,
        device="cpu"
    )
    print(f"✓ GeneODE initialized successfully")
    print(f"  - Full genes: {ode_model.n}")
    print(f"  - Sub genes: {ode_model.m}")
    print(f"  - Sub gene list: {ode_model.sub_genes}")
except Exception as e:
    print(f"✗ GeneODE initialization failed: {e}")
    exit(1)

# Test forward pass
print("\nTesting GeneODE forward pass...")
try:
    output = ode_model(test_data)
    print(f"✓ Forward pass successful: {output.shape}")
    print(f"  - Input shape: {test_data.shape}")
    print(f"  - Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    exit(1)

# Test ODE_ML_Hybrid
print("\nTesting ODE_ML_Hybrid...")
try:
    # Create a simple ML model
    ml_model = torch.nn.Linear(n_genes, n_genes)
    
    # Create hybrid model
    hybrid_model = ODE_ML_Hybrid(
        ode_model=ode_model,
        ml_model=ml_model,
        timesteps=10
    )
    print("✓ ODE_ML_Hybrid initialized successfully")
    
    # Test forward pass with different time steps
    for t in [0, 5, 9]:
        output = hybrid_model(test_data, t)
        print(f"✓ Hybrid forward pass at t={t}: {output.shape}")
        
except Exception as e:
    print(f"✗ ODE_ML_Hybrid test failed: {e}")
    exit(1)

print("\n🎉 All tests passed successfully!")
print("The union type syntax fixes are working correctly.")

# Clean up
import os
if os.path.exists(tsv_path):
    os.remove(tsv_path)
    print(f"✓ Cleaned up test file: {tsv_path}")
