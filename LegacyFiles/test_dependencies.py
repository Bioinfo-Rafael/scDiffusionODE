#!/usr/bin/env python3
"""
Test script to check all dependencies for cell_train_copy.py
"""

import sys

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            exec(f"from {package_name} import {module_name}")
        else:
            exec(f"import {module_name}")
        print(f"✓ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {module_name} import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ {module_name} import error: {e}")
        return False

print("Testing dependencies for cell_train_copy.py...")
print("=" * 50)

# Basic imports
dependencies = [
    ("argparse", None),
    ("torch", None),
    ("numpy", None),
    ("random", None),
    ("scanpy", None),
]

# Guided diffusion imports
guided_diffusion_modules = [
    ("dist_util", "guided_diffusion"),
    ("logger", "guided_diffusion"),
    ("load_data", "guided_diffusion.cell_datasets_loader"),
    ("create_named_schedule_sampler", "guided_diffusion.resample"),
    ("create_gaussian_diffusion", "guided_diffusion.script_util"),
    ("TrainLoop", "guided_diffusion.train_util"),
    ("Cell_Unet", "guided_diffusion.cell_model"),
]

# ODE imports (already tested, but let's verify again)
ode_modules = [
    ("GeneODE", "ODE.ode_analysis"),
    ("ODE_ML_Hybrid", "ODE.ode_analysis"),
]

all_passed = True

print("\n1. Testing basic Python modules...")
for module, package in dependencies:
    if not test_import(module, package):
        all_passed = False

print("\n2. Testing guided_diffusion modules...")
for module, package in guided_diffusion_modules:
    if not test_import(module, package):
        all_passed = False

print("\n3. Testing ODE modules...")
for module, package in ode_modules:
    if not test_import(module, package):
        all_passed = False

print("\n" + "=" * 50)
if all_passed:
    print("🎉 All dependencies are available!")
    print("You can try running the full training script now.")
else:
    print("❌ Some dependencies are missing.")
    print("Please install the missing packages before running the training script.")

print(f"\nPython version: {sys.version}")
