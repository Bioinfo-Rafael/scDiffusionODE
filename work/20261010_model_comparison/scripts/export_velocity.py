"""Direct-script entry point for export_velocity."""
import importlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
if __name__ == "__main__":
    importlib.import_module("work.20261010_model_comparison.cli").main(["export_velocity",*sys.argv[1:]])
