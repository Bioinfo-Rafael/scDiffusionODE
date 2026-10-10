"""Direct-script entry point for prepare."""
import importlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
if __name__ == "__main__":
    importlib.import_module("work.20261010_model_comparison.cli").main(["prepare",*sys.argv[1:]])
