"""Read-only parent/worker preflight. Never installs or updates any packages."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cellrank-python', default=os.environ.get('CELLRANK_PYTHON'))
    args, _ = parser.parse_known_args()
    if sys.version_info < (3, 9): raise RuntimeError('Parent requires Python >=3.9')
    for name in ('torch', 'numpy', 'scipy', 'pandas', 'sklearn', 'matplotlib', 'anndata', 'h5py'):
        importlib.import_module(name)
    import torch
    point = torch.tensor([2.])
    _, derivative = torch.func.jvp(lambda x: x*x, (point,), (torch.ones_like(point),))
    if not torch.allclose(derivative, torch.tensor([4.])): raise RuntimeError('Parent JVP check failed')
    from CellRank.subprocess_bridge import check_backend
    try:
        backend = check_backend(args.cellrank_python)
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError('Set CELLRANK_PYTHON to the separate worker Python. Do not install CellRank into scdiffusion.') from exc
    print(json.dumps(dict(parent_python=sys.executable, parent_version=sys.version,
                         torch=torch.__version__, cuda=torch.cuda.is_available(),
                         cellrank_imported_in_parent='cellrank' in sys.modules, cellrank_backend=backend), indent=2))


if __name__ == '__main__': main()
