"""Load hash-checked, unchanged upstream computation; adapt only execution context."""
import ast
import gc
import importlib.util
import json
from pathlib import Path

import config as C
from common import require, sha256

SOURCE = C.HERE / 'vendor' / 'benchmark_source'


def verify_sources():
    manifest = json.loads((SOURCE / 'sources.json').read_text())
    require(manifest['commit'] == C.BENCHMARK_COMMIT, 'upstream commit mismatch')
    for name, item in manifest['files'].items():
        require(sha256(SOURCE / name) == item['sha256'], f'upstream source changed: {name}')
    return manifest


def utils():
    verify_sources()
    spec = importlib.util.spec_from_file_location('benchmark_official_utils', SOURCE / 'utils.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def namespace(directory):
    import numpy as np
    import scanpy as sc
    import scvelo as scv
    directory = Path(directory).resolve()
    return dict(np=np, sc=sc, scv=scv, gc=gc, utils=utils(),
                DATA_DIR=directory.parent, DATASET=directory.name,
                SAVE_DATA=True, SEED=C.SEED)


def preprocess(folds, directory):
    """Execute the entire original computation cell, without rewriting its statements."""
    verify_sources()
    notebook = json.loads((SOURCE / 'preprocessing_general.ipynb').read_text())
    cell = notebook['cells'][3]
    require(cell['cell_type'] == 'code', 'unexpected upstream notebook layout')
    code = ''.join(cell['source'])
    require(code.startswith('for i in range(len(sub_adata_lst)):'), 'unexpected computation cell')
    (Path(directory) / 'processed').mkdir(parents=True, exist_ok=False)
    env = namespace(directory)
    env['sub_adata_lst'] = folds
    exec(compile(code, str(SOURCE / 'preprocessing_general.ipynb') + ':cell3', 'exec'), env)


def stochastic_function(directory):
    """Extract the exact function AST, avoiding upstream argparse and its call-name typo."""
    verify_sources()
    path = SOURCE / '03_run_scvelo_stc.py'
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_scvelo_st']
    require(len(functions) == 1, 'upstream stochastic function missing')
    env = namespace(directory)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), env)
    return env['run_scvelo_st']
