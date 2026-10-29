"""NPZ/JSON/CSR boundary: the parent never imports CellRank in subprocess mode."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
VERSION = '2.0.7'
SCHEMA = 'cellrank_worker_v1'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def worker_environment():
    env = os.environ.copy()
    # A child interpreter must not inherit the parent's import overrides/user site.
    for key in ('PYTHONPATH', 'PYTHONHOME'): env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', MPLBACKEND='Agg')
    return env


def python_executable(value=None):
    value = value or os.environ.get('CELLRANK_PYTHON')
    if not value: return None  # Backward-compatible, explicitly recorded in-process path.
    executable = shutil.which(os.path.expanduser(str(value)))
    if not executable:
        raise RuntimeError('CELLRANK_PYTHON must name an existing Python executable (not a shell command)')
    # Do not resolve symlinks: a venv Python's location determines its environment.
    return os.path.abspath(executable)


def runtime_info():
    packages = {p: importlib.metadata.version(p) for p in
                ('cellrank', 'scanpy', 'anndata', 'numpy', 'scipy', 'pandas', 'scikit-learn')}
    if packages['cellrank'] != VERSION: raise RuntimeError('CellRank 2.0.7 required')
    return dict(python=sys.executable, python_version=sys.version, packages=packages,
                torch_imported='torch' in sys.modules,
                source_sha256={p.name: digest(p) for p in
                               (HERE/'cellrank_core.py', HERE/'subprocess_bridge.py', HERE/'worker.py')})


def check_backend(python=None):
    executable = python_executable(python)
    if executable is None:
        from . import cellrank_core  # Only the legacy in-process branch imports this.
        return dict(mode='in_process', **runtime_info())
    command = [executable, str(HERE/'worker.py'), '--check']
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=worker_environment(), timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('CellRank worker preflight failed: '+str(exc)) from exc
    if result.returncode:
        raise RuntimeError('CellRank worker preflight failed; parent environment was not changed:\n'+result.stderr[-12000:])
    return dict(mode='subprocess', **json.loads(result.stdout))


def run_in_process(x, v, cell_ids, gene_ids, terminal_cells, settings):
    from . import cellrank_core as official
    kernel = official.velocity_kernel(x, v, pd.DataFrame(index=cell_ids), pd.DataFrame(index=gene_ids),
                                     n_neighbors=settings['n_neighbors'], seed=settings['seed'])
    p, p_checks = official.transition_matrix(kernel, similarity=settings['similarity'],
                                            softmax_scale=settings['softmax_scale'], seed=settings['seed'])
    estimator = official.terminal_estimator(kernel, terminal_cells)
    times, probabilities = official.absorption_times(estimator, solver=settings['solver'], tol=settings['tol'])
    checks = official.validate_absorption(estimator, times, probabilities, tol=settings['tol'])
    if times.index.tolist() != list(cell_ids): raise RuntimeError('Official cell order changed')
    return dict(transition=p, times=times.to_numpy(), time_columns=times.columns.tolist(),
                probabilities=np.asarray(probabilities), probability_columns=list(probabilities.names),
                cell_ids=list(cell_ids), gene_ids=list(gene_ids), transition_checks=p_checks,
                absorption_checks=checks, runtime=runtime_info())


def read_response(directory, request):
    directory = Path(directory)
    meta = json.loads((directory/'response.json').read_text())
    if meta['schema'] != SCHEMA or meta['request_sha256'] != digest(directory/'request.json'):
        raise RuntimeError('CellRank worker response/request mismatch')
    for name in ('result.npz', 'transition.npz'):
        if digest(directory/name) != meta['files_sha256'][name]:
            raise RuntimeError('CellRank worker result integrity failure: '+name)
    with np.load(directory/'result.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    for key in ('cell_ids', 'gene_ids'):
        if arrays[key].tolist() != request[key]: raise RuntimeError('CellRank worker '+key+' order mismatch')
    n, k = len(request['cell_ids']), len(request['terminal_cells'])
    names = list(request['terminal_cells'])
    if set(meta['probability_columns']) != set(names) or set(meta['time_columns']) != {name+'_mean' for name in names}:
        raise RuntimeError('CellRank worker lineage columns mismatch')
    if arrays['times'].shape != (n, k) or arrays['probabilities'].shape != (n, k):
        raise RuntimeError('CellRank worker result shape mismatch')
    p = sparse.load_npz(directory/'transition.npz').tocsr()
    if p.shape != (n, n) or not np.isfinite(p.data).all() or (p.data < 0).any() or not np.allclose(np.asarray(p.sum(1)).ravel(), 1, atol=1e-10):
        raise RuntimeError('Invalid worker transition matrix')
    if not np.isfinite(arrays['probabilities']).all(): raise RuntimeError('Invalid worker probabilities')
    return dict(transition=p, **arrays, **{key: meta[key] for key in
                ('time_columns', 'probability_columns', 'transition_checks', 'absorption_checks', 'runtime')})


def run_cellrank(x, v, cell_ids, gene_ids, terminal_cells, settings, *, python=None,
                 work_directory=None, timeout=None):
    executable = python_executable(python)
    if executable is None:
        result = run_in_process(x, v, cell_ids, gene_ids, terminal_cells, settings)
        result['runtime']['mode'] = 'in_process'
        return result
    if work_directory is not None:
        Path(work_directory).mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='cellrank-worker-', dir=work_directory))
    success = False
    try:
        np.savez(directory/'input.npz', X=x, V=v)
        request = dict(schema=SCHEMA, input_sha256=digest(directory/'input.npz'),
                       cell_ids=list(cell_ids), gene_ids=list(gene_ids),
                       terminal_cells=terminal_cells, settings=settings)
        (directory/'request.json').write_text(json.dumps(request, allow_nan=False))
        command = [executable, str(HERE/'worker.py'), str(directory)]
        print('CellRank subprocess: '+str(directory/'worker.log'), flush=True)
        try:
            with open(directory/'worker.log', 'w') as log:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=worker_environment(), timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError('CellRank worker failed; artifacts kept at '+str(directory)) from exc
        if run.returncode:
            tail = (directory/'worker.log').read_text(errors='replace')[-12000:]
            raise RuntimeError('CellRank worker exited '+str(run.returncode)+'; artifacts: '+str(directory)+'\n'+tail)
        result = read_response(directory, request)
        result['runtime'].update(mode='subprocess', command=command, request_sha256=digest(directory/'request.json'),
                                 input_sha256=request['input_sha256'], work_directory=str(directory),
                                 worker_log=(directory/'worker.log').read_text(errors='replace')[-12000:])
        success = True
        return result
    finally:
        # Failed requests are retained for diagnosis. Explicit output directories retain all exchanges.
        if success and work_directory is None: shutil.rmtree(directory)
