"""Auditable HDF5/subprocess boundary, never a Python Hodge implementation."""
import os
from pathlib import Path
import subprocess
import tempfile
import h5py
import numpy as np
from common.base import sha256

HERE = Path(__file__).resolve().parent
COMMIT = 'd5d6b09431c0ccbb16e0b63414a5038511c19b85'


def run_ddhodge(x, v, *, k=12, rdim=4, seed=0, maxiter=5000, ssa=True, julia=None, timeout=3600):
    x, v = np.asarray(x, dtype=float), np.asarray(v, dtype=float)
    if x.ndim != 2 or x.shape != v.shape or not np.isfinite(x).all() or not np.isfinite(v).all():
        raise ValueError('Expected matching finite expression and velocity (N,D)')
    if not 2 <= rdim <= min(x.shape[1], k) or not k < len(x):
        raise ValueError('Official workflow requires 2 <= rdim <= min(D,k), k < N')
    command = [julia or os.environ.get('JULIA', 'julia'), '--startup-file=no',
               '--project='+str(HERE/'julia_env'), str(HERE/'run_ddhodge.jl')]
    with tempfile.TemporaryDirectory(prefix='ddhodge-') as tmp:
        source, dest = Path(tmp)/'input.h5', Path(tmp)/'output.h5'
        with h5py.File(source, 'x') as f:
            f['shape_ND'] = np.array(x.shape, dtype=np.int64)
            f['X_C_flat'], f['V_C_flat'] = x.ravel(order='C'), v.ravel(order='C')
            for key, value in dict(k=k, rdim=rdim, seed=seed, maxiter=maxiter, ssa=int(ssa)).items():
                f[key] = value
        try:
            run = subprocess.run(command+[str(source), str(dest)], capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as exc:
            raise RuntimeError('Julia unavailable. Initialize Hodge/julia_env; no fallback solver is used.') from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Official ddHodge subprocess timed out; no result accepted') from exc
        if run.returncode:
            raise RuntimeError('Official ddHodge failed (no fallback):\n'+run.stdout+'\n'+run.stderr)
        with h5py.File(dest, 'r') as f:
            arrays = {key: f[key][()] for key in f if f[key].dtype.kind not in 'OSU'}
            info = {key: f[key].asstr()[()] for key in ('julia_version', 'ddhodge_version', 'ddhodge_commit')}
        if info['ddhodge_commit'] != COMMIT or arrays['u'].shape != (len(x),):
            raise RuntimeError('ddHodge version or output order/shape contract failed')
        if not np.array_equal(arrays['shape_DN'], [x.shape[1], len(x)]):
            raise RuntimeError('Julia transpose contract failed')
        info.update(command=command, stdout=run.stdout, stderr=run.stderr,
                    manifest_sha256=sha256(HERE/'julia_env/Manifest.toml'),
                    settings=dict(k=k, rdim=rdim, seed=seed, maxiter=maxiter, ssa=ssa,
                                  lambda_hessian=0.1, epsilon_jacobian=0.1, krytol=1e-32),
                    sign='F_gradient=-grad(U); official d0*u=-w least squares')
        return arrays, info
