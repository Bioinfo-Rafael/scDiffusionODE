"""No pickle or replacement checkpoint: checksummed NPZ/JSON/CSR + annotated h5ad."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from scipy import sparse
from .base import data_identity, sha256
from .scalar_extension import ScalarExtension


def save_projector(projector, directory):
    if projector.scalar is None:
        raise RuntimeError('No fitted scalar to save')
    dest = Path(directory).resolve()
    if dest.exists():
        raise FileExistsError(f'Refusing overwrite: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.'+dest.name+'-', dir=dest.parent))
    try:
        np.savez_compressed(temp/'scalar.npz', **projector.scalar.arrays())
        np.savez_compressed(temp/'official.npz', **projector.intermediates)
        projector.adata.write_h5ad(temp/'annotated.h5ad')
        if projector.transition_matrix is not None:
            sparse.save_npz(temp/'transition.npz', projector.transition_matrix)
        metadata = dict(projector.metadata)
        metadata['files_sha256'] = {p.name: sha256(p) for p in temp.iterdir()}
        (temp/'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)+'\n')
        # mkdir is the exclusive reservation; do not silently replace another run.
        dest.mkdir()
        for path in temp.iterdir():
            os.rename(path, dest/path.name)
        return dest
    finally:
        shutil.rmtree(temp)


def load_projector(cls, directory, velocity, adata):
    root = Path(directory)
    metadata = json.loads((root/'metadata.json').read_text())
    if metadata['schema'] != 'projection_v1' or metadata['method'] != cls.method:
        raise ValueError('Saved schema/method mismatch')
    for filename, digest in metadata['files_sha256'].items():
        if Path(filename).name != filename or sha256(root/filename) != digest:
            raise ValueError(f'Saved artifact integrity failure: {filename}')
    if metadata['expression_and_ids_sha256'] != data_identity(adata):
        raise ValueError('Saved AnnData expression/cell/feature order mismatch')
    if metadata['feature_ids'] != adata.var_names.tolist():
        raise ValueError('Saved feature order mismatch')
    identity = getattr(velocity, 'provenance', {})
    if metadata['velocity'].get('checkpoint_sha256'):
        for key in ('checkpoint_sha256', 'weights', 'diffusion_timestep'):
            if identity.get(key) != metadata['velocity'].get(key):
                raise ValueError(f'Velocity identity mismatch: {key}')
    # Generic callable identity cannot be proved by a name; verify its stored evaluations.
    with np.load(root/'official.npz', allow_pickle=False) as z:
        intermediates = {key: z[key] for key in z.files}
    from .velocity_adapter import evaluate_velocity
    if not np.allclose(evaluate_velocity(velocity, intermediates['X']), intermediates['V'], rtol=1e-6, atol=1e-8):
        raise ValueError('Supplied velocity differs from saved reference evaluations')
    obj = cls(velocity, adata, nonnegative=metadata['nonnegative'])
    with np.load(root/'scalar.npz', allow_pickle=False) as z:
        obj.scalar = ScalarExtension.restore(dict(z), metadata['interpolation'])
    obj.metadata, obj.intermediates = metadata, intermediates
    if (root/'transition.npz').exists():
        obj.transition_matrix = sparse.load_npz(root/'transition.npz')
    return obj
