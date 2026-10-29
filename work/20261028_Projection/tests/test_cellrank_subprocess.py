import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from CellRank.subprocess_bridge import run_cellrank, read_response, digest, check_backend

WORKER = os.environ.get('PROJECTION_TEST_CELLRANK_PYTHON', sys.executable)
SETTINGS = dict(n_neighbors=12, seed=0, similarity='cosine', softmax_scale=2., solver='gmres', tol=1e-8)


def test_subprocess_matches_official_competing_lineages_and_rejects_bad_ids(synthetic, constant_velocity, tmp_path, monkeypatch):
    monkeypatch.delenv('CELLRANK_PYTHON', raising=False)
    terminals = {'end': synthetic.obs_names[synthetic.X[:, 0] > 2.2].tolist(),
                 'other': synthetic.obs_names[synthetic.X[:, 0] < .7].tolist()}
    args = (synthetic.X, constant_velocity(synthetic.X), synthetic.obs_names.tolist(),
            synthetic.var_names.tolist(), terminals, SETTINGS)
    direct = run_cellrank(*args)
    child = run_cellrank(*args, python=WORKER, work_directory=tmp_path)
    assert child['runtime']['mode'] == 'subprocess'
    assert direct['time_columns'] == child['time_columns']
    np.testing.assert_allclose(child['transition'].toarray(), direct['transition'].toarray(), atol=1e-12)
    np.testing.assert_allclose(child['probabilities'], direct['probabilities'], atol=1e-10)
    np.testing.assert_allclose(child['times'], direct['times'], atol=1e-5, rtol=1e-6)
    assert np.isinf(child['times']).any()  # Competing-terminal conditional domain preserved.
    directory = Path(child['runtime']['work_directory'])
    request = json.loads((directory/'request.json').read_text())
    with np.load(directory/'result.npz', allow_pickle=False) as saved: arrays = dict(saved)
    arrays['gene_ids'] = arrays['gene_ids'][::-1]
    np.savez_compressed(directory/'result.npz', **arrays)
    meta = json.loads((directory/'response.json').read_text())
    meta['files_sha256']['result.npz'] = digest(directory/'result.npz')
    (directory/'response.json').write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match='gene_ids order mismatch'): read_response(directory, request)


def test_worker_failure_is_not_replaced_by_parent_solver(synthetic, constant_velocity, tmp_path):
    with pytest.raises(RuntimeError, match='worker exited'):
        run_cellrank(synthetic.X, constant_velocity(synthetic.X), synthetic.obs_names.tolist(),
                     synthetic.var_names.tolist(), {'end': ['cell0']}, dict(SETTINGS, n_neighbors=100),
                     python=WORKER, work_directory=tmp_path)
    logs = list(tmp_path.glob('cellrank-worker-*/worker.log'))
    assert len(logs) == 1 and 'n_neighbors' in logs[0].read_text()
    with pytest.raises(RuntimeError, match='existing Python executable'):
        check_backend(str(tmp_path/'missing-python'))


def test_parent_can_compute_and_save_without_cellrank_or_scanpy_imports(tmp_path):
    root = str(Path(__file__).resolve().parents[1])
    code = r'''
import sys, importlib.abc
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('cellrank', 'scanpy'):
            raise AssertionError('Parent tried to import '+fullname)
sys.meta_path.insert(0, Forbidden())
sys.path.insert(0, sys.argv[1])
import anndata as ad, numpy as np, pandas as pd
from CellRank.projector import CellRankProjector
x=np.random.default_rng(13).uniform(.5,2.5,(80,3))
a=ad.AnnData(x, obs=pd.DataFrame({'type':np.where(x[:,0]>2.2,'end','start')},index=['cell'+str(i) for i in range(80)]),var=pd.DataFrame(index=['g1','g2','g3']))
velocity=lambda x: np.broadcast_to([1.,.2,.1],np.asarray(x).shape).copy()
p=CellRankProjector(velocity,a,interpolation={'n_components':3,'n_centers':40})
p.fit_scalar(terminal_obs_key='type',terminal_values=['end'],n_neighbors=12,cellrank_python=sys.argv[2])
p.save(sys.argv[3])
restored=CellRankProjector.load_saved(sys.argv[3],velocity,a)
np.testing.assert_allclose(restored.scalar.values(x),p.scalar.values(x))
assert 'cellrank' not in sys.modules and 'scanpy' not in sys.modules
print('parent import isolation and persistence verified')
'''
    result = subprocess.run([sys.executable, '-c', code, root, WORKER, str(tmp_path/'saved')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
