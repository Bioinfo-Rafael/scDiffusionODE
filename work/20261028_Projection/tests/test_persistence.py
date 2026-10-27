import json
import numpy as np
import anndata as ad
import pytest
from CellRank.projector import CellRankProjector


def test_roundtrip_h5ad_projection_and_integrity(synthetic, constant_velocity, tmp_path):
    p = CellRankProjector(constant_velocity, synthetic,
                          interpolation={'n_components': 3, 'n_centers': 40})
    p.fit_scalar(terminal_obs_key='type', terminal_values=['end'], n_neighbors=12)
    path = p.save(tmp_path/'result')
    a = ad.read_h5ad(path/'annotated.h5ad')
    assert json.loads(a.uns['projection_analysis']['cellrank']['metadata_json'])['method'] == 'cellrank'
    q = CellRankProjector.load_saved(path, constant_velocity, a)
    x = synthetic.X[0]
    c = p.scalar.values(x)[0]
    np.testing.assert_array_equal(q.get_projected_field(c)(x), p.get_projected_field(c)(x))
    with pytest.raises(FileExistsError): p.save(path)
    with pytest.raises(ValueError, match='differs'):
        CellRankProjector.load_saved(path, lambda x: -constant_velocity(x), a)
    bad = a[:, ::-1].copy()
    with pytest.raises(ValueError, match='mismatch'):
        CellRankProjector.load_saved(path, constant_velocity, bad)
    with (path/'scalar.npz').open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        CellRankProjector.load_saved(path, constant_velocity, a)


def test_hodge_saved_roundtrip(synthetic, tmp_path):
    import os
    import shutil
    from Hodge.projector import HodgeProjector
    if not shutil.which(os.environ.get('JULIA', 'julia')):
        pytest.skip('Julia unavailable: Hodge persistence not executed')
    velocity = lambda x: -np.asarray(x)
    p = HodgeProjector(velocity, synthetic, interpolation={'n_components': 3, 'n_centers': 64})
    p.fit_scalar(k=12, rdim=3, ssa=False)
    c = float(p.scalar.values(synthetic.X[0])[0])
    original = p.get_projected_field(c)
    path = p.save(tmp_path/'hodge')
    q = HodgeProjector.load_saved(path, velocity, ad.read_h5ad(path/'annotated.h5ad'))
    np.testing.assert_array_equal(q.get_projected_field(c)(synthetic.X[0]), original(synthetic.X[0]))
    assert q.metadata['selected_projection']['c'] == c
