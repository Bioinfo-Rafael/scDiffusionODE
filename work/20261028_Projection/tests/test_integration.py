import json
import os
from pathlib import Path
import numpy as np
import pytest
from common.velocity_adapter import ODEVelocity, load_d20
from Hodge.projector import HodgeProjector


def test_constructor_lazy_and_feature_order(synthetic):
    def fail(_): raise AssertionError('Constructor must not evaluate velocity')
    p = HodgeProjector(fail, synthetic)
    assert p.scalar is None
    fail.feature_ids = ['g3', 'g2', 'g1']
    with pytest.raises(ValueError, match='order mismatch'): HodgeProjector(fail, synthetic)


def test_torch_adapter_single_batch_fixed_timestep():
    import torch
    class DummyField(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)
        def forward(self, x, t):
            assert torch.all(t == 49)
            return x*self.weight
    field = DummyField()
    before = field.weight.detach().clone()
    velocity = ODEVelocity(field, ['a', 'b'], {})
    np.testing.assert_array_equal(velocity([1., 2.]), [2., 4.])
    np.testing.assert_array_equal(velocity([[1., 2.], [2., 3.]]), [[2., 4.], [4., 6.]])
    assert torch.equal(before, field.weight)


def test_real_checkpoint_end_to_end():
    path = os.environ.get('PROJECTION_REAL_CONFIG')
    if not path:
        pytest.skip('Real D20 test requires PROJECTION_REAL_CONFIG; synthetic checkpoints do not count')
    cfg = json.loads(Path(path).read_text())
    velocity, adata = load_d20(cfg)
    # Explicit small real-cell smoke subset, not evidence for full-data behavior.
    a = adata[:128].copy()
    p = HodgeProjector(velocity, a, interpolation=cfg['interpolation'])
    p.fit_scalar(k=12, rdim=4)
    x = a.X[:1].toarray() if hasattr(a.X, 'toarray') else a.X[:1]
    assert np.isfinite(p.get_projected_field(p.scalar.values(x)[0])(x)).all()
    from CellRank.projector import CellRankProjector
    if 'real_test_terminal_obs_key' not in cfg or 'real_test_terminal_values' not in cfg:
        pytest.fail('Real integration needs explicit real_test_terminal_obs_key/values in config')
    b = adata[:128].copy()
    q = CellRankProjector(velocity, b, interpolation=cfg['interpolation'])
    q.fit_scalar(terminal_obs_key=cfg['real_test_terminal_obs_key'],
                 terminal_values=cfg['real_test_terminal_values'], lineage=cfg.get('real_test_lineage'))
    eligible = np.flatnonzero(b.obs['projection_cellrank_eligible'])[0]
    x = b.X[eligible:eligible+1]
    x = x.toarray() if hasattr(x, 'toarray') else x
    assert np.isfinite(q.get_projected_field(q.scalar.values(x)[0])(x)).all()
