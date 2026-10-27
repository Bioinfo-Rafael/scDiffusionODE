from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(13)
    x = rng.uniform(.5, 2.5, (80, 3))
    a = ad.AnnData(x, obs=pd.DataFrame(index=[f'cell{i}' for i in range(len(x))]),
                   var=pd.DataFrame(index=['g1', 'g2', 'g3']))
    a.obs['type'] = np.where(x[:, 0] > 2.2, 'end', 'start')
    return a


@pytest.fixture
def constant_velocity():
    return lambda x: np.broadcast_to([1., .2, .1], np.asarray(x).shape).copy()
