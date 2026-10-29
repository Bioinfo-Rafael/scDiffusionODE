"""Exercise the CLI population boundary and reject old all-training artifacts."""
import json
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from common.base import data_identity
from scripts import run_divergence_analysis as runner


def test_both_scalar_fits_exclude_other_classes_and_reject_old_artifacts(tmp_path, monkeypatch):
    classes = runner.cell_classes()
    labels = [classes[0], 'Other', *classes[1:], 'Other']
    full = ad.AnnData(np.arange(len(labels)*3, dtype=float).reshape(-1, 3),
                     obs=pd.DataFrame({'celltype': labels}, index=[f'c{i}' for i in range(len(labels))]))
    expected = full[full.obs.celltype.isin(classes)].copy()
    fitted = {}

    def projector(method):
        class Fake:
            def __init__(self, velocity, population, **kwargs):
                self.population = population

            def fit_scalar(self, **kwargs):
                fitted[method] = (self.population.copy(), kwargs)

            def save(self, path):
                path.mkdir(parents=True)
                (path/'metadata.json').write_text(json.dumps({'expression_and_ids_sha256': data_identity(self.population)}))

            @classmethod
            def load_saved(cls, *args):
                pytest.fail('All-training artifacts must not be reused')
        Fake.method = method
        return Fake

    old = tmp_path/'old'
    old.mkdir()
    (old/'metadata.json').write_text(json.dumps({
        'expression_and_ids_sha256': data_identity(full), 'terminal_obs_key': 'celltype',
        'terminal_cells': {'Erythroid3': expected.obs_names[expected.obs.celltype == 'Erythroid3'].tolist()}}))
    from analysis import smoke_data
    monkeypatch.setattr(smoke_data, 'make_smoke', lambda seed: (
        SimpleNamespace(provenance={'test': True}), full, SimpleNamespace(device='cpu', dtype='float64')))
    monkeypatch.setattr(runner, 'check_backend', lambda *args: {'test': True})
    monkeypatch.setattr(runner, 'HodgeProjector', projector('hodge'))
    monkeypatch.setattr(runner, 'CellRankProjector', projector('cellrank'))

    class FitsComplete(Exception):
        pass

    def stop_after_fits(*args):
        raise FitsComplete()

    monkeypatch.setattr(runner, 'scalar_selection', stop_after_fits)
    output = tmp_path/'result'
    monkeypatch.setattr(sys, 'argv', ['run', '--smoke', '--output', str(output),
                        '--hodge-dir', str(old), '--cellrank-dir', str(old), '--samples-per-class', '1'])
    with pytest.raises(FitsComplete):
        runner.main()
    assert set(fitted) == {'hodge', 'cellrank'}
    for population, settings in fitted.values():
        assert population.obs_names.tolist() == expected.obs_names.tolist()
        np.testing.assert_array_equal(population.X, expected.X)
    assert fitted['cellrank'][1]['terminal_values'] == ['Erythroid3']
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['training_population'] == 7
    assert manifest['scalar_population'] == 5
    assert manifest['scalar_cell_ids'] == expected.obs_names.tolist()
    assert manifest['identity']['input'] == data_identity(expected)
