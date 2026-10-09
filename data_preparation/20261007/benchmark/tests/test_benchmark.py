"""Synthetic I/O, strict alignment and real (unmocked) upstream metric tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import config as C
import prepare
import run
from common import sha256, write_json

N_FULL, N_ERY, N_GENES = 110, 100, 48


def source_and_prediction():
    rng = np.random.default_rng(8)
    time = np.linspace(0, 1, N_FULL)
    slope = rng.uniform(-0.5, 2, N_GENES)
    offset = rng.uniform(1, 3, N_GENES)
    counts = offset + time[:, None] * slope + rng.uniform(0, 0.03, (N_FULL, N_GENES))
    sums = counts.sum(axis=1, keepdims=True)
    x = (1e4 * counts / sums).astype(np.float32)
    velocity = (1e4 * (slope * sums - counts * slope.sum()) / sums**2).astype(np.float32)
    labels = np.repeat(C.ERYTHROID_CELLTYPES, 20).tolist() + ['Other'] * 10
    stages = [list(C.STAGE_TO_DAY)[min(int(t * 9), 8)] for t in time]
    obs = pd.DataFrame({'celltype': labels, 'stage': stages}, index=[f'cell_{i}' for i in range(N_FULL)])
    var = pd.DataFrame({'gene_name': [f'g{i}' for i in range(N_GENES)]}, index=[f'g{i}' for i in range(N_GENES)])
    source = ad.AnnData(X=sparse.csr_matrix(x), obs=obs, var=var)
    source.layers['spliced'] = source.X.copy()
    source.layers['unspliced'] = sparse.csr_matrix(x[:, ::-1])
    source.uns['mouse_gastrulation_preprocessing'] = {
        'target_sum': 1e4, 'normalization': 'scanpy.pp.normalize_total', 'log1p': False}
    pred = ad.AnnData(X=None, obs=obs.copy(), var=var.copy())
    pred.layers['velocity'] = velocity
    pred.obs['latent_time'] = time
    pred.uns['benchmark_velocity'] = {
        'definition': 'ds_dt', 'expression_scale': C.EXPRESSION_SCALE,
        'time_direction': 'forward', 'training_n_cells': N_FULL,
        'time_unit': 'synthetic time', 'inference_description': 'analytic normalized derivative',
        'checkpoint': 'synthetic; no model training'}
    return source, pred


def fake_geometry(ref):
    """Only a shape-correct test fixture; never used by production preparation."""
    ref.obsm['X_pca'] = np.zeros((ref.n_obs, C.N_PCS))
    ref.obsm['X_umap'] = np.zeros((ref.n_obs, 2))
    ref.obsp['distances'] = sparse.eye(ref.n_obs, format='csr')
    ref.obsp['connectivities'] = sparse.eye(ref.n_obs, format='csr')
    ref.uns['neighbors'] = {'params': {'n_neighbors': 30, 'n_pcs': 30}}
    ref.var['benchmark_geometry_hvg'] = True
    return ref


@pytest.fixture(scope='module')
def reference(request):
    source, _ = source_and_prediction()
    ref = prepare.make_reference(source, expected_full=N_FULL, expected_ery=N_ERY)
    return prepare.build_geometry(ref) if request.config.getoption('--run-integration') else fake_geometry(ref)


def align(ref, pred, genes=None, time_key=None):
    return run.align_prediction(ref, pred, genes if genes is not None else ref.var_names,
                                'velocity', time_key, expected_full=N_FULL, expected_ery=N_ERY)


def test_reference_io_and_source_unchanged(tmp_path, reference):
    source, pred = source_and_prediction()
    path = tmp_path / 'source.h5ad'
    source.write_h5ad(path)
    before = sha256(path)
    output = prepare.make_reference(ad.read_h5ad(path), expected_full=N_FULL, expected_ery=N_ERY)
    output.write_h5ad(tmp_path / 'ery.h5ad')
    restored = ad.read_h5ad(tmp_path / 'ery.h5ad')
    assert restored.shape == (N_ERY, N_GENES)
    assert restored.obs_names.equals(source.obs_names[:N_ERY])
    assert restored.var_names.equals(source.var_names)
    np.testing.assert_array_equal(restored.X.toarray(), source.X[:N_ERY].toarray())
    assert sha256(path) == before
    reference.write_h5ad(tmp_path / 'reference.h5ad')
    pred.write_h5ad(tmp_path / 'prediction.h5ad')
    aligned, _ = align(ad.read_h5ad(tmp_path / 'reference.h5ad'), ad.read_h5ad(tmp_path / 'prediction.h5ad'))
    assert aligned.shape == reference.shape


@pytest.mark.parametrize('full', [True, False])
@pytest.mark.parametrize('sparse_velocity', [True, False])
def test_permutations(reference, full, sparse_velocity):
    _, pred = source_and_prediction()
    if not full:
        pred = pred[reference.obs_names].copy()
    baseline, _ = align(reference, pred)
    if sparse_velocity:
        pred.layers['velocity'] = sparse.csr_matrix(pred.layers['velocity'])
    shuffled = pred[np.random.default_rng(9).permutation(pred.n_obs), pred.var_names[::-1]].copy()
    actual, details = align(reference, shuffled)
    velocity = actual.layers['candidate_velocity']
    if sparse.issparse(velocity): velocity = velocity.toarray()
    np.testing.assert_array_equal(velocity, baseline.layers['candidate_velocity'])
    np.testing.assert_array_equal(actual.obs.candidate_time, baseline.obs.candidate_time)
    assert details['evaluation_n_cells'] == N_ERY


def test_explicit_gene_subset(reference):
    _, pred = source_and_prediction()
    subset = pred[:, :35].copy()
    with pytest.raises(ValueError, match='gene IDs differ'):
        align(reference, subset)
    aligned, _ = align(reference, subset[:, ::-1], subset.var_names)
    assert aligned.n_vars == 35
    np.testing.assert_array_equal(aligned.layers['spliced'].toarray(), reference.layers['spliced'][:, :35].toarray())


@pytest.mark.parametrize('case,match', [
    ('missing_cell', 'cell IDs'), ('missing_gene', 'gene IDs'),
    ('unknown_cell', 'cell IDs'), ('unknown_gene', 'gene IDs'),
    ('duplicate_cell', 'duplicate'), ('duplicate_gene', 'duplicate'),
    ('nan', 'NaN/Inf'), ('inf', 'NaN/Inf'), ('overflow', 'float32'),
    ('time_nan', 'NaN/Inf'), ('time_constant', 'constant'),
    ('wrong_scale', 'scale'), ('x0', 'ds_dt'), ('subset_training', 'full cells'),
    ('conflict', 'conflicts'), ('missing_velocity', 'missing velocity'),
])
def test_invalid_predictions(reference, case, match):
    _, pred = source_and_prediction()
    if case == 'missing_cell': pred = pred[1:].copy()
    if case == 'missing_gene': pred = pred[:, 1:].copy()
    if case == 'unknown_cell': pred.obs_names = ['unknown', *pred.obs_names[1:]]
    if case == 'unknown_gene': pred.var_names = ['unknown', *pred.var_names[1:]]
    if case == 'duplicate_cell': pred.obs_names = [pred.obs_names[1], *pred.obs_names[1:]]
    if case == 'duplicate_gene': pred.var_names = [pred.var_names[1], *pred.var_names[1:]]
    if case == 'nan': pred.layers['velocity'][-1, 0] = np.nan  # outside erythroid also rejected
    if case == 'inf': pred.layers['velocity'][0, 0] = np.inf
    if case == 'overflow':
        pred.layers['velocity'] = pred.layers['velocity'].astype(float)
        pred.layers['velocity'][0, 0] = 1e100
    if case == 'time_nan': pred.obs.loc[pred.obs_names[0], 'latent_time'] = np.nan
    if case == 'time_constant': pred.obs['latent_time'] = 1.
    if case == 'wrong_scale': pred.uns['benchmark_velocity']['expression_scale'] = 'log1p'
    if case == 'x0': pred.uns['benchmark_velocity']['definition'] = 'x0'
    if case == 'subset_training': pred.uns['benchmark_velocity']['training_n_cells'] = N_ERY
    if case == 'conflict': pred.obs.loc[pred.obs_names[0], 'stage'] = 'E8.5'
    if case == 'missing_velocity': del pred.layers['velocity']
    with pytest.raises(ValueError, match=match):
        align(reference, pred)


@pytest.mark.parametrize('case', ['unknown_stage', 'missing_stage', 'normalization', 'cell_count'])
def test_invalid_source(case):
    source, _ = source_and_prediction()
    if case == 'unknown_stage': source.obs.loc[source.obs_names[0], 'stage'] = 'E9'
    if case == 'missing_stage': source.obs.loc[source.obs_names[0], 'stage'] = None
    if case == 'normalization': source.layers['spliced'] *= 2
    if case == 'cell_count': source = source[1:].copy()
    with pytest.raises(ValueError):
        prepare.make_reference(source, expected_full=N_FULL, expected_ery=N_ERY)


@pytest.mark.parametrize('model_time', [True, False])
@pytest.mark.integration
def test_real_official_four_metrics(reference, tmp_path, model_time):
    _, pred = source_and_prediction()
    if not model_time: del pred.obs['latent_time']
    aligned, details = align(reference, pred)
    scores, transitions = run.official_evaluation(aligned, tmp_path)
    assert set(scores) == {'cbdir', 'icvcoh', 'cto', 'tsc'}
    assert all(np.isfinite(list(scores.values())))
    assert all(-1 <= scores[k] <= 1 for k in ('cbdir', 'icvcoh', 'tsc'))
    assert 0 <= scores['cto'] <= 1
    if model_time:
        assert scores['cto'] == 1.
        assert scores['tsc'] > .98
    assert details['time_kind'] == ('model_latent_time' if model_time else 'velocity_pseudotime')
    assert len(pd.read_csv(tmp_path / 'cell_times.csv')) == N_ERY
    for metric in scores:
        assert (tmp_path / 'evaluation' / f'{metric}_df.csv').is_file()
    assert len(pd.read_csv(tmp_path / 'cbdir_transitions.csv')) == 4
    write_json(tmp_path / 'test_scores.json', {'scores': scores, 'time_transitions': transitions})


def test_missing_explicit_time(reference):
    _, pred = source_and_prediction()
    with pytest.raises(ValueError, match='missing model time'):
        align(reference, pred, time_key='not_there')


@pytest.mark.integration
def test_main_outputs_and_no_overwrite(reference, tmp_path, monkeypatch):
    import functools
    import json
    _, pred = source_and_prediction()
    pred.write_h5ad(tmp_path / 'input.h5ad')
    reference.write_h5ad(tmp_path / 'reference.h5ad')
    before = sha256(tmp_path / 'input.h5ad')
    monkeypatch.setattr(C, 'DATA', tmp_path)
    monkeypatch.setattr(run, 'align_prediction', functools.partial(
        run.align_prediction, expected_full=N_FULL, expected_ery=N_ERY))
    monkeypatch.setattr(sys, 'argv', ['run.py', '--_worker', '--evaluation', 'full', '--method', 'synthetic',
                                    '--prediction', str(tmp_path / 'input.h5ad'),
                                    '--reference', str(tmp_path / 'reference.h5ad')])
    run.main()
    result = tmp_path / 'results' / 'synthetic'
    metadata = json.loads((result / 'metadata.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata['veloev_commit'] == C.VELOEV_COMMIT
    assert metadata['evaluation_n_cells'] == N_ERY
    assert metadata['time_kind'] == 'model_latent_time'
    assert (result / 'run.log').stat().st_size > 0
    assert (result / 'metrics.csv').is_file()
    assert sha256(tmp_path / 'input.h5ad') == before
    with pytest.raises(FileExistsError):
        run.main()


def test_shape_rejected_by_anndata():
    _, pred = source_and_prediction()
    with pytest.raises(ValueError):
        pred.layers['velocity'] = np.ones((pred.n_obs, pred.n_vars - 1))


@pytest.mark.integration
def test_pseudotime_reproducible(reference, tmp_path):
    _, pred = source_and_prediction()
    del pred.obs['latent_time']
    records = []
    for name in ('first', 'second'):
        directory = tmp_path / name
        directory.mkdir()
        aligned, _ = align(reference, pred)
        scores, _ = run.official_evaluation(aligned, directory)
        records.append((scores, pd.read_csv(directory / 'cell_times.csv').inferred_time.to_numpy()))
    assert records[0][0] == records[1][0]
    np.testing.assert_array_equal(records[0][1], records[1][1])


def test_failure_record(reference, tmp_path, monkeypatch):
    import json
    _, pred = source_and_prediction()
    pred.write_h5ad(tmp_path / 'input.h5ad')
    reference.write_h5ad(tmp_path / 'reference.h5ad')
    monkeypatch.setattr(C, 'DATA', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['run.py', '--_worker', '--evaluation', 'full', '--method', 'bad_count',
                                    '--prediction', str(tmp_path / 'input.h5ad'),
                                    '--reference', str(tmp_path / 'reference.h5ad')])
    # Production CLI must reject synthetic cell counts; no weakening of the CLI guard.
    with pytest.raises(ValueError, match='9815'):
        run.main()
    result = tmp_path / 'results' / 'bad_count'
    assert json.loads((result / 'metadata.json').read_text())['status'] == 'failed'
    assert not (result / 'metrics.csv').exists()
    assert '9815' in (result / 'run.log').read_text()
