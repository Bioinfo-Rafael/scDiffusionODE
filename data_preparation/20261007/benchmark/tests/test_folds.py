"""Small synthetic tests; no dataset download, PCA/UMAP fit or model inference."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import config as C
import folds
import prepare
import run
import upstream
from common import digest_ids, sha256
from test_benchmark import N_FULL, N_ERY, source_and_prediction, fake_geometry

REPRO = C.HERE.parents[2] / 'work' / '20261009_reproduce'
spec = importlib.util.spec_from_file_location('reproduction_adapter', REPRO / 'reproduce.py')
reproduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduction)


def small_reference():
    source, pred = source_and_prediction()
    return prepare.make_reference(source, expected_full=N_FULL, expected_ery=N_ERY), pred


def test_official_stratified_test_indices():
    ref, _ = small_reference()
    parts = folds.split_reference(ref)
    expected = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for part, (train, test) in zip(parts, expected.split(np.zeros(N_ERY), ref.obs.celltype)):
        assert part.obs_names.tolist() == ref.obs_names[test].tolist()
        assert len(part) < len(train)
        assert part.obs.celltype.value_counts().max() - part.obs.celltype.value_counts().min() <= 1
    all_ids = [cell for part in parts for cell in part.obs_names]
    assert len(all_ids) == len(set(all_ids)) == N_ERY
    assert set(all_ids) == set(ref.obs_names)
    assert [p.obs_names.tolist() for p in parts] == [p.obs_names.tolist() for p in folds.split_reference(ref)]
    parts[0].X.data[:] = 0
    assert ref.X.sum() > 0 and parts[1].X.sum() > 0


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source, _ = source_and_prediction()
    source.obsm['X_pca'] = np.ones((N_FULL, 30)) * -99
    source.write_h5ad(tmp_path / 'source.h5ad')
    calls = []
    def geometry(part):
        assert not part.obsm and not part.obsp
        calls.append(part.obs_names.tolist())
        fake_geometry(part)
        part.obsm['X_pca'][:] = len(calls)
        return part
    monkeypatch.setattr(prepare, 'build_geometry', geometry)
    before = sha256(tmp_path / 'source.h5ad')
    directory = tmp_path / 'reference'
    manifest = folds.prepare_folds(tmp_path / 'source.h5ad', directory, 'normalized',
                                   expected_full=N_FULL, expected_ery=N_ERY)
    assert sha256(tmp_path / 'source.h5ad') == before
    assert calls == [f['cell_ids'] for f in manifest['folds']]
    assert len(calls) == 3
    return directory, manifest


def test_independent_preparation_and_manifests(prepared, tmp_path):
    directory, manifest = prepared
    assert folds.validate_manifest(directory, N_ERY)['status'] == 'complete'
    for i, item in enumerate(manifest['folds']):
        ref = ad.read_h5ad(directory / item['reference'])
        assert np.all(ref.obsm['X_pca'] == i + 1)
        assert ref.uns['benchmark']['fold'] == i
        assert (directory / f'fold_{i}' / 'hvg_ids.txt').read_text().splitlines() == item['hvg_ids']
    with pytest.raises(FileExistsError):
        folds.prepare_folds(tmp_path / 'source.h5ad', directory, 'normalized')


@pytest.mark.parametrize('full', [True, False])
def test_fold_velocity_time_alignment(prepared, full):
    directory, manifest = prepared
    _, pred = source_and_prediction()
    if not full:
        pred = pred[manifest['erythroid_cell_ids']].copy()
    original = pred.copy()
    pred = pred[::-1, ::-1].copy()
    for item in manifest['folds']:
        ref = ad.read_h5ad(directory / item['reference'])
        aligned, _ = run.align_prediction(ref, pred, ref.var_names, 'velocity',
                                          expected_full=N_FULL, expected_ery=ref.n_obs)
        matched = original[ref.obs_names, ref.var_names]
        np.testing.assert_array_equal(aligned.layers['candidate_velocity'], matched.layers['velocity'])
        np.testing.assert_array_equal(aligned.obs.candidate_time, matched.obs.latent_time)
        # A partial full-trained prediction must not masquerade as a complete erythroid prediction.
        with pytest.raises(ValueError, match='cell IDs'):
            run.align_prediction(ref, pred[ref.obs_names], ref.var_names, 'velocity',
                                 expected_full=N_FULL, expected_ery=ref.n_obs)


def test_fold_missing_inputs(prepared):
    directory, manifest = prepared
    _, pred = source_and_prediction()
    ref = ad.read_h5ad(directory / manifest['folds'][0]['reference'])
    for bad in (pred[1:].copy(), pred[:, 1:].copy()):
        with pytest.raises(ValueError):
            run.align_prediction(ref, bad, ref.var_names, 'velocity', expected_full=N_FULL, expected_ery=ref.n_obs)
    pred.layers['velocity'][0, 0] = np.nan
    with pytest.raises(ValueError, match='NaN/Inf'):
        run.align_prediction(ref, pred, ref.var_names, 'velocity', expected_full=N_FULL, expected_ery=ref.n_obs)


@pytest.mark.parametrize('bad', ['duplicate_fold', 'overlap', 'hash', 'split'])
def test_invalid_manifest(prepared, bad):
    directory, manifest = prepared
    if bad == 'duplicate_fold': manifest['folds'][1]['fold'] = 0
    if bad == 'overlap': manifest['folds'][1]['cell_ids'][0] = manifest['folds'][0]['cell_ids'][0]
    if bad == 'hash': manifest['folds'][0]['sha256'] = 'bad'
    if bad == 'split': manifest['split']['random_state'] = 1234
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): folds.validate_manifest(directory, N_ERY)


def test_aggregation_and_errors():
    rows = [dict(fold=i, **{key: float(i) for key in folds.METRICS}) for i in range(3)]
    actual = folds.aggregate(rows)
    np.testing.assert_allclose(actual['mean'], 1)
    np.testing.assert_allclose(actual['std'], np.std([0, 1, 2], ddof=0))
    assert (actual.ddof == 0).all()
    for bad in (rows[:2], [rows[0], rows[0], rows[2]]):
        with pytest.raises(ValueError): folds.aggregate(bad)
    rows[0]['tsc'] = np.nan
    with pytest.raises(ValueError): folds.aggregate(rows)


def test_three_fold_orchestration(prepared, tmp_path, monkeypatch):
    directory, _ = prepared
    _, pred = source_and_prediction()
    prediction = tmp_path / 'prediction.h5ad'
    pred.write_h5ad(prediction)
    calls = []
    def evaluate(a, folder):
        calls.append(a.obs_names.tolist())
        a.write_h5ad(folder / 'mock_input.h5ad')
        return {k: float(len(calls)) for k in folds.METRICS}, [(6.5, 6.75)]
    monkeypatch.setattr(run, 'official_evaluation', evaluate)
    result = folds.evaluate_folds(directory, [prediction], tmp_path / 'result', method='test', expected_ery=N_ERY)
    assert len(calls) == 3 and len(set(sum(calls, []))) == N_ERY
    assert len(pd.read_csv(result / 'metrics_per_fold.csv')) == 3
    assert len(pd.read_csv(result / 'metrics_summary.csv')) == 4
    meta = json.loads((result / 'metadata.json').read_text())
    assert meta['status'] == 'complete' and not meta['training_repeated']
    with pytest.raises(FileExistsError):
        folds.evaluate_folds(directory, [prediction], result, method='test', expected_ery=N_ERY)


def test_partial_failure_retained(prepared, tmp_path, monkeypatch):
    directory, _ = prepared
    _, pred = source_and_prediction()
    pred.write_h5ad(tmp_path / 'prediction.h5ad')
    count = []
    def evaluate(a, folder):
        count.append(folder)
        if len(count) == 2: raise ValueError('synthetic failure')
        return {k: .5 for k in folds.METRICS}, []
    monkeypatch.setattr(run, 'official_evaluation', evaluate)
    result = tmp_path / 'failed'
    with pytest.raises(ValueError, match='synthetic failure'):
        folds.evaluate_folds(directory, [tmp_path / 'prediction.h5ad'], result, method='test', expected_ery=N_ERY)
    assert json.loads((result / 'metadata.json').read_text())['status'] == 'failed'
    assert len(pd.read_csv(result / 'metrics_per_fold.csv')) == 1
    assert not (result / 'metrics_summary.csv').exists()
    assert 'synthetic failure' in (result / 'fold_1' / 'run.log').read_text()


def test_raw_rejects_normalized_input():
    ref, _ = small_reference()
    with pytest.raises(ValueError, match='integer counts'): folds.raw_reference(ref, N_ERY)


def test_raw_preserves_library_size_annotations_discards_geometry():
    ref, _ = small_reference()
    ref.X.data[:] = np.floor(ref.X.data)
    for key in ('spliced', 'unspliced'):
        ref.layers[key].data[:] = np.floor(ref.layers[key].data)
    ref.obs['initial_size_spliced'] = np.arange(N_ERY) + 1000
    fake_geometry(ref)
    raw = folds.raw_reference(ref, N_ERY)
    assert raw.obs.equals(ref.obs)
    assert not raw.obsm and not raw.obsp and not raw.uns


def test_original_stochastic_body_call_order(tmp_path, monkeypatch):
    events = []
    a = SimpleNamespace(obs={}, write_h5ad=lambda path: events.append(('save', path.name)))
    def time(a, **kwargs):
        events.append(('pseudotime', kwargs))
        a.obs['scvelo_stc_velocity_pseudotime'] = np.array([.1, .9])
    namespace = dict(DATA_DIR=tmp_path, DATASET='dataset', SAVE_DATA=True,
                     sc=SimpleNamespace(read_h5ad=lambda path: a), gc=SimpleNamespace(collect=lambda: None),
                     scv=SimpleNamespace(tl=SimpleNamespace(
                         velocity=lambda a, **kw: events.append(('velocity', kw)),
                         velocity_graph=lambda a, **kw: events.append(('graph', kw)), velocity_pseudotime=time)))
    monkeypatch.setattr(upstream, 'namespace', lambda directory: namespace)
    upstream.stochastic_function(tmp_path)(2)
    assert events == [('velocity', dict(mode='stochastic', vkey='scvelo_stc_velocity')),
                      ('graph', dict(vkey='scvelo_stc_velocity')),
                      ('pseudotime', dict(vkey='scvelo_stc_velocity')),
                      ('save', 'adata_run_scvelo_stc_2.h5ad')]
    assert a.obs['scvelo_stc_time'] is a.obs['scvelo_stc_velocity_pseudotime']


def test_comparison_thresholds_and_negative_tsc():
    paper = pd.read_csv(REPRO / 'paper_expectations.csv')
    actual = paper.copy()
    actual.loc[0, 'cbdir'] += 5e-5
    result = reproduction.compare_scores(actual, paper)
    row = result[(result.metric == 'cbdir') & (result.fold == 0)].iloc[0]
    assert not row.within_1e_5 and row.within_1e_4
    assert result[(result.metric == 'tsc') & (result.fold == 2)].iloc[0].reproduced < 0
    with pytest.raises(ValueError): reproduction.compare_scores(actual.iloc[:2], paper)


def test_raw_adapter_preserves_time_mask_and_moments(prepared):
    directory, manifest = prepared
    ref = ad.read_h5ad(directory / manifest['folds'][0]['reference'])
    ref.uns['benchmark']['expression_scale'] = C.RAW_EXPRESSION_SCALE
    ref.layers['Ms'] = np.ones(ref.shape, dtype=np.float32)
    ref.layers['Mu'] = np.ones(ref.shape, dtype=np.float32) * 2
    _, full_pred = source_and_prediction()
    original = full_pred[ref.obs_names].copy()
    original.layers['scvelo_stc_velocity'] = original.layers['velocity'].copy()
    original.var['scvelo_stc_velocity_genes'] = np.arange(original.n_vars) % 2 == 0
    original.obs['scvelo_stc_time'] = original.obs.latent_time
    original.obs['scvelo_stc_velocity_pseudotime'] = original.obs.latent_time
    pred = reproduction.adapt_prediction(original, ref, 'test-artifact')
    aligned, details = run.align_prediction(ref, pred[::-1, ::-1], ref.var_names, 'scvelo_stc_velocity',
        'scvelo_stc_time', expected_full=N_FULL, expected_ery=ref.n_obs,
        inference_protocol='per-fold', time_kind='precomputed_velocity_pseudotime')
    np.testing.assert_array_equal(aligned.obs.candidate_time, original.obs.scvelo_stc_time)
    np.testing.assert_array_equal(aligned.var.candidate_velocity_genes, original.var.scvelo_stc_velocity_genes)
    np.testing.assert_array_equal(aligned.layers['Ms'], ref.layers['Ms'])
    assert details['time_kind'] == 'precomputed_velocity_pseudotime'


def test_source_snapshot_checksums():
    assert upstream.verify_sources()['commit'] == C.BENCHMARK_COMMIT


def test_original_notebook_cell_runs_each_fold_independently(tmp_path, monkeypatch):
    ref, _ = small_reference()
    parts = folds.split_reference(ref)
    calls = []
    def normalize(a, **kw):
        calls.append(('normalize', a.obs_names.tolist(), kw))
    def moments(a, **kw):
        calls.append(('moments', a.obs_names.tolist(), kw))
        fake_geometry(a)
    def umap(a, **kw):
        calls.append(('umap', a.obs_names.tolist(), kw))
    namespace = dict(DATA_DIR=tmp_path, DATASET='original', SAVE_DATA=True, SEED=C.SEED,
        gc=SimpleNamespace(collect=lambda: None),
        utils=SimpleNamespace(fill_in_neighbors_indices=lambda a: None),
        sc=SimpleNamespace(pp=SimpleNamespace(highly_variable_genes=lambda a, **kw: None),
                           tl=SimpleNamespace(umap=umap)),
        scv=SimpleNamespace(pp=SimpleNamespace(filter_and_normalize=normalize, moments=moments)))
    monkeypatch.setattr(upstream, 'namespace', lambda directory: namespace)
    upstream.preprocess(parts, tmp_path / 'original')
    assert [c[0] for c in calls] == ['normalize', 'moments', 'umap'] * 3
    for i, part in enumerate(parts):
        assert calls[i * 3][1] == part.obs_names.tolist()
        assert calls[i * 3][2] == dict(min_shared_counts=20, n_top_genes=2000)
        assert calls[i * 3 + 1][2] == dict(n_neighbors=30, n_pcs=30)
        assert calls[i * 3 + 2][2] == dict(random_state=C.SEED)
        assert (tmp_path / 'original' / 'processed' / f'adata_preprocessed_{i}.h5ad').is_file()


def test_official_postprocess_does_not_recompute_supplied_time(tmp_path, monkeypatch):
    import importlib
    import pickle
    from common import check_vendor
    from scipy import sparse
    check_vendor()
    module = importlib.import_module('veloev.postprocessing.postprocess')
    ref, pred = small_reference()
    fake_geometry(ref)
    aligned, _ = run.align_prediction(ref, pred, ref.var_names, 'velocity', expected_full=N_FULL, expected_ery=N_ERY)
    aligned.uns['neighbors']['indices'] = np.tile(np.arange(30), (N_ERY, 1))
    (tmp_path / 'processed').mkdir()
    aligned.write_h5ad(tmp_path / 'processed' / 'adata_run_candidate_full.h5ad')
    def graph(a, **kw):
        a.uns['candidate_velocity_graph'] = sparse.eye(a.n_obs, format='csr')
    def embedding(a, **kw):
        a.obsm['candidate_velocity_umap'] = np.zeros((a.n_obs, 2))
    def forbidden(*args, **kwargs):
        raise AssertionError('supplied pseudotime must not be recomputed')
    monkeypatch.setattr(module.scv.tl, 'velocity_graph', graph)
    monkeypatch.setattr(module.scv.tl, 'velocity_embedding', embedding)
    monkeypatch.setattr(module.scv.tl, 'velocity_pseudotime', forbidden)
    module.postprocess(methods=['candidate'], task='directional_temporal', k_fold=0,
                       cluster_key='celltype', time_key='stage_day', result_path=tmp_path, n_jobs=1)
    with (tmp_path / 'postprocess' / 'candidate_full.pkl').open('rb') as f:
        result = pickle.load(f)
    np.testing.assert_array_equal(result['method_time'], aligned.obs.candidate_time)


def test_input_comparison_is_id_based():
    source, _ = source_and_prediction()
    raw = source[:N_ERY].copy()
    result = reproduction.compare_inputs(raw[::-1, ::-1], source)
    assert result['same_erythroid_cell_set'] and not result['same_erythroid_cell_order']
    assert result['same_gene_set'] and not result['same_gene_order']
    assert result['layers']['spliced']['shared_raw_max_abs_difference'] == 0
    assert all(result['annotation_matches'].values())


def test_four_official_metrics_on_small_precomputed_fixture(tmp_path):
    """Real official metric calls only; no scVelo inference, graph or geometry computation."""
    import pickle
    from common import check_vendor
    check_vendor()
    from veloev.evaluation.evaluation import single_metric
    ref, pred = small_reference()
    path = tmp_path / 'postprocess'
    path.mkdir()
    times = pd.Series(pred.obs.loc[ref.obs_names, 'latent_time'], index=ref.obs_names)
    n = ref.n_obs
    post = dict(exp_emb=np.column_stack([times.to_numpy(), np.zeros(n)]),
                velocity_emb=np.tile([1., 0.], (n, 1)),
                neighbor_indices=(np.arange(n)[:, None] + np.arange(30)) % n,
                cell_label=ref.obs.celltype.astype(str), time_label=ref.obs.stage_day,
                method_time=times, pseudo_time=None)
    with (path / 'candidate_full.pkl').open('wb') as f:
        pickle.dump(post, f)
    days = sorted(ref.obs.stage_day.unique())
    scores = {}
    for metric in folds.METRICS:
        frame = single_metric(metric=metric, result_path=tmp_path, methods=['candidate'], k_fold=0,
                              cell_type_transitions=C.CELL_TYPE_TRANSITIONS,
                              time_transitions=list(zip(days[:-1], days[1:])))
        scores[metric] = float(frame.iloc[0, 1])
    for metric in ('cbdir', 'icvcoh', 'cto'):
        assert scores[metric] == pytest.approx(1.)
    assert .98 < scores['tsc'] <= 1.
