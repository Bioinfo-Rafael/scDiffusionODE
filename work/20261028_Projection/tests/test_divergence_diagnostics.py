import numpy as np
import torch
from analysis.diagnostics import (scalar_selection, analyze_overview, method_summary,
                                  stratified_sample, representative_indices, analyze_genes)
from analysis.surface_divergence import TorchDerivatives
from Hodge.projector import HodgeProjector
from scalar_fixtures import PlaneScalar


def test_full_population_median_exclusions_and_failure_cache(synthetic, tmp_path):
    p = HodgeProjector(lambda x: -np.asarray(x), synthetic)
    class SupportPlane(PlaneScalar):
        support_radius = 1.
        def distribution_distance(self, x):
            return 10*(np.atleast_2d(x)[:, 1] > 2).astype(float)
    p.scalar = SupportPlane()
    p.metadata = {'eligible_cells': synthetic.obs_names[1:].tolist()}
    synthetic.obs['celltype'] = 'test'
    values, eligible, selection = scalar_selection(p, synthetic)
    assert not eligible[0]
    assert selection['c'] == np.median(synthetic.X[1:, 0])
    sample = np.arange(20)
    derivatives = TorchDerivatives(lambda x: -x)
    table, points = analyze_overview(p, derivatives, synthetic, sample, values, eligible, selection['c'],
                                    tmp_path, probes=8)
    assert table.iloc[0].status == 'scalar_ineligible'
    assert (table.status == 'retraction_failed').any()
    assert (table.status == 'not_sampled').any()
    assert table.loc[table.status != 'ok', 'surface_divergence'].isna().all()
    summary = method_summary(table, selection)
    assert summary['attempted_retractions'] == 19
    assert summary['retraction_success_count'] < 19
    class NoDerivatives:
        def value(self, _): raise AssertionError('Must reuse cached results')
    again, restored_points = analyze_overview(p, NoDerivatives(), synthetic, sample, values, eligible,
                                              selection['c'], tmp_path, probes=8)
    np.testing.assert_allclose(again.surface_divergence, table.surface_divergence, equal_nan=True)
    assert set(points) == set(restored_points)


def test_representatives_prioritize_same_successful_cells(synthetic, tmp_path):
    synthetic.obs['celltype'] = np.repeat(['a', 'b'], 40)
    p = HodgeProjector(lambda x: -np.asarray(x), synthetic)
    p.scalar = PlaneScalar(); p.metadata = {'eligible_cells': synthetic.obs_names.tolist()}
    values, eligible, selection = scalar_selection(p, synthetic)
    sample = stratified_sample(synthetic.obs.celltype, 3, 4, ['a', 'b'])
    d = TorchDerivatives(lambda x: -x)
    table, points = analyze_overview(p, d, synthetic, sample, values, eligible, selection['c'], tmp_path/'overview', probes=4)
    selected = representative_indices({'hodge': table, 'cellrank': table.copy()}, ['a', 'b'], 1, 42)
    np.testing.assert_array_equal(selected['hodge'], selected['cellrank'])
    genes, summary, kept = analyze_genes(p, d, table, points, selected['hodge'], synthetic.var_names, tmp_path/'genes')
    assert genes['total'].shape == (2, 3) and (summary.status == 'ok').all()
    np.testing.assert_allclose(genes['total'].sum(1), -2)
    class NoDerivatives:
        def value(self, _): raise AssertionError('Must reuse cached exact contributions')
    restored, _, _ = analyze_genes(p, NoDerivatives(), table, points, selected['hodge'], synthetic.var_names, tmp_path/'genes')
    np.testing.assert_array_equal(restored['total'], genes['total'])
