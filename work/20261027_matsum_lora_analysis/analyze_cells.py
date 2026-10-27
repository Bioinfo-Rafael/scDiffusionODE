"""Cell-dependent C_ck=a_ck B_k and W_c=W0+Σ C_ck. Exact component moments and batched gene rankings."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from artifacts import groups, write_json, log
from numerics import coefficient_moments, component_moments, relative_std, gene_strength, sample_by_type
import plots


def batched_gene_rankings(a, adata, basis, base, field, output, cfg):
    """Exact all-cell mean incoming/outgoing |W| scores, in bounded batches."""
    labels = adata.obs.celltype.astype(str).to_numpy()
    names = ['ALL'] + sorted(set(labels))
    group_sums = {name: np.zeros((2, len(base)), dtype=np.float64) for name in names}
    counts = {name: 0 for name in names}
    device = next(field.parameters()).device
    b = torch.as_tensor(basis, device=device, dtype=torch.float64)
    w0 = torch.as_tensor(base, device=device, dtype=torch.float64)
    batch_size = cfg['matrix_batch_size']
    with torch.inference_mode():
        for start in range(0, len(a), batch_size):
            end = min(start + batch_size, len(a))
            coeff = torch.as_tensor(a[start:end], device=device, dtype=torch.float64)
            # Exact all-cell ranking from mean_c Σ|W_c,ij|, not |mean_c W_c,ij|.
            weights = (coeff @ b.reshape(8, -1)).reshape(-1, len(base), len(base)) + w0
            weights = weights.abs()
            weights.diagonal(dim1=1, dim2=2).zero_()
            incoming = weights.sum(dim=2).cpu().numpy()
            outgoing = weights.sum(dim=1).cpu().numpy()
            values = np.stack([incoming, outgoing], axis=1)
            group_sums['ALL'] += values.sum(axis=0); counts['ALL'] += len(values)
            for label in set(labels[start:end]):
                selected = labels[start:end] == label
                group_sums[label] += values[selected].sum(axis=0); counts[label] += int(selected.sum())
            if start == 0 or end == len(a) or start // 4096 != end // 4096:
                log(f'Gene rankings from W: {end}/{len(a)}')
    rankings = []
    for label in names:
        strengths = group_sums[label] / counts[label]
        for i, gene in enumerate(adata.var_names):
            rankings.append(dict(group=label, n=counts[label], gene_id=gene,
                                 mean_cell_incoming=strengths[0, i], mean_cell_outgoing=strengths[1, i]))
    result = pd.DataFrame(rankings)
    for direction in ('incoming', 'outgoing'):
        result[direction + '_rank'] = result.groupby('group')['mean_cell_' + direction].rank(method='min', ascending=False)
    result.to_csv(Path(output) / 'effective_gene_rankings.csv', index=False)
    result[(result.incoming_rank <= 30) | (result.outgoing_rank <= 30)].to_csv(Path(output) / 'top30_effective_genes.csv', index=False)
    wide = result[result.group != 'ALL'].pivot(index='group', columns='gene_id', values='mean_cell_outgoing')
    variability = pd.DataFrame({'std_of_celltype_mean_outgoing': wide.std(ddof=0),
                                'range_of_celltype_mean_outgoing': wide.max() - wide.min()})
    variability.sort_values('std_of_celltype_mean_outgoing', ascending=False).to_csv(Path(output) / 'genes_varying_by_celltype.csv')


def group_weight_moments(a, labels, basis, base, genes, output, cfg):
    """Requested C moments only, stored losslessly as factors plus shared B."""
    output = Path(output)
    global_a = coefficient_moments(a)
    global_c_mean, global_c_std = component_moments(global_a['mean'], global_a['std'], basis)
    global_c_ratio = np.stack([relative_std(global_c_mean[k], global_c_std[k], cfg['near_zero_mean_relative'],
                                          cfg['near_zero_mean_absolute'])[0] for k in range(8)])
    grouped = list(groups(labels))
    max_coefficient = max(np.abs(a[indices].mean(axis=0)).max() for _, _, indices in grouped)
    limit_c = max(max_coefficient * np.abs(basis).max(), 1e-15)
    validity, gene_rows = [], []
    for slug, label, indices in grouped:
        log(f'Exact cell-group matrix moments: {label}, n={len(indices)}')
        folder = output / 'groups' / slug; folder.mkdir(parents=True, exist_ok=True)
        moments = coefficient_moments(a[indices])
        mean_w = base + np.einsum('k,kij->ij', moments['mean'], basis)
        incoming, outgoing = gene_strength(mean_w)
        pd.DataFrame({'gene_id': genes, 'incoming_abs_mean_matrix': incoming,
                      'outgoing_abs_mean_matrix': outgoing}).sort_values(
            'outgoing_abs_mean_matrix', ascending=False).to_csv(folder / 'mean_W_gene_rankings.csv', index=False)
        mean_c, std_c = component_moments(moments['mean'], moments['std'], basis)
        ratio_c = []
        for k in range(8):
            ratio, valid, threshold = relative_std(mean_c[k], std_c[k], cfg['near_zero_mean_relative'], cfg['near_zero_mean_absolute'])
            ratio_c.append(ratio)
            validity.append(dict(group=label, matrix=f'C{k + 1}', entries=valid.size, undefined=int((~valid).sum()), threshold=threshold))
            # E|a_k B_k| is exact and cheap. Rankings for mean signed matrices are also saved.
            incoming, outgoing = gene_strength(basis[k])
            for i, gene in enumerate(genes):
                gene_rows.append(dict(group=label, component=k + 1, gene_id=gene,
                    mean_cell_incoming=moments['mean_abs'][k] * incoming[i], mean_cell_outgoing=moments['mean_abs'][k] * outgoing[i],
                    mean_matrix_incoming=abs(moments['mean'][k]) * incoming[i], mean_matrix_outgoing=abs(moments['mean'][k]) * outgoing[i]))
        ratio_c = np.asarray(ratio_c)
        np.savez_compressed(folder / 'matrix_moments.npz',
                            coefficient_mean=moments['mean'], coefficient_std=moments['std'],
                            coefficient_covariance=moments['covariance'], coefficient_mean_absolute=moments['mean_abs'])
        write_json(folder / 'definition.json', dict(group=label, n=len(indices), ddof=0, C_mean='mean(a_k)*B_k',
                   C_std='std(a_k)*abs(B_k)', C_storage='exact factors stored in matrix_moments.npz; B in basis/shared_matrices.npz'))
        plots.matrix_panels(mean_c, folder / 'mean_C_full.png', f'{label}: mean signed a_k B_k', limit_c)
        for name, values, reference in [('std', std_c, global_c_std), ('std_over_mean', ratio_c, global_c_ratio)]:
            plots.hist_panels(values, folder / f'C_{name}_histogram.png', f'{label}: C {name} across cells; entries pooled',
                              bins=cfg['histogram_bins'], reference=None if slug == 'all' else reference)
    pd.DataFrame(validity).to_csv(output / 'ratio_validity.csv', index=False)
    ranks = pd.DataFrame(gene_rows)
    for direction in ('incoming', 'outgoing'):
        ranks[direction + '_rank'] = ranks.groupby(['group', 'component'])['mean_cell_' + direction].rank(ascending=False, method='min')
    ranks.to_csv(output / 'component_gene_rankings.csv.gz', index=False)


def sampled_cell_spectra(a, obs, basis_output, output, cfg):
    """Requested C spectra: abs(a_ck) times ordinary B spectrum, exactly."""
    output = Path(output) / 'sampled_graphs'; output.mkdir(parents=True, exist_ok=True)
    labels = obs.celltype.astype(str).to_numpy()
    selected = sample_by_type(labels, cfg['spectral_cells_per_type'], cfg['seed'])
    obs.iloc[selected].to_csv(output / 'sampled_cells.csv', index_label='cell_id')
    ordinary_basis = []
    for k in range(8):
        with np.load(Path(basis_output) / f'B{k + 1}_graph.npz') as z:
            ordinary_basis.append(z['ordinary_eigenvalues'])
    ordinary_basis = np.asarray(ordinary_basis)
    for number, index in enumerate(selected):
        values = np.abs(a[index, :, None]) * ordinary_basis
        np.savez_compressed(output / f'cell_{number:04d}.npz', cell_id=str(obs.index[index]),
                            coefficients=a[index], C_ordinary_eigenvalues=values)
        plots.spectra(values, output / f'cell_{number:04d}_C_ordinary.png',
                      f'{obs.index[index]}: C_k ordinary spectra (scaled B_k spectra)')
    write_json(output / 'definition.json', dict(seed=cfg['seed'], per_type_requested=cfg['spectral_cells_per_type'],
               component_spectra='ordinary spectrum(C_ck)=abs(a_ck)*spectrum(B_k)',
               graph='abs((C+C.T)/2), diagonal removed; signed direction information is discarded'))


def run_cell_analysis(a, adata, basis, base, field, basis_output, output, cfg):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    labels = adata.obs.celltype.astype(str).to_numpy()
    group_weight_moments(a, labels, basis, base, adata.var_names, output, cfg)
    batched_gene_rankings(a, adata, basis, base, field, output, cfg)
    sampled_cell_spectra(a, adata.obs, basis_output, output, cfg)
