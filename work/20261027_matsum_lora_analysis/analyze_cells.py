"""Cell-dependent C_ck=a_ck B_k and W_c=W0+Σ C_ck. Exact moments, batched effects."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from artifacts import groups, write_json, log
from numerics import coefficient_moments, component_moments, effective_moments, relative_std, gene_strength, sample_by_type
from spectral import spectrum
import plots


def batched_cell_effects(a, adata, basis, base, field, output, cfg):
    """All cells: E|W| gene strengths, preactivation and component-ablation effects.

    Allocate at most matrix_batch_size × d × d; keep only per-cell eight-component
    norms and group sums of gene strengths, never all cell-specific matrices.
    """
    labels = adata.obs.celltype.astype(str).to_numpy()
    names = ['ALL'] + sorted(set(labels))
    group_sums = {name: np.zeros((2, len(base)), dtype=np.float64) for name in names}
    counts = {name: 0 for name in names}
    device = next(field.parameters()).device
    b = torch.as_tensor(basis, device=device, dtype=torch.float64)
    w0 = torch.as_tensor(base, device=device, dtype=torch.float64)
    bias = field.b.detach().to(dtype=torch.float64)
    norms = np.empty((len(a), 8)); removed = np.empty_like(norms)
    base_norms = np.empty(len(a))
    matrix_norms = np.empty(len(a))
    batch_size = cfg['matrix_batch_size']
    with torch.inference_mode():
        for start in range(0, len(a), batch_size):
            end = min(start + batch_size, len(a))
            coeff = torch.as_tensor(a[start:end], device=device, dtype=torch.float64)
            block = adata.X[start:end]
            block = block.toarray() if hasattr(block, 'toarray') else np.asarray(block)
            x = torch.as_tensor(block, device=device, dtype=torch.float64)
            # C_k x and nonlinear leave-one-component-out effect; decay cancels.
            contributions = torch.einsum('kij,bj->bki', b, x) * coeff[:, :, None]
            base_x = x @ w0.T
            preactivation = base_x + contributions.sum(dim=1) + bias
            full_production = torch.nn.functional.softplus(preactivation)
            ablated = torch.nn.functional.softplus(preactivation[:, None, :] - contributions)
            norms[start:end] = torch.linalg.vector_norm(contributions, dim=2).cpu().numpy()
            removed[start:end] = torch.linalg.vector_norm(full_production[:, None, :] - ablated, dim=2).cpu().numpy()
            base_norms[start:end] = torch.linalg.vector_norm(base_x, dim=1).cpu().numpy()
            # Exact all-cell ranking from mean_c Σ|W_c,ij|, not |mean_c W_c,ij|.
            weights = (coeff @ b.reshape(8, -1)).reshape(-1, len(base), len(base)) + w0
            matrix_norms[start:end] = torch.linalg.vector_norm(weights.reshape(end - start, -1), dim=1).cpu().numpy()
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
                log(f'Cell matrices/effects: {end}/{len(a)}')
    frame = adata.obs[['celltype', 'stage']].copy()
    for k in range(8):
        frame[f'C{k + 1}_frobenius'] = np.abs(a[:, k]) * np.linalg.norm(basis[k])
        frame[f'C{k + 1}_x_L2'] = norms[:, k]
        frame[f'C{k + 1}_ablation_L2'] = removed[:, k]
    frame['W0_x_L2'] = base_norms; frame['W_frobenius'] = matrix_norms
    frame.to_csv(Path(output) / 'cell_effects.csv.gz', index_label='cell_id')
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
    means = frame.groupby('celltype', observed=True).mean(numeric_only=True)
    means.to_csv(Path(output) / 'mean_effects_by_celltype.csv')
    for suffix in ('frobenius', 'x_L2', 'ablation_L2'):
        columns = [f'C{k + 1}_{suffix}' for k in range(8)]
        plots.stacked(means[columns].to_numpy(), means.index, Path(output) / f'component_{suffix}.png',
                      f'Mean component {suffix}; nonadditive magnitudes, not fractions of total output')
    wide = result[result.group != 'ALL'].pivot(index='group', columns='gene_id', values='mean_cell_outgoing')
    variability = pd.DataFrame({'std_of_celltype_mean_outgoing': wide.std(ddof=0),
                                'range_of_celltype_mean_outgoing': wide.max() - wide.min()})
    variability.sort_values('std_of_celltype_mean_outgoing', ascending=False).to_csv(Path(output) / 'genes_varying_by_celltype.csv')


def group_weight_moments(a, labels, basis, base, genes, zoom, output, cfg):
    """Save exact W moments; C moments are stored as coefficient factors + shared B."""
    output = Path(output)
    global_a = coefficient_moments(a)
    global_c_mean, global_c_std = component_moments(global_a['mean'], global_a['std'], basis)
    global_w_mean, global_w_std = effective_moments(global_a['mean'], global_a['covariance'], basis, base)
    global_c_ratio = np.stack([relative_std(global_c_mean[k], global_c_std[k], cfg['near_zero_mean_relative'],
                                          cfg['near_zero_mean_absolute'])[0] for k in range(8)])
    global_w_ratio = relative_std(global_w_mean, global_w_std, cfg['near_zero_mean_relative'], cfg['near_zero_mean_absolute'])[0]
    grouped = list(groups(labels))
    max_coefficient = max(np.abs(a[indices].mean(axis=0)).max() for _, _, indices in grouped)
    limit_c = max(max_coefficient * np.abs(basis).max(), 1e-15)
    limit_w = max(float(np.abs(base).max() + max_coefficient * np.abs(basis).sum(axis=0).max()), 1e-15)
    within_variance = np.zeros_like(base); between_variance = np.zeros_like(base)
    validity, gene_rows = [], []
    for slug, label, indices in grouped:
        log(f'Exact cell-group matrix moments: {label}, n={len(indices)}')
        folder = output / 'groups' / slug; folder.mkdir(parents=True, exist_ok=True)
        moments = coefficient_moments(a[indices])
        mean_c, std_c = component_moments(moments['mean'], moments['std'], basis)
        mean_w, std_w = effective_moments(moments['mean'], moments['covariance'], basis, base)
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
        ratio_w, valid_w, threshold = relative_std(mean_w, std_w, cfg['near_zero_mean_relative'], cfg['near_zero_mean_absolute'])
        validity.append(dict(group=label, matrix='W', entries=valid_w.size, undefined=int((~valid_w).sum()), threshold=threshold))
        np.savez_compressed(folder / 'matrix_moments.npz', mean_W=mean_w, std_W=std_w, std_over_mean_W=ratio_w,
                            coefficient_mean=moments['mean'], coefficient_std=moments['std'],
                            coefficient_covariance=moments['covariance'], coefficient_mean_absolute=moments['mean_abs'])
        write_json(folder / 'definition.json', dict(group=label, n=len(indices), ddof=0, C_mean='mean(a_k)*B_k',
                   C_std='std(a_k)*abs(B_k)', C_storage='exact factors stored in matrix_moments.npz; B in basis/shared_matrices.npz'))
        plots.matrix_panels(mean_c, folder / 'mean_C_full.png', f'{label}: mean signed a_k B_k', limit_c)
        plots.matrix_panels(mean_c[:, zoom][:, :, zoom], folder / 'mean_C_zoom.png', f'{label}: mean signed a_k B_k', limit_c, np.asarray(genes)[zoom])
        plots.heatmap(mean_w, folder / 'mean_W_full.png', f'{label}: mean W; target rows / source columns', limit=limit_w)
        plots.heatmap(mean_w[np.ix_(zoom, zoom)], folder / 'mean_W_zoom.png', f'{label}: mean W',
                      labels=(np.asarray(genes)[zoom], np.asarray(genes)[zoom]), limit=limit_w)
        for name, values, reference in [('std', std_c, global_c_std), ('std_over_mean', ratio_c, global_c_ratio),
                                        ('std_over_abs_mean', np.abs(ratio_c), np.abs(global_c_ratio))]:
            plots.hist_panels(values, folder / f'C_{name}_histogram.png', f'{label}: C {name} across cells; entries pooled',
                              bins=cfg['histogram_bins'], reference=None if slug == 'all' else reference)
        for name, values, reference in [('std', std_w, global_w_std), ('std_over_mean', ratio_w, global_w_ratio),
                                        ('std_over_abs_mean', np.abs(ratio_w), np.abs(global_w_ratio))]:
            plots.histogram(values, folder / f'W_{name}_histogram.png', f'{label}: W {name} across cells; d*d entries',
                            bins=cfg['histogram_bins'], reference=None if slug == 'all' else reference)
        incoming, outgoing = gene_strength(mean_w)
        pd.DataFrame({'gene_id': genes, 'incoming_abs_mean_matrix': incoming, 'outgoing_abs_mean_matrix': outgoing}).sort_values(
            'outgoing_abs_mean_matrix', ascending=False).to_csv(folder / 'mean_W_gene_rankings.csv', index=False)
        if slug != 'all':
            p = len(indices) / len(a)
            within_variance += p * std_w**2
            between_variance += p * (mean_w - global_w_mean)**2
    total = global_w_std**2
    if not np.allclose(total, within_variance + between_variance, rtol=1e-7, atol=1e-12):
        raise ValueError('Matrix variance decomposition failed')
    eta = np.divide(between_variance, total, out=np.full_like(total, np.nan), where=total > 0)
    np.savez_compressed(output / 'W_variance_decomposition.npz', total=total, within=within_variance, between=between_variance, eta_squared=eta)
    plots.histogram(eta, output / 'W_eta_squared_histogram.png', 'Fraction of W variance explained by celltype means (association)')
    pd.DataFrame(validity).to_csv(output / 'ratio_validity.csv', index=False)
    ranks = pd.DataFrame(gene_rows)
    for direction in ('incoming', 'outgoing'):
        ranks[direction + '_rank'] = ranks.groupby(['group', 'component'])['mean_cell_' + direction].rank(ascending=False, method='min')
    ranks.to_csv(output / 'component_gene_rankings.csv.gz', index=False)


def sampled_cell_spectra(a, obs, basis, base, basis_output, output, cfg):
    """C spectra follow |a_k| scaling exactly; compute W spectra for balanced real cells."""
    output = Path(output) / 'sampled_graphs'; output.mkdir(parents=True, exist_ok=True)
    labels = obs.celltype.astype(str).to_numpy()
    selected = sample_by_type(labels, cfg['spectral_cells_per_type'], cfg['seed'])
    obs.iloc[selected].to_csv(output / 'sampled_cells.csv', index_label='cell_id')
    ordinary_basis, normalized_basis = [], []
    for k in range(8):
        with np.load(Path(basis_output) / f'B{k + 1}_graph.npz') as z:
            ordinary_basis.append(z['ordinary_eigenvalues']); normalized_basis.append(z['normalized_eigenvalues'])
    ordinary_basis, normalized_basis = np.array(ordinary_basis), np.array(normalized_basis)
    all_values = {'ordinary': [], 'normalized': []}
    for number, index in enumerate(selected):
        log(f'Sampled W spectrum {number + 1}/{len(selected)}: {obs.index[index]}')
        weight = base + np.einsum('k,kij->ij', a[index], basis)
        item = spectrum(weight, cfg['spectral_dimensions'])
        np.savez_compressed(output / f'cell_{number:04d}.npz', cell_id=str(obs.index[index]), coefficients=a[index],
                            C_ordinary_eigenvalues=np.abs(a[index, :, None]) * ordinary_basis,
                            C_normalized_eigenvalues=(a[index, :, None] != 0) * normalized_basis,
                            W_ordinary_eigenvalues=item['ordinary']['eigenvalues'], W_normalized_eigenvalues=item['normalized']['eigenvalues'])
        plots.spectra(np.abs(a[index, :, None]) * ordinary_basis, output / f'cell_{number:04d}_C_ordinary.png',
                      f'{obs.index[index]}: C_k ordinary spectra (scaled B_k spectra)')
        plots.spectra((a[index, :, None] != 0) * normalized_basis, output / f'cell_{number:04d}_C_normalized.png',
                      f'{obs.index[index]}: C_k normalized spectra (invariant for nonzero a_k)')
        for variant in all_values: all_values[variant].append(item[variant]['eigenvalues'])
    selected_labels = labels[selected]
    for variant, values in all_values.items():
        values = np.asarray(values)
        np.savez_compressed(output / f'{variant}_spectra.npz', eigenvalues=values, cell_ids=np.asarray(obs.index[selected], dtype=str), celltype=selected_labels)
        for slug, label, indices in groups(selected_labels):
            if slug == 'all': continue
            fig, ax = plots.plt.subplots(figsize=(9, 5))
            x = np.arange(values.shape[1]); sample = values[indices]
            for row in sample: ax.plot(x, row, color='gray', lw=.5, alpha=.6)
            ax.plot(x, np.median(sample, axis=0), color='tab:blue', label='median')
            ax.fill_between(x, np.quantile(sample, .25, axis=0), np.quantile(sample, .75, axis=0), alpha=.25)
            ax.set_title(f'{label}: {variant} W Laplacian; n={len(sample)}'); ax.set_xlabel('Sorted eigenvalue index'); ax.legend()
            plots.save(fig, output / f'{slug}_{variant}.png')
        fig, ax = plots.plt.subplots(figsize=(12, 7))
        for label in sorted(set(selected_labels)):
            median = np.median(values[selected_labels == label], axis=0)
            ax.plot(np.arange(len(median)), median, label=label)
        ax.set_title(f'{variant} W spectra: celltype medians, common axes')
        ax.set_xlabel('Sorted eigenvalue index'); ax.set_ylabel('Eigenvalue')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6)
        plots.save(fig, output / f'{variant}_celltype_medians.png')
    write_json(output / 'definition.json', dict(seed=cfg['seed'], per_type_requested=cfg['spectral_cells_per_type'],
               component_spectra='ordinary spectrum(C_ck)=abs(a_ck)*spectrum(B_k); normalized invariant unless a_ck=0',
               graph='abs((W+W.T)/2), diagonal removed; signed direction information is discarded'))


def run_cell_analysis(a, adata, basis, base, field, zoom, basis_output, output, cfg):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    labels = adata.obs.celltype.astype(str).to_numpy()
    group_weight_moments(a, labels, basis, base, adata.var_names, zoom, output, cfg)
    batched_cell_effects(a, adata, basis, base, field, output, cfg)
    sampled_cell_spectra(a, adata.obs, basis, base, basis_output, output, cfg)
