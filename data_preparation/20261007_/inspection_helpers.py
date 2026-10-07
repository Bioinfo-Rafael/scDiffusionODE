"""Read-only diagnostics. Dense conversion is restricted to bounded matched subsets."""
import numpy as np
import pandas as pd
from scipy import sparse, stats
from IPython.display import display
import matplotlib.pyplot as plt

SEED = 20261007
QUANTILES = [0.5, 0.9, 0.99, 0.999]
CANDIDATES = ['celltype', 'clusters', 'stage', 'sample', 'theiler', 'sequencing.batch']


def matrices(adata):
    if adata.X is not None:
        yield 'X', adata.X
    yield from adata.layers.items()


def structure(adata):
    print(adata)
    for attr in ['obs', 'var']:
        print(attr, getattr(adata, attr).columns.tolist())
    for attr in ['layers', 'obsm', 'varm', 'obsp', 'uns']:
        print(attr, list(getattr(adata, attr).keys()))
    print('raw:', None if adata.raw is None else adata.raw.shape)
    print('Count-named layers (names alone do not prove raw counts):',
          [k for k in adata.layers if any(s in k.lower() for s in ['count', 'raw'])])
    for key in ['log1p', 'neighbors', 'pca']:
        if key in adata.uns:
            print(f'uns[{key!r}] =', adata.uns[key])
    for key in ['X_pca', 'X_umap']:
        if key in adata.obsm:
            print(key, adata.obsm[key].shape)


def obs_report(frame):
    display(frame.head())
    display(pd.DataFrame([{'column': col, 'dtype': str(frame[col].dtype),
                           'unique': frame[col].nunique(), 'missing': frame[col].isna().sum()}
                          for col in frame]))
    for col in frame:
        print('\n', col, '(annotation candidate)' if col in CANDIDATES else '')
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            print('categories:', frame[col].cat.categories.tolist())
        display(frame[col].value_counts(dropna=False).head(30))
    print('Absent candidates:', [c for c in CANDIDATES if c not in frame])


def id_report(left, right, label):
    a, b = pd.Index(left), pd.Index(right)
    au, bu = a.unique(), b.unique()
    common = au.intersection(bu, sort=False)
    aonly, bonly = au.difference(bu, sort=False), bu.difference(au, sort=False)
    # Exclude every occurrence of ambiguous IDs from aligned comparisons, without editing data.
    unambiguous_a = a[~a.duplicated(keep=False)]
    unambiguous_b = b[~b.duplicated(keep=False)]
    aligned = unambiguous_a.intersection(unambiguous_b, sort=False)
    result = {'axis': label, 'full_count': len(a), 'ery_count': len(b),
              'full_unique': len(au), 'ery_unique': len(bu),
              'full_duplicate_extra_rows': int(a.duplicated().sum()),
              'ery_duplicate_extra_rows': int(b.duplicated().sum()),
              'full_has_duplicates': a.has_duplicates, 'ery_has_duplicates': b.has_duplicates,
              'common_unique': len(common),
              'ery_coverage_common_over_n': len(common) / len(b) if len(b) else np.nan,
              'full_coverage_common_over_n': len(common) / len(a) if len(a) else np.nan,
              'ery_row_membership_fraction': float(b.isin(au).mean()) if len(b) else np.nan,
              'ery_only_unique': len(bonly), 'full_only_unique': len(aonly),
              'same_order': a.equals(b), 'unambiguous_common': len(aligned)}
    display(pd.Series(result))
    print('ery coverage in full = common unique IDs / ery rows =', result['ery_coverage_common_over_n'])
    print('With duplicate IDs, this differs from row membership fraction.')
    for name, ids in [('common_ids', common), ('ery_only_ids', bonly), ('full_only_ids', aonly)]:
        print(name, ids[:20].tolist())
    return result, common, aligned


def compare_columns(left, right, ids, crosstabs=True):
    columns = left.columns.intersection(right.columns, sort=False)
    print('COMMON COLUMNS:', columns.tolist())
    if len(ids) == 0:
        print('No unambiguous shared IDs; column comparisons skipped.')
        return pd.DataFrame()
    records = []
    for col in columns:
        a = left.loc[ids, col].astype('string')
        b = right.loc[ids, col].astype('string')
        valid = a.notna() & b.notna()
        equal = a.eq(b).fillna(False) & valid
        mismatch = valid & ~equal
        records.append({'column': col, 'matches': int(equal.sum()),
                        'mismatches': int(mismatch.sum()), 'comparable': int(valid.sum()),
                        'agreement_nonmissing': float(equal.sum() / valid.sum()) if valid.any() else np.nan,
                        'one_missing': int((a.isna() ^ b.isna()).sum()),
                        'both_missing': int((a.isna() & b.isna()).sum())})
        examples = mismatch | (a.isna() ^ b.isna())
        if examples.any():
            print(col, 'disagreement examples')
            display(pd.DataFrame({'full': a[examples], 'ery': b[examples]}).head(20))
        if crosstabs and valid.any():
            n_a, n_b = a[valid].nunique(), b[valid].nunique()
            if n_a <= 100 and n_b <= 100 and n_a * n_b <= 2500:
                print(col, 'crosstab (nonmissing pairs)')
                display(pd.crosstab(a[valid], b[valid], rownames=['full'], colnames=['ery']))
            else:
                print(col, 'crosstab skipped: high cardinality', n_a, n_b)
    result = pd.DataFrame(records).set_index('column') if records else pd.DataFrame()
    display(result)
    return result


def describe_matrix(matrix, seed=SEED, sample_size=100_000):
    """Exact moments via bounded blocks; quantiles via uniform coordinate sampling.

    Nonzero sample uses a uniform priority reservoir over finite nonzero entries.
    Sparse implicit zeros are included in moments and coordinate quantiles.
    """
    rng = np.random.default_rng(seed)
    rows, cols = matrix.shape
    n = rows * cols
    sums = np.zeros(cols, dtype=np.float64)
    squares = np.zeros(cols, dtype=np.float64)
    gene_n = np.zeros(cols, dtype=np.int64)
    library = np.empty(rows, dtype=np.float64)
    total = total2 = 0.0
    nonzero = negative = nonfinite = 0
    low, high = np.inf, -np.inf
    reservoir = np.empty(0)
    priorities = np.empty(0)
    block_rows = max(1, min(256, 1_000_000 // max(cols, 1)))
    for start in range(0, rows, block_rows):
        stop = min(start + block_rows, rows)
        block = matrix[start:stop].astype(np.float64, copy=True)
        if sparse.issparse(block):
            block = block.tocsr()
            block.sum_duplicates()
            values = block.data
            finite = np.isfinite(values)
            bad = ~finite
            bad_cols = block.indices[bad]
            library[start:stop] = np.asarray(block.sum(axis=1)).ravel()
            gene_n += block.shape[0] - np.bincount(bad_cols, minlength=cols)
            nonfinite += int(bad.sum())
            finite_values = values[finite]
            if block.nnz < block.shape[0] * cols:
                low, high = min(low, 0), max(high, 0)
            block.data[bad] = 0
            sums += np.asarray(block.sum(axis=0)).ravel()
            squares += np.asarray(block.multiply(block).sum(axis=0)).ravel()
        else:
            values = np.asarray(block)
            finite = np.isfinite(values)
            library[start:stop] = values.sum(axis=1)
            gene_n += finite.sum(axis=0)
            nonfinite += int((~finite).sum())
            finite_values = values[finite]
            block[~finite] = 0
            sums += block.sum(axis=0)
            squares += np.square(block).sum(axis=0)
        if finite_values.size:
            low, high = min(low, finite_values.min()), max(high, finite_values.max())
            total += finite_values.sum(dtype=np.float64)
            total2 += np.square(finite_values).sum(dtype=np.float64)
            negative += int((finite_values < 0).sum())
        nz = finite_values[finite_values != 0]
        nonzero += nz.size
        # Keep the sample_size smallest independent random priorities seen so far.
        keys = rng.random(nz.size)
        if nz.size > sample_size:
            ix = np.argpartition(keys, sample_size - 1)[:sample_size]
            nz, keys = nz[ix], keys[ix]
        reservoir = np.concatenate([reservoir, nz])
        priorities = np.concatenate([priorities, keys])
        if reservoir.size > sample_size:
            ix = np.argpartition(priorities, sample_size - 1)[:sample_size]
            reservoir, priorities = reservoir[ix], priorities[ix]
    count = n - nonfinite
    mean = total / count if count else np.nan
    variance = max(total2 / count - mean * mean, 0) if count else np.nan
    # Sample coordinates, including implicit zeros, with replacement; never densify matrix.
    if n:
        rr = rng.integers(rows, size=sample_size)
        cc = rng.integers(cols, size=sample_size)
        sample = np.asarray(matrix[rr, cc]).ravel()
        sample = sample[np.isfinite(sample)]
    else:
        sample = np.empty(0)
    quant = np.quantile(sample, QUANTILES) if sample.size else np.full(4, np.nan)
    integer_fraction = float((np.abs(reservoir - np.rint(reservoir)) <= 1e-6).mean()) if reservoir.size else np.nan
    summary = {'shape': matrix.shape, 'dtype': str(matrix.dtype),
               'storage': 'sparse' if sparse.issparse(matrix) else 'dense',
               'min': low if count else np.nan, 'max': high if count else np.nan,
               'mean': mean, 'variance': variance, 'nonfinite_count': nonfinite,
               'nonzero_fraction_finite': nonzero / count if count else np.nan,
               'zero_fraction_finite': (count - nonzero) / count if count else np.nan,
               'negative_count': negative, 'has_negative': bool(negative),
               'median_estimate': quant[0], 'quantile_sample_n': len(sample),
               'nonzero_sample_n': len(reservoir), 'integer_like_fraction': integer_fraction,
               'fractional_fraction': 1 - integer_fraction,
               **{f'q{q}_estimate': v for q, v in zip(QUANTILES, quant)}}
    gene_mean = np.divide(sums, gene_n, out=np.full(cols, np.nan), where=gene_n > 0)
    gene_var = np.maximum(np.divide(squares, gene_n, out=np.full(cols, np.nan), where=gene_n > 0) - gene_mean**2, 0)
    return {'summary': summary, 'gene_mean': gene_mean, 'gene_var': gene_var,
            'library': library, 'nonzero_sample': reservoir}


def library_report(values):
    finite = values[np.isfinite(values)]
    return {'n': len(values), 'nonfinite': len(values) - len(finite),
            'mean': finite.mean() if len(finite) else np.nan,
            'std': finite.std() if len(finite) else np.nan,
            'median': np.median(finite) if len(finite) else np.nan,
            **{f'q{q}': np.quantile(finite, q) if len(finite) else np.nan
               for q in [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]}}


def positions(index, ids):
    # IDs passed here must have exactly one occurrence in each original index.
    wanted = set(ids)
    mapping = {value: i for i, value in enumerate(index) if value in wanted}
    return np.array([mapping[value] for value in ids], dtype=int)


def bounded_subset(matrix, rows, cols):
    if len(rows) > 500 or len(cols) > 500:
        raise ValueError('Dense inspection subsets are limited to 500 x 500')
    if sparse.issparse(matrix):
        # Paired broadcast indexing avoids a potentially huge intermediate row slice.
        small = matrix[np.asarray(rows)[:, None], np.asarray(cols)[None, :]]
        return small.toarray().astype(np.float64) if sparse.issparse(small) else np.asarray(small, dtype=np.float64)
    return np.asarray(matrix[np.ix_(rows, cols)], dtype=np.float64)


def agreement(a, b):
    a, b = a.ravel(), b.ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    x, y = a[finite], b[finite]
    variable = len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0
    ratio_mask = x != 0
    ratios = y[ratio_mask] / x[ratio_mask]
    return {'sample_pairs': len(a), 'finite_pairs': len(x),
            'pearson': stats.pearsonr(x, y).statistic if variable else np.nan,
            'spearman': stats.spearmanr(x, y).statistic if variable else np.nan,
            'MAE': np.mean(np.abs(x-y)) if len(x) else np.nan,
            'RMSE': np.sqrt(np.mean((x-y)**2)) if len(x) else np.nan,
            'exact_match_fraction': np.mean(x == y) if len(x) else np.nan,
            'both_nonzero_pairs': int(((x != 0) & (y != 0)).sum()),
            'either_nonzero_pairs': int(((x != 0) | (y != 0)).sum()),
            'exact_match_fraction_either_nonzero': np.mean(x[(x != 0) | (y != 0)] == y[(x != 0) | (y != 0)]) if ((x != 0) | (y != 0)).any() else np.nan,
            'scale_ratio_definition': 'second / first, first != 0',
            'scale_ratio_n': len(ratios),
            'scale_ratio_q01_q50_q99': np.quantile(ratios, [.01, .5, .99]).tolist() if len(ratios) else []}


def scatter(x, y, xlabel, ylabel, path, log=False):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    if log:
        keep &= (x > 0) & (y > 0)
    x, y = x[keep], y[keep]
    rng = np.random.default_rng(SEED)
    if len(x) > 50_000:
        ix = rng.choice(len(x), 50_000, replace=False)
        x, y = x[ix], y[ix]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=3, alpha=.25, rasterized=True)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=f'{path.stem} (plotted n={len(x)})')
    if log:
        ax.set_xscale('log'); ax.set_yscale('log')
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.show(); plt.close(fig)


def histogram(series, title, path, clipped=False):
    clean = {k: np.asarray(v)[np.isfinite(v)] for k, v in series.items()}
    pooled = np.concatenate(list(clean.values())) if clean else np.array([])
    if not len(pooled):
        print('No finite values:', title)
        return
    lo, hi = pooled.min(), np.quantile(pooled, .999) if clipped else pooled.max()
    if lo == hi:
        lo, hi = lo - .5, hi + .5
    bins = np.linspace(lo, hi, 101)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, vals in clean.items():
        if ((vals >= lo) & (vals <= hi)).any():
            ax.hist(vals, bins=bins, density=True, histtype='step', label=label)
    ax.set(xlabel=title, ylabel='Density within plotted range', title=title + (' (pooled q99.9 limit)' if clipped else ''))
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.show(); plt.close(fig)
