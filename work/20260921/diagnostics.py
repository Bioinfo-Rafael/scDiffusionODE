"""Distribution, real-defined occupancy, mixing, and the eleven overview figures."""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from geometry import ratio


def directions(dimension, count, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(dimension, count))
    return v / np.linalg.norm(v, axis=0, keepdims=True)


def exposure(q, p, projections):
    if q.shape != p.shape or len(q) < 2:
        raise ValueError('SWD requires equal-sized samples with at least two points')
    q, p = np.asarray(q, dtype=float), np.asarray(p, dtype=float)
    delta = np.sort(q @ projections, axis=0) - np.sort(p @ projections, axis=0)
    qtrace, ptrace = q.var(axis=0, ddof=1).sum(), p.var(axis=0, ddof=1).sum()
    return dict(swd=float(np.abs(delta).mean()),
                centroid_distance=float(np.linalg.norm(q.mean(axis=0) - p.mean(axis=0))),
                covariance_trace_q=float(qtrace), covariance_trace_p=float(ptrace),
                covariance_trace_ratio=float(ratio(ptrace, qtrace)))


def mode_occupancy(real, generated, labels, timestep, k=15):
    if not 1 <= k <= len(real):
        raise ValueError('Invalid mode-transfer k')
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='brute', n_jobs=1).fit(real)
    ids = neighbors.kneighbors(generated, return_distance=False)
    rows, summaries, assignments = [], [], []
    for column, values in labels.items():
        # Missing annotation remains an explicit real-defined category.
        categories, codes = np.unique(values, return_inverse=True)
        votes = np.array([np.bincount(codes[n], minlength=len(categories)) for n in ids])
        assigned = votes.argmax(axis=1)  # deterministic lexicographic tie break
        real_p = np.bincount(codes, minlength=len(categories)) / len(codes)
        gen_p = np.bincount(assigned, minlength=len(categories)) / len(assigned)
        mid = (real_p + gen_p) / 2
        def kl(a):
            nz = a > 0
            return float(np.sum(a[nz] * np.log(a[nz] / mid[nz])))
        for label, rp, gp in zip(categories, real_p, gen_p):
            rows.append(dict(annotation=column, label=label, real_fraction=rp,
                             generated_fraction=gp, difference=gp-rp, timestep=timestep))
        summaries.append(dict(annotation=column, timestep=timestep,
                              total_variation_distance=float(np.abs(real_p-gen_p).sum()/2),
                              jensen_shannon_divergence=(kl(real_p)+kl(gen_p))/2))
        assignments.extend(dict(annotation=column, timestep=timestep, trajectory_id=i,
                                label=categories[a], vote_fraction=float(votes[i, a] / k))
                           for i, a in enumerate(assigned))
    return rows, summaries, assignments


def mixing(real, generated, k, seed, near_zero=0.05):
    rng = np.random.default_rng(seed)
    n = min(len(real), len(generated))
    if not 1 <= k < 2*n:
        raise ValueError('Mixing k must be less than the balanced joint sample size')
    ri = rng.choice(len(real), n, replace=False)
    gi = np.sort(rng.choice(len(generated), n, replace=False))
    joint = np.concatenate([real[ri], generated[gi]])
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='brute', n_jobs=1).fit(joint)
    indices = nn.kneighbors(joint[n:], return_distance=False)
    # Remove by identity, not position: duplicate coordinates may create ties.
    neighbors = np.array([row[row != n+i][:k] for i, row in enumerate(indices)])
    fractions = (neighbors < n).mean(axis=1)
    return dict(mean_real_neighbor_fraction=float(fractions.mean()),
                median_real_neighbor_fraction=float(np.median(fractions)),
                fraction_near_zero=float((fractions <= near_zero).mean()),
                near_zero_threshold=near_zero, balanced_count=n,
                random_mixing_baseline=n/(2*n-1)), gi, fractions


def make_figures(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = output / 'figures'
    folder.mkdir()
    g = pd.read_csv(output / 'tangent_normal.csv')
    g = g[g.regime == 'main_low_noise']
    dims = sorted(g.d.unique())
    # Each dimension gets its own axis; high-noise rows never enter main figures.
    def panels():
        fig, axes = plt.subplots(1, max(1, len(dims)), figsize=(5*max(1, len(dims)), 4), squeeze=False)
        for ax, d in zip(axes[0], dims):
            ax.set_title(f'd={d}')
        return fig, axes[0]
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / name, dpi=160, bbox_inches='tight')
        plt.close(fig)
    names = [('score', '01_score_normal_tangent_vs_t.png'),
             ('model_drift', '02_model_drift_normal_tangent_vs_t.png'),
             ('noise', '03_noise_normal_tangent_vs_t.png')]
    for vector, name in names:
        fig, axes = panels()
        for ax, d in zip(axes, dims):
            mean = g[g.d == d].groupby('t').mean(numeric_only=True)
            for component in ('tangent', 'normal'):
                ax.plot(mean.index, mean[f'{vector}_{component}_norm'], marker='.', label=component)
            ax.set(xlabel='t (sampling moves right to left)', ylabel=f'Mean {vector} norm')
            ax.set_xscale('symlog', linthresh=1)
            ax.invert_xaxis()
            ax.legend()
        save(fig, name)
    fig, axes = panels()
    for ax, d in zip(axes, dims):
        mean = g[g.d == d].groupby('t').mean(numeric_only=True)
        for v in ('score', 'model_drift', 'noise', 'total'):
            ax.plot(mean.index, mean[v+'_normal_fraction_sq'], marker='.', label=v)
        ax.set(xlabel='t', ylabel='Mean squared normal fraction', ylim=(0, 1.02))
        ax.set_xscale('symlog', linthresh=1)
        ax.invert_xaxis()
        ax.legend(fontsize=8)
    save(fig, '04_normal_fraction_vs_t.png')
    for key, name in [('model_drift_normal_norm', '05_model_drift_vs_manifold_distance.png'),
                      ('model_drift_tangent_norm', '06_tangent_drift_vs_manifold_distance.png'),
                      ('cos_normal', '07_cosine_to_manifold_normal_vs_distance.png')]:
        fig, axes = panels()
        for ax, d in zip(axes, dims):
            # Per-t curves avoid conflating step-size/noise effects with distance.
            for t, subset in g[g.d == d].groupby('t'):
                subset = subset[['manifold_distance', key]].dropna()
                bins = pd.qcut(subset.manifold_distance, q=min(12, len(subset)), duplicates='drop')
                avg = subset.groupby(bins, observed=True).mean(numeric_only=True)
                ax.plot(avg.manifold_distance, avg[key], marker='.', label=f't={t}')
            ax.set(xlabel='Local affine plane normal residual', ylabel=key)
            ax.legend(fontsize=7)
        save(fig, name)
    e = pd.read_csv(output / 'exposure.csv').sort_values('t')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(e.t, e.swd, marker='o', label='forward vs reverse')
    ax.plot(e.t, e.swd_forward_baseline, marker='.', label='independent forward vs forward')
    ax.set(xlabel='t (sampling moves right to left)', ylabel='Sliced 1-Wasserstein distance')
    ax.invert_xaxis()
    ax.legend()
    save(fig, '08_exposure_swd_vs_t.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ('centroid_distance', 'covariance_trace_ratio')):
        ax.plot(e.t, e[col], marker='o')
        ax.set(xlabel='t', ylabel=col)
        ax.invert_xaxis()
    axes[1].axhline(1, color='gray', linestyle='--')
    save(fig, '09_exposure_centroid_covariance_vs_t.png')
    o = pd.read_csv(output / 'mode_occupancy.csv')
    s = pd.read_csv(output / 'mode_summary.csv')
    annotations = list(o.annotation.unique())
    fig, axes = plt.subplots(max(1, len(annotations)), 2, figsize=(14, 4*max(1, len(annotations))), squeeze=False)
    if not annotations:
        for ax in axes.flat:
            ax.text(.1, .5, 'No Superclass/celltype annotation available', transform=ax.transAxes)
    for row, annotation in enumerate(annotations):
        table = o[o.annotation == annotation].pivot(index='label', columns='timestep', values='difference')
        table = table.reindex(sorted(table.columns, reverse=True), axis=1)
        ax = axes[row, 0]
        limit = max(1e-6, np.abs(table.values).max())
        im = ax.imshow(table.values, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(table.columns)), ['final' if t == -1 else str(t) for t in table.columns], rotation=60)
        ax.set_yticks(range(len(table)), table.index, fontsize=max(4, min(9, 180/max(1,len(table)))))
        ax.set_title(f'{annotation}: generated fraction - real fraction')
        fig.colorbar(im, ax=ax)
        sub = s[s.annotation == annotation].sort_values('timestep', ascending=False)
        ax = axes[row, 1]
        for col in ('total_variation_distance', 'jensen_shannon_divergence'):
            ax.plot(range(len(sub)), sub[col], marker='.', label=col)
        ax.set_xticks(range(len(sub)), ['final' if t == -1 else str(t) for t in sub.timestep], rotation=60)
        ax.set_title(annotation)
        ax.legend(fontsize=8)
    save(fig, '10_mode_occupancy_over_time.png')
    m = pd.read_csv(output / 'mixing.csv').sort_values('timestep', ascending=False)
    points = pd.read_csv(output / 'mixing_points.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for col in ('mean_real_neighbor_fraction', 'median_real_neighbor_fraction', 'fraction_near_zero'):
        axes[0].plot(range(len(m)), m[col], marker='.', label=col)
    axes[0].plot(range(len(m)), m.random_mixing_baseline, linestyle='--', label='random mixing baseline')
    labels = ['final' if t == -1 else str(t) for t in m.timestep]
    axes[0].set_xticks(range(len(m)), labels, rotation=60)
    axes[0].set(ylim=(0, 1.02), xlabel='pred_xstart timestep / final sample')
    axes[0].legend(fontsize=7)
    axes[1].boxplot([points[points.timestep == t].real_neighbor_fraction for t in m.timestep],
                    showfliers=False)
    axes[1].set_xticks(range(1, len(labels)+1), labels)
    axes[1].tick_params(axis='x', rotation=60)
    axes[1].set(ylabel='Per-generated-cell real-neighbor fraction', ylim=(0, 1.02))
    save(fig, '11_real_generated_mixing_over_time.png')
