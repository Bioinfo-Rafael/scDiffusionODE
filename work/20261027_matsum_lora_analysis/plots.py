"""Direct plotting helpers. All scientific quantities are computed by caller modules."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WARNING = '⚠ WARNING: mean after absolute value; sign discarded'


def histogram_edges(values, bins):
    """Nearly constant ratios need finite-width bins, not repeated float edges."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values): return np.linspace(-.5, .5, bins + 1)
    low, high = float(values.min()), float(values.max())
    scale = max(abs(low), abs(high), 1.)
    if high - low < 1e-10 * scale:
        center = (high + low) / 2
        return np.linspace(center - .01 * scale, center + .01 * scale, bins + 1)
    return np.linspace(low, high, bins + 1)


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def heatmap(matrix, path, title, labels=None, signed=True, limit=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    if signed:
        finite = np.asarray(matrix)[np.isfinite(matrix)]
        limit = limit or max(float(np.max(np.abs(finite))) if len(finite) else 0., 1e-15)
        image = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest', aspect='auto')
    else:
        image = ax.imshow(matrix, cmap='viridis', interpolation='nearest', aspect='auto', vmax=limit)
    if labels is not None:
        rows, cols = labels
        ax.set_yticks(range(len(rows)), labels=rows, fontsize=7)
        ax.set_xticks(range(len(cols)), labels=cols, rotation=90, fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    save(fig, path)


def matrix_panels(matrices, path, title, limit, genes=None):
    fig, axes = plt.subplots(2, 4, figsize=(18, 10), constrained_layout=True)
    for k, ax in enumerate(axes.flat):
        image = ax.imshow(matrices[k], cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
        ax.set_title(f'k={k + 1}')
        ax.set_xlabel('Source gene'); ax.set_ylabel('Target gene')
        if genes is not None:
            ax.set_xticks(range(len(genes)), labels=genes, rotation=90, fontsize=4)
            ax.set_yticks(range(len(genes)), labels=genes, fontsize=4)
    fig.suptitle(title)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.6)
    save(fig, path)


def hist_panels(values, path, title, bins=60, reference=None, edges=None):
    """Each item is a flattened distribution. Reference is plotted with common bins."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for k, ax in enumerate(axes.flat):
        data = np.asarray(values[k]).ravel()
        finite = data[np.isfinite(data)]
        ref = np.array([]) if reference is None else np.asarray(reference[k]).ravel()
        ref = ref[np.isfinite(ref)]
        combined = np.concatenate([finite, ref])
        use_edges = edges[k] if edges is not None else histogram_edges(combined, bins)
        for x, label, color in [(ref, 'all cells', 'gray'), (finite, 'selected group', 'tab:blue')]:
            if len(x):
                counts, _ = np.histogram(x, bins=use_edges)
                density = counts / (len(x) * np.diff(use_edges))
                ax.stairs(density, use_edges, label=label, color=color)
        ax.set_title(f'k={k + 1}; undefined={len(data) - len(finite)}')
        ax.set_ylabel('Density')
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, path)


def histogram(values, path, title, bins=60, reference=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.asarray(values).ravel(); x = x[np.isfinite(x)]
    ref = np.asarray([] if reference is None else reference).ravel(); ref = ref[np.isfinite(ref)]
    edges = histogram_edges(np.concatenate([x, ref]), bins)
    for data, label in [(ref, 'all cells'), (x, 'selected group')]:
        if len(data):
            ax.hist(data, bins=edges, density=True, histtype='step', label=label)
    ax.set_title(title); ax.set_ylabel('Density'); ax.legend()
    save(fig, path)


def stacked(values, labels, path, title):
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * .38), 6))
    positive = np.zeros(len(labels)); negative = positive.copy()
    for k in range(values.shape[1]):
        value = values[:, k]
        bottom = np.where(value >= 0, positive, negative)
        ax.bar(np.arange(len(labels)), value, bottom=bottom, label=f'a{k + 1}')
        positive += np.maximum(value, 0); negative += np.minimum(value, 0)
    ax.axhline(0, color='black', lw=.5)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=7)
    ax.set_title(title); ax.legend(ncol=8, fontsize=8)
    save(fig, path)


def colors_for(labels):
    names = sorted(set(map(str, labels)))
    cmap = plt.get_cmap('turbo')
    mapping = {name: cmap(i / max(1, len(names) - 1)) for i, name in enumerate(names)}
    return mapping, [mapping[str(label)] for label in labels]


def scatter(coords, labels, path, title, axis_names=None, zero_lines=False):
    mapping, colors = colors_for(labels)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=2, alpha=.4, rasterized=True)
    handles = [Line2D([], [], marker='o', linestyle='', color=color, label=name) for name, color in mapping.items()]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6)
    ax.set_title(title)
    if axis_names is not None:
        ax.set_xlabel(axis_names[0]); ax.set_ylabel(axis_names[1])
    if zero_lines:
        ax.axhline(0, color='gray', lw=.5); ax.axvline(0, color='gray', lw=.5)
    save(fig, path)


def pair_scatter(a, labels, path, seed):
    """28 individual pairs and one lower-triangle overview, signed coefficients."""
    from itertools import combinations
    # Random draw order avoids always painting the same celltype last.
    order = np.random.default_rng(seed).permutation(len(a))
    a, labels = a[order], labels[order]
    mapping, colors = colors_for(labels)
    fig, axes = plt.subplots(8, 8, figsize=(20, 20))
    for i in range(8):
        for j in range(8):
            ax = axes[i, j]
            if i < j:
                ax.axis('off'); continue
            if i == j:
                ax.hist(a[:, i], bins=40, color='gray')
            else:
                ax.scatter(a[:, j], a[:, i], c=colors, s=.4, alpha=.3, rasterized=True)
                ax.axhline(0, lw=.3); ax.axvline(0, lw=.3)
            if i == 7: ax.set_xlabel(f'a{j + 1}')
            if j == 0: ax.set_ylabel(f'a{i + 1}')
            ax.tick_params(labelsize=5)
    handles = [Line2D([], [], marker='o', linestyle='', color=color, label=name) for name, color in mapping.items()]
    fig.legend(handles=handles, loc='upper right', fontsize=7)
    save(fig, Path(path) / 'all_28_pairs.png')
    for i, j in combinations(range(8), 2):
        scatter(a[:, [i, j]], labels, Path(path) / f'a{i + 1}_a{j + 1}.png', f'Signed coefficients: a{i + 1} vs a{j + 1}',
                axis_names=(f'a{i + 1}', f'a{j + 1}'), zero_lines=True)


def spectra(values, path, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, row in enumerate(values):
        ax.plot(np.arange(len(row)), row, label=f'k={k + 1}')
    ax.set_xlabel('Eigenvalue index (ascending, including zero)'); ax.set_ylabel('Eigenvalue')
    ax.set_title(title); ax.legend()
    save(fig, path)
