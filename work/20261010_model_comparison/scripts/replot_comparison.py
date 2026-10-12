"""Redraw comparison plots with signed and absolute CBDir side by side."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def replot(csv_path):
    frame = pd.read_csv(csv_path)
    out = Path(csv_path).parent
    metrics = ('cbdir', 'abs_cbdir', 'icvcoh', 'cto', 'tsc')
    titles = ('CBDir', '|CBDir|', 'ICVCoh', 'CTO', 'TSC')
    keys = ['inference_protocol', 'profile', 'reference_manifest_sha256']
    for i, ((protocol, profile, _), original) in enumerate(frame.groupby(keys)):
        part = original.copy()
        # Display |mean|, retaining the original population SD. Fold markers
        # show |fold score|; this is not a recomputed mean of absolute folds.
        for column in ('cbdir_mean', 'cbdir_fold0', 'cbdir_fold1', 'cbdir_fold2'):
            part['abs_'+column] = part[column].abs()
        part['abs_cbdir_std'] = part['cbdir_std']
        labels = part.condition + ' / ' + part.field
        y = np.arange(len(part))
        fig, axes = plt.subplots(1, 5, figsize=(21, max(4, len(part)*.25)), layout='constrained')
        for ax, metric, title in zip(axes, metrics, titles):
            ax.errorbar(part[metric+'_mean'], y, xerr=part[metric+'_std'], fmt='o')
            for fold in range(3):
                ax.scatter(part[f'{metric}_fold{fold}'], y, s=12, alpha=.5)
            ax.set_yticks(y, labels if ax is axes[0] else [])
            ax.set_title(title)
        fig.suptitle(f'{protocol} / {profile}; |CBDir|: |mean| and |folds|, original SD')
        fig.savefig(out/f'metrics_{i}.png', dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, max(4, len(part)*.25)), layout='constrained')
        im = ax.imshow(part[[m+'_mean' for m in metrics]].to_numpy(), aspect='auto')
        ax.set_xticks(range(5), titles)
        ax.set_yticks(y, labels)
        fig.colorbar(im, ax=ax, label='metric mean (|CBDir|: absolute mean)')
        fig.savefig(out/f'heatmap_{i}.png', dpi=160)
        plt.close(fig)
        print(out/f'heatmap_{i}.png')
        print(out/f'metrics_{i}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    replot(parser.parse_args().csv)
