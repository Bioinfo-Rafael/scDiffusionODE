"""Explicit diffusion-time and optimizer-step plots; no log clipping of zero losses."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .branches import LABELS, PAIRS
from .losses import RAW, WEIGHTED, NAMES

METRICS = {'cosine_mean': 'Per-cell cosine (mean)',
           'pearson_mean': 'Per-cell Pearson (mean)',
           'pearson_global': 'Global Pearson (all cells x genes)',
           'mse_mean': 'Per-cell gene MSE (mean)',
           'norm_ratio_mean': 'Per-cell L2 norm ratio (mean)'}
COLORS = ('#0072B2', '#D55E00', '#009E73')


def finish(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def positive(values, log):
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, values, np.nan) if log else values


def scale_axis(ax, log):
    if log:
        # An all-zero series must not cause a log-scale crash or become epsilon.
        if not any(np.any(np.asarray(line.get_ydata()) > 0) for line in ax.lines):
            if len(ax.get_shared_y_axes().get_siblings(ax)) == 1:
                ax.set_ylim(1e-12, 1.)
            ax.text(.5, .5, 'No positive values', transform=ax.transAxes, ha='center')
        ax.set_yscale('log')
    ax.grid(alpha=.2)


def plot_losses(history, mean, std, fraction, output, title, window):
    output = Path(output)
    for weighting, columns in (('raw', RAW), ('weighted', WEIGHTED)):
        for mode, frame in (('unsmoothed', history), ('rolling', mean)):
            for log in (False, True):
                fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
                for ax, column, label, color in zip(axes.flat, (*columns, 'total_loss'),
                                                   (*NAMES, 'Total (weighted)'), (*COLORS, '#333333')):
                    values = frame[column].to_numpy()
                    ax.plot(frame.step, positive(values, log), color=color, lw=1.1,
                            label=f'{label}: {mode}')
                    if mode == 'rolling':
                        sd = std[column].to_numpy()
                        ax.fill_between(frame.step.to_numpy(), positive(values-sd, log),
                                        positive(values+sd, log), color=color, alpha=.18,
                                        label='Mean +/- population SD')
                    if log:
                        count = int((values <= 0).sum())
                        ax.text(.02, .02, f'{count} nonpositive points omitted', transform=ax.transAxes, fontsize=8)
                    ax.set(title=label, xlabel='Optimizer update (training step)', ylabel='Loss')
                    scale_axis(ax, log)
                    ax.legend(fontsize=8)
                axis = 'log' if log else 'linear'
                fig.suptitle(f'{title}\n{weighting} components | {mode} | {axis}'
                             + (f' | trailing {window} optimizer steps' if mode == 'rolling' else '')
                             + ('\nRaw components are not additive; total uses weighted terms.' if weighting == 'raw' else ''))
                finish(fig, output/f'loss_{weighting}_{mode}_{axis}.png')
    for mode, suffix in (('unsmoothed', ''), ('rolling', '_rolling')):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
        columns = [k+suffix for k in ('reconstruction', 'grn', 'consistency')]
        values = fraction[columns].to_numpy().T
        axes[0].stackplot(fraction.step.to_numpy(), values, labels=NAMES, colors=COLORS, alpha=.8)
        for label, color, value in zip(NAMES, COLORS, values):
            axes[1].plot(fraction.step, value, label=label, color=color)
        for ax in axes:
            ax.set(xlabel='Optimizer update (training step)', ylabel='Fraction of weighted total', ylim=(0, 1))
            ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.suptitle(f'{title}\nWeighted loss contributions | {mode}'
                     + (f' | ratio of trailing {window}-step mean losses' if suffix else ''))
        finish(fig, output/f'loss_fraction_{mode}.png')


def pair_label(pair):
    a, b = PAIRS[pair]
    return f'{LABELS[a]} / {LABELS[b]}'


def plot_branches(pairs, norms, output, title):
    output = Path(output)
    for metric, ylabel in METRICS.items():
        for log in ((False, True) if metric in ('mse_mean', 'norm_ratio_mean') else (False,)):
            fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
            for ax, pair in zip(axes, PAIRS):
                data = pairs[pairs.pair == pair]
                for (step, weights), group in data.groupby(['checkpoint_step', 'weights'], sort=False):
                    group = group.sort_values('diffusion_timestep')
                    line, = ax.plot(group.diffusion_timestep, positive(group[metric], log), marker='.',
                                    label=f'{weights}, update {step}')
                    if metric.endswith('_mean'):
                        prefix = metric[:-5]
                        ax.fill_between(group.diffusion_timestep.to_numpy(), positive(group[prefix+'_q25'], log),
                                        positive(group[prefix+'_q75'], log), color=line.get_color(), alpha=.1)
                ax.set(title=pair_label(pair), xlabel='Diffusion timestep t', ylabel=ylabel)
                if metric in ('cosine_mean', 'pearson_mean', 'pearson_global'):
                    ax.set_ylim(-1.05, 1.05); ax.axhline(0, color='gray', lw=.6)
                if metric == 'norm_ratio_mean':
                    ax.axhline(1, color='gray', ls='--', lw=.8)
                scale_axis(ax, log); ax.legend(fontsize=8)
            axis = 'log' if log else 'linear'
            fig.suptitle(f'{title}\n{ylabel} | {axis}; bands = cell Q25-Q75; target = x0'
                         + ('; nonpositive values omitted' if log else ''))
            finish(fig, output/f'branches_{metric}_{axis}.png')
    for log in (False, True):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
        for ax, branch in zip(axes, LABELS):
            for (step, weights), group in norms[norms.branch == branch].groupby(['checkpoint_step', 'weights'], sort=False):
                group = group.sort_values('diffusion_timestep')
                line, = ax.plot(group.diffusion_timestep, positive(group['mean'], log), marker='.', label=f'{weights}, update {step}')
                ax.fill_between(group.diffusion_timestep.to_numpy(), positive(group.q25, log), positive(group.q75, log),
                                color=line.get_color(), alpha=.1)
            ax.set(title=LABELS[branch], xlabel='Diffusion timestep t', ylabel='Per-cell L2 norm (mean)')
            scale_axis(ax, log); ax.legend(fontsize=8)
        axis = 'log' if log else 'linear'
        fig.suptitle(f'{title}\nL2 norms | {axis}; bands = cell Q25-Q75')
        finish(fig, output/f'branches_norm_{axis}.png')
    if pairs.checkpoint_step.nunique() > 1:
        chosen = [t for t in (0, 49, 499, 999) if t in set(pairs.diffusion_timestep)]
        if not chosen:
            chosen = [int(pairs.diffusion_timestep.min()), int(pairs.diffusion_timestep.max())]
        for metric, label in METRICS.items():
            fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
            for ax, pair in zip(axes, PAIRS):
                data = pairs[(pairs.pair == pair) & pairs.diffusion_timestep.isin(chosen)]
                for (t, weights), group in data.groupby(['diffusion_timestep', 'weights']):
                    group = group.sort_values('checkpoint_step')
                    ax.plot(group.checkpoint_step, group[metric], marker='o', label=f'{weights}, t={t}')
                ax.set(title=pair_label(pair), xlabel='Checkpoint optimizer update', ylabel=label)
                ax.grid(alpha=.2); ax.legend(fontsize=8)
            fig.suptitle(title+'\nCheckpoint progression on fixed cells/noise; target = x0')
            finish(fig, output/f'checkpoint_{metric}.png')


def plot_comparison(pairs, norms, output):
    """Each panel is one campaign x condition; latest selected step only."""
    output = Path(output)
    campaigns = list(pairs.campaign.unique())
    conditions = sorted(pairs.condition.unique())
    for weights in pairs.weights.unique():
        for metric, label in {**METRICS, 'norm': 'Per-cell L2 norm (mean)'}.items():
            log = metric in ('mse_mean', 'norm_ratio_mean', 'norm')
            fig, axes = plt.subplots(len(campaigns), len(conditions), squeeze=False,
                                     figsize=(5*len(conditions), 4*len(campaigns)), layout='constrained', sharey=True)
            for i, campaign in enumerate(campaigns):
                for j, condition in enumerate(conditions):
                    ax = axes[i, j]
                    frame = norms if metric == 'norm' else pairs
                    data = frame[(frame.campaign == campaign) & (frame.condition == condition) & (frame.weights == weights)]
                    if data.empty:
                        ax.text(.5, .5, 'Unavailable', transform=ax.transAxes, ha='center'); continue
                    step = data.checkpoint_step.max()
                    data = data[data.checkpoint_step == step]
                    for (name, group), color in zip(data.groupby('branch' if metric == 'norm' else 'pair', sort=False), COLORS):
                        group = group.sort_values('diffusion_timestep')
                        ax.plot(group.diffusion_timestep, positive(group['mean' if metric == 'norm' else metric], log),
                                color=color, label=LABELS[name] if metric == 'norm' else pair_label(name))
                    ax.set(title=f'{campaign}\n{condition}\n{weights}, update {step}', xlabel='Diffusion timestep t', ylabel=label)
                    scale_axis(ax, log); ax.legend(fontsize=7)
                    if metric in ('cosine_mean', 'pearson_mean', 'pearson_global'):
                        ax.set_ylim(-1.05, 1.05)
            fig.suptitle(f'{label} | target = x0 | '+('log y; nonpositive omitted' if log else 'linear y'))
            finish(fig, output/f'comparison_{weights}_{metric}.png')


def plot_loss_comparison(entries, output):
    campaigns = list(dict.fromkeys(entry[0] for entry in entries))
    conditions = sorted({entry[1] for entry in entries})
    for mode, frame_index in (('unsmoothed', 2), ('rolling', 3)):
        for log in (False, True):
            fig, axes = plt.subplots(len(campaigns), len(conditions), squeeze=False,
                                     figsize=(5*len(conditions), 4*len(campaigns)), layout='constrained', sharey=True)
            for entry in entries:
                campaign, condition = entry[:2]
                frame = entry[frame_index]
                ax = axes[campaigns.index(campaign), conditions.index(condition)]
                for column, name, color in zip((*WEIGHTED, 'total_loss'), (*NAMES, 'Total'), (*COLORS, '#333333')):
                    ax.plot(frame.step, positive(frame[column], log), color=color, label=name, lw=1)
                ax.set(title=f'{campaign}\n{condition}', xlabel='Optimizer update', ylabel='Weighted loss')
                scale_axis(ax, log); ax.legend(fontsize=7)
            axis = 'log' if log else 'linear'
            fig.suptitle(f'Weighted losses | {mode} | {axis}; common y scale'
                         + ('; nonpositive omitted' if log else ''))
            finish(fig, Path(output)/f'comparison_loss_{mode}_{axis}.png')
        fig, axes = plt.subplots(len(campaigns), len(conditions), squeeze=False,
                                 figsize=(5*len(conditions), 4*len(campaigns)), layout='constrained', sharey=True)
        for campaign, condition, history, mean, fraction in entries:
            ax = axes[campaigns.index(campaign), conditions.index(condition)]
            suffix = '_rolling' if mode == 'rolling' else ''
            values = fraction[[k+suffix for k in ('reconstruction', 'grn', 'consistency')]].to_numpy().T
            ax.stackplot(fraction.step.to_numpy(), values, labels=NAMES, colors=COLORS, alpha=.8)
            ax.set(title=f'{campaign}\n{condition}', xlabel='Optimizer update', ylabel='Fraction of weighted total', ylim=(0, 1))
            ax.legend(fontsize=7)
        fig.suptitle(f'Weighted loss fractions | {mode}')
        finish(fig, Path(output)/f'comparison_loss_fraction_{mode}.png')
