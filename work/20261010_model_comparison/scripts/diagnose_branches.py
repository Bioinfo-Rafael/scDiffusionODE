"""Plot ODE / CellUNet / x0 metrics and all training loss components (no training)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
PACKAGE = 'work.20261010_model_comparison'
DEFAULT_ROOTS = [HERE/'runs/retry_20261012_052933', HERE/'runs/d_ode_sgd_30000']
DEFAULT_TIMES = '0,1,10,25,49,99,199,299,399,499,599,699,799,899,999'


def integers(value):
    """Comma list or Python-style start:stop[:step] (stop excluded)."""
    if ':' in value:
        parts = [int(p) for p in value.split(':')]
        if len(parts) not in (2, 3):
            raise ValueError('Use start:stop[:step] or a comma-separated list')
        values = list(range(*parts))
    else:
        values = [int(p) for p in value.split(',')]
    if not values:
        raise ValueError('Empty integer selection')
    return sorted(set(values))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-roots', type=Path, nargs='+', default=DEFAULT_ROOTS)
    p.add_argument('--conditions', nargs='+', default=[f.stem for f in sorted((HERE/'configs/ode_sgd').glob('D*.json'))])
    p.add_argument('--data-dir', type=Path, default=HERE/'runs/data')
    p.add_argument('--output-dir', type=Path, help='New directory below work/20261010_model_comparison (never overwritten)')
    p.add_argument('--checkpoint-steps', default='latest', help='latest or comma list, e.g. 5000,10000,20000,30000')
    p.add_argument('--weights', nargs='+', choices=['raw', 'ema'], default=['raw', 'ema'])
    p.add_argument('--timesteps', default=DEFAULT_TIMES, help='Comma list or start:stop:step; all: 0:1000')
    p.add_argument('--cells', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=1234, help='Fixed cell selection and diagnostic noise')
    p.add_argument('--rolling-window', type=int, default=100, help='Trailing optimizer steps, not log-row count')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--loss-only', action='store_true', help='Only JSONL logs; no checkpoints or training data required')
    p.add_argument('--check-only', action='store_true', help='Check input file presence/configuration without writing plots')
    args = p.parse_args(argv)
    if min(args.cells, args.batch_size, args.rolling_window) < 1:
        p.error('cells, batch-size and rolling-window must be positive')
    times = integers(args.timesteps)
    if times[0] < 0 or times[-1] >= 1000:
        p.error('diffusion timesteps must be within 0..999')
    steps = None if args.checkpoint_steps == 'latest' else integers(args.checkpoint_steps)
    if steps is not None and steps[0] < 1:
        p.error('checkpoint steps must be positive optimizer update counts')
    if len(set(args.weights)) != len(args.weights) or len(set(args.conditions)) != len(args.conditions):
        p.error('duplicate weights or conditions')
    if len({path.name for path in args.run_roots}) != len(args.run_roots):
        p.error('run-roots must have distinct directory names')
    C = importlib.import_module(PACKAGE+'.common')
    jobs = []
    for root in args.run_roots:
        root = root.resolve()
        for name in args.conditions:
            if Path(name).name != name:
                p.error('conditions must be directory names, not paths')
            run = root/name
            config = C.read_json(run/'effective_config.json')
            C.validate(config)
            if config['condition'] != name or config['family'] != 'consistency' or config['prediction'] != 'START_X':
                raise ValueError(f'{run}: expected a START_X consistency condition')
            logs = [path for path in (run/'logs').glob('loss_*.jsonl') if '.rolled_back.' not in path.name]
            if not logs:
                raise FileNotFoundError(f'{run}: no loss_*.jsonl logs')
            checkpoints = []
            if not args.loss_only:
                K = importlib.import_module(PACKAGE+'.adapters.checkpoint')
                checkpoints = [K.resolve(run)] if steps is None else [run/'checkpoints'/f'model{step:06d}.pt' for step in steps]
                for checkpoint in checkpoints:
                    if not checkpoint.is_file():
                        raise FileNotFoundError(checkpoint)
            jobs.append((root.name, run, config, checkpoints))
    if not args.loss_only and not (args.data_dir/'manifest.json').is_file():
        raise FileNotFoundError(args.data_dir/'manifest.json')
    if args.check_only:
        print(json.dumps(dict(status='input_files_present', jobs=len(jobs),
                              checkpoints=sum(len(j[3]) for j in jobs),
                              note='Numerical contents and provenance are validated during execution'), indent=2))
        return

    import numpy as np
    import pandas as pd
    L = importlib.import_module(PACKAGE+'.analysis.losses')
    P = importlib.import_module(PACKAGE+'.analysis.plotting')
    output = C.confined(args.output_dir or HERE/'runs'/('branch_diagnostics_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', options=vars(args), jobs=[],
                  definitions=dict(target='clean x0; both branches receive identical q_sample(x0,t)',
                      correlation='Pearson across genes within each cell; global flattened Pearson also saved',
                      norm='per-cell L2 across genes', norm_ratio='mean of cellwise numerator/denominator ratios',
                      undefined='NaN for zero-norm/constant-vector angle or zero-denominator ratio; counts saved',
                      rolling='trailing optimizer-step window; SD ddof=0; fraction = ratio of rolling weighted losses',
                      loss='all live loss JSONL steps, excluding rolled_back segments; sampled training t',
                      metrics='fixed cell subset and float64 noise per t; eval/no_grad; no optimizer',
                      log='nonpositive values omitted only from rendering; CSV keeps zeros'),
                  source_sha256={str(path.relative_to(ROOT)): C.sha256(path)
                                 for path in [Path(__file__), *(HERE/'analysis').glob('*.py'),
                                              ROOT/'work/20260830/analysis/metrics.py']})
    C.write_json(output/'manifest.json', report)
    try:
        all_pairs, all_norms, loss_summaries, loss_entries = [], [], [], []
        data = meta = selected = None
        if not args.loss_only:
            import torch
            D = importlib.import_module(PACKAGE+'.adapters.data')
            B = importlib.import_module(PACKAGE+'.analysis.branches')
            dev = C.device(args.device)
        for campaign, run, config, checkpoints in jobs:
            title = f'{campaign} | {config["condition"]} | ODE {config.get("ode_optimizer", "AdamW")} | lambda={config["consistency_lambda"]}'
            dest = output/campaign/config['condition']
            dest.mkdir(parents=True)
            print(f'Loss plots: {title}', flush=True)
            history, mean, std, fraction = L.load(run, args.rolling_window)
            for frame, name in ((history, 'loss_history'), (mean, 'loss_rolling_mean'),
                                (std, 'loss_rolling_std'), (fraction, 'loss_fraction')):
                frame.to_csv(dest/(name+'.csv'), index=False)
            P.plot_losses(history, mean, std, fraction, dest/'figures', title, args.rolling_window)
            loss_entries.append((campaign, config['condition'], history, mean, fraction))
            late = fraction[history.step >= history.step.max()*.75]
            loss_summaries.append(dict(campaign=campaign, condition=config['condition'],
                consistency_lambda=config['consistency_lambda'], ode_optimizer=config.get('ode_optimizer', 'AdamW'),
                first_step=int(history.step.min()), last_step=int(history.step.max()), rows=len(history),
                **{name+'_late_mean_fraction': float(late[name].mean()) for name in ('reconstruction', 'grn', 'consistency')}))
            job_report = dict(campaign=campaign, run=str(run), config=config,
                              logs={str(path): C.sha256(path) for path in sorted((run/'logs').glob('loss_*.jsonl')) if '.rolled_back.' not in path.name},
                              checkpoints=[], loss_rows=len(history),
                              missing_optimizer_steps=int(history.step.max()-history.step.min()+1-len(history)))
            pair_frames, norm_frames = [], []
            if checkpoints:
                if data is None:
                    data, meta = D.read_data(config, args.data_dir)
                    selected = np.sort(np.random.default_rng(args.seed).choice(data.n_obs, min(args.cells, data.n_obs), replace=False))
                    x0 = torch.from_numpy(D.dense_rows(data.X[selected]))
                    pd.DataFrame(dict(row_index=selected, cell_id=np.asarray(meta['cell_ids'])[selected])).to_csv(output/'selected_cells.csv', index=False)
                    report['training_sha256'] = meta['training_sha256']
                    report['gene_ids_sha256'] = C.digest_ids(meta['gene_ids'])
                    report['analyzed_cells'] = len(selected)
                if data.shape != (config['expected_cells'], config['n_hvg']):
                    raise ValueError('Campaign data dimensions differ')
                diffusion = C.diffusion(config)
                for checkpoint in checkpoints:
                    for weights in args.weights:
                        print(f'Branches: {title} | {checkpoint.name} | {weights}', flush=True)
                        model, bundle, path = K.load(config, run, meta, value=str(checkpoint), weights=weights, target=dev)
                        metadata = dict(campaign=campaign, condition=config['condition'],
                            consistency_lambda=config['consistency_lambda'], ode_optimizer=config.get('ode_optimizer', 'AdamW'),
                            checkpoint_step=bundle['step'], checkpoint=str(path), weights=weights)
                        pair_frame, norm_frame = B.evaluate(model, diffusion, x0, times,
                            batch_size=args.batch_size, seed=args.seed, device=dev, metadata=metadata)
                        pair_frames.append(pair_frame); norm_frames.append(norm_frame)
                        job_report['checkpoints'].append(dict(path=str(path), sha256=C.sha256(path), step=bundle['step'], weights=weights))
                        del model, bundle
                        if dev.type == 'cuda':
                            torch.cuda.empty_cache()
                pairs, norms = pd.concat(pair_frames, ignore_index=True), pd.concat(norm_frames, ignore_index=True)
                pairs.to_csv(dest/'pair_metrics_by_timestep.csv', index=False)
                norms.to_csv(dest/'norms_by_timestep.csv', index=False)
                P.plot_branches(pairs, norms, dest/'figures', title)
                all_pairs.append(pairs); all_norms.append(norms)
            report['jobs'].append(job_report)
            C.write_json(output/'manifest.json', report)
        pd.DataFrame(loss_summaries).to_csv(output/'loss_summary.csv', index=False)
        P.plot_loss_comparison(loss_entries, output/'comparison')
        if all_pairs:
            pairs, norms = pd.concat(all_pairs, ignore_index=True), pd.concat(all_norms, ignore_index=True)
            pairs.to_csv(output/'all_pair_metrics.csv', index=False)
            norms.to_csv(output/'all_norms.csv', index=False)
            P.plot_comparison(pairs, norms, output/'comparison')
        report['status'] = 'complete'
        C.write_json(output/'manifest.json', report)
        print(f'Complete: {output}', flush=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        C.write_json(output/'manifest.json', report)
        raise


if __name__ == '__main__':
    main()
