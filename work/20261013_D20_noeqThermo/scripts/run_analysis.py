#!/usr/bin/env python3
"""Local/remote entry point; never trains, syncs Git, or launches remote jobs."""
from pathlib import Path
import argparse
import json
import sys
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1]))
import importlib
A = importlib.import_module('work.20261013_D20_noeqThermo.src.d20_adapter')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--config', type=Path)
    p.add_argument('--run-dir', type=Path, default=A.DEFAULT_RUN)
    p.add_argument('--data-dir', type=Path, help='model_comparison manifest directory (paths inside manifest must exist)')
    p.add_argument('--checkpoint', help='explicit checkpoint or latest (default)')
    p.add_argument('--weights', choices=['ema', 'raw'], default='ema')
    p.add_argument('--velocity-export', type=Path, help='verified model_comparison h5ad plus JSON sidecar')
    p.add_argument('--device', default='auto')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--check-only', action='store_true', help='strict checkpoint/data/export validation and real-cell inference probe, without thermodynamics')
    args = p.parse_args()
    config = A.load_analysis_config(args.mode, args.config)
    adapter = A.D20Adapter(args.run_dir, data_dir=args.data_dir, checkpoint=args.checkpoint,
        weights=args.weights, device=args.device, velocity_export=args.velocity_export)
    if args.check_only:
        a = adapter.adata[:min(8, adapter.adata.n_obs)]
        _, report = adapter(A.DATA.dense_rows(a.X), a.var_names.tolist(), {'evaluation_cell_ids': a.obs_names.tolist()})
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    output = args.output_dir or HERE / 'outputs' / (args.mode + '_' + args.weights)
    print(json.dumps({'available_labels': adapter.stamp['available_labels'], 'output': str(output)}, ensure_ascii=False), flush=True)
    result = A.run(adapter, config, output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
