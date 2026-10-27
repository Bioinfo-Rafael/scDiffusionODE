#!/usr/bin/env python3
"""Remove only known unrequested outputs, compact redundant arrays, rebuild index. No Git edits."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import re
from figure_guide import describe_figure
from requested_outputs import requested_figure, obsolete_data

MOMENT_KEYS = {'coefficient_mean', 'coefficient_std', 'coefficient_covariance', 'coefficient_mean_absolute'}


def compact_keys(relative):
    path = Path(relative)
    if path.name == 'matrix_moments.npz' and 'groups' in path.parts:
        return MOMENT_KEYS
    if path.parent.name == 'basis' and re.fullmatch(r'B[1-8]_graph\.npz', path.name):
        return {'adjacency', 'degree', 'ordinary_eigenvalues', 'ordinary_low_eigenvectors'}
    if path.parent.name == 'sampled_graphs' and re.fullmatch(r'cell_\d+\.npz', path.name):
        return {'cell_id', 'coefficients', 'C_ordinary_eigenvalues'}
    return None


def plan_run(root):
    root = Path(root).resolve()
    if (root / 'analysis').is_dir(): root = root / 'analysis'
    if not (root / 'manifest.json').is_file():
        raise ValueError(f'Refusing cleanup without analysis manifest: {root}')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('status') not in ('complete', 'failed', 'interrupted'):
        raise ValueError(f'解析が完了または停止したrunだけを指定してください: {root}')
    delete, compact, totals = [], [], defaultdict(int)
    for model in ('matsum', 'lora'):
        model_dir = root / model
        for path in sorted(model_dir.rglob('*')):
            if not path.is_file() or path.is_symlink(): continue
            # Never follow a symlinked parent out of this run.
            if not path.resolve().is_relative_to(root): continue
            relative = path.relative_to(model_dir).as_posix()
            size = path.stat().st_size
            suffix = '.csv.gz' if path.name.endswith('.csv.gz') else path.suffix
            totals[suffix] += size
            known_plot = path.suffix == '.png' and describe_figure(path)[0] != '個別の出力図'
            if known_plot and not requested_figure(relative) or path.suffix != '.png' and obsolete_data(relative):
                delete.append({'path': str(path.relative_to(root)), 'bytes': size})
            keys = compact_keys(relative)
            if keys:
                compact.append({'path': str(path.relative_to(root)), 'bytes_before': size, 'keep_keys': sorted(keys)})
    return root, dict(delete=delete, compact=compact, bytes_by_extension=dict(totals),
                      delete_bytes=sum(item['bytes'] for item in delete))


def apply_plan(root, plan):
    import numpy as np
    # Validate all compactable arrays before deleting anything.
    for item in plan['compact']:
        path = root / item['path']
        if not (root / path.relative_to(root).parts[0] / 'basis/shared_matrices.npz').is_file():
            raise ValueError(f'Missing shared matrices required for reconstruction: {path}')
        with np.load(path, allow_pickle=False) as z:
            extras = {'mean_W', 'std_W', 'std_over_mean_W'} if path.name == 'matrix_moments.npz' else (
                {'normalized_eigenvalues', 'normalized_low_eigenvectors'} if path.parent.name == 'basis' else
                {'C_normalized_eigenvalues', 'W_ordinary_eigenvalues', 'W_normalized_eigenvalues'})
            keep = set(item['keep_keys'])
            if not keep <= set(z.files) or not set(z.files) <= keep | extras:
                raise ValueError(f'Unexpected NPZ schema; preserving files: {path}')
    for item in plan['compact']:
        path = root / item['path']
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) == set(item['keep_keys']):
                item['bytes_after'] = path.stat().st_size
                continue
            arrays = {key: z[key] for key in item['keep_keys']}
        temp = path.with_name('.' + path.name + '.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temp.replace(path)
        item['bytes_after'] = path.stat().st_size
    for item in plan['delete']:
        (root / item['path']).unlink()
    plan['applied_at_jst'] = datetime.now(timezone(timedelta(hours=9))).isoformat()
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_%f')
    (root / f'cleanup_{stamp}.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    from report import write_report
    write_report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--apply', action='store_true', help='Delete listed extra artifacts and compact redundant arrays')
    args = parser.parse_args()
    roots = sorted({p.parent for name in ('matsum', 'lora') for p in args.output.rglob(name) if p.is_dir()}) if args.all else [args.output]
    if not roots: parser.error('No analysis results found')
    for candidate in roots:
        root, plan = plan_run(candidate)
        print(root, flush=True)
        for extension, size in sorted(plan['bytes_by_extension'].items()):
            print(f'  {extension}: {size / 1024**2:.1f} MiB')
        print(f"  delete: {len(plan['delete'])} files, {plan['delete_bytes'] / 1024**2:.1f} MiB; compact: {len(plan['compact'])} NPZ", flush=True)
        if args.apply:
            apply_plan(root, plan)
            saved = sum(item['bytes_before'] - item['bytes_after'] for item in plan['compact']) + plan['delete_bytes']
            print(f'  applied; saved {saved / 1024**2:.1f} MiB; index.html rebuilt', flush=True)
        else:
            for item in plan['delete']: print('  DELETE ' + item['path'])
            print('  Preview only. Use --apply to execute.')


if __name__ == '__main__': main()
