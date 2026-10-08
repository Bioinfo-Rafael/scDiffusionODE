#!/usr/bin/env python3
"""Rebuild only UMAP coordinates and all 25 PNGs from existing samples."""
import argparse
import subprocess
from common import CONDITIONS, REPO, STEPS, SUITE, inside, os, sys


def latest_sampled_campaign():
    candidates = []
    for path in (SUITE / 'runs').iterdir():
        if not path.is_dir():
            continue
        required = [path / name / 'samples' / f'step_{step:04d}.npz'
                    for name in CONDITIONS for step in STEPS]
        if all(file.is_file() for file in required):
            candidates.append((max(file.stat().st_mtime_ns for file in required), path))
    if not candidates:
        raise FileNotFoundError('No campaign with all four conditions and all six saved states')
    return max(candidates, key=lambda item: (item[0], str(item[1])))[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', help='Default: most recently sampled complete four-condition campaign')
    args = parser.parse_args(argv)
    campaign = inside(args.campaign) if args.campaign else latest_sampled_campaign()
    print(f'UMAP-only campaign: {campaign}', flush=True)
    for script, extra in [('plot_umap.py', ['--refit']), ('plot_comparison.py', [])]:
        command = [sys.executable, '-B', str(SUITE / 'scripts' / script),
                   '--campaign', str(campaign), *extra]
        subprocess.run(command, cwd=REPO,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), check=True)
    print(f'Completed: {campaign / "umap"} (24 panels + 1 comparison PNG)', flush=True)


if __name__ == '__main__':
    main()
