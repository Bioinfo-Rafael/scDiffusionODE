"""Explicit terminal states; fit/save, visualize externally, then resume with chosen c."""
import argparse
import json
from pathlib import Path
import sys
import anndata as ad
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common.velocity_adapter import load_d20
from CellRank.projector import CellRankProjector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(HERE/'config.json'))
    parser.add_argument('--output', default=str(HERE/'outputs/cellrank'))
    parser.add_argument('--resume')
    parser.add_argument('--c', type=float)
    parser.add_argument('--terminal-obs-key')
    parser.add_argument('--terminal-values', nargs='+')
    parser.add_argument('--lineage', help='Required for multiple terminal groups')
    parser.add_argument('--max-cells', type=int)
    parser.add_argument('--n-neighbors', type=int, default=30)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.c is not None and not args.resume:
        parser.error('--c requires --resume: inspect saved scalar first')
    if not args.resume and (not args.terminal_obs_key or not args.terminal_values):
        parser.error('Specify --terminal-obs-key and --terminal-values explicitly')
    config = json.loads(Path(args.config).read_text())
    velocity, adata = load_d20(config, args.device)
    if args.resume:
        adata = ad.read_h5ad(Path(args.resume)/'annotated.h5ad')
        projector = CellRankProjector.load_saved(args.resume, velocity, adata)
    else:
        if args.max_cells:
            adata = adata[:args.max_cells].copy()
        projector = CellRankProjector(velocity, adata, interpolation=config['interpolation'])
        projector.fit_scalar(terminal_obs_key=args.terminal_obs_key,
            terminal_values=args.terminal_values, lineage=args.lineage,
            n_neighbors=args.n_neighbors, seed=config['seed'])
        print('Saved:', projector.save(args.output))
    print(adata.obs[['projection_cellrank_absorption_steps', 'projection_cellrank']].head())
    print('Selectable c range:', projector.scalar.valid_range)
    if args.c is None:
        print('Visualize annotated.h5ad outside this script, then rerun with --resume DIR --c VALUE.')
        return
    projected_field = projector.get_projected_field(c=args.c)
    row = np.flatnonzero(adata.obs['projection_cellrank_eligible'].to_numpy())[0]
    block = adata.X[row:row+1]
    x = np.asarray(block.toarray() if hasattr(block, 'toarray') else block)[0]
    y = projected_field.surface_points(x)
    v, diagnostics = projected_field.evaluate(x)
    np.testing.assert_allclose(projected_field(x), v)
    print('Surface point:', y)
    print('Projected velocity:', v)
    print('Diagnostics:', diagnostics)


if __name__ == '__main__':
    main()
