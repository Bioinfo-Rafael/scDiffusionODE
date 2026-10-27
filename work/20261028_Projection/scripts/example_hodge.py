"""Two stages: fit/save for external visualization; reload with a user-chosen --c."""
import argparse
import json
from pathlib import Path
import sys
import anndata as ad
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common.velocity_adapter import load_d20
from Hodge.projector import HodgeProjector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(HERE/'config.json'))
    parser.add_argument('--output', default=str(HERE/'outputs/hodge'))
    parser.add_argument('--resume', help='Saved artifact directory from the fit stage')
    parser.add_argument('--c', type=float, help='Choose after inspecting annotated.h5ad externally')
    parser.add_argument('--max-cells', type=int, help='Explicit first-N subset for resource-limited smoke runs')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--k', type=int, default=12)
    parser.add_argument('--rdim', type=int, default=4)
    args = parser.parse_args()
    if args.c is not None and not args.resume:
        parser.error('--c requires --resume: inspect saved scalar first')
    config = json.loads(Path(args.config).read_text())
    velocity, adata = load_d20(config, args.device)
    if args.resume:
        adata = ad.read_h5ad(Path(args.resume)/'annotated.h5ad')
        projector = HodgeProjector.load_saved(args.resume, velocity, adata)
    else:
        if args.max_cells:
            adata = adata[:args.max_cells].copy()
        projector = HodgeProjector(velocity, adata, interpolation=config['interpolation'])
        projector.fit_scalar(k=args.k, rdim=args.rdim, seed=config['seed'])
        print('Saved:', projector.save(args.output))
    print(adata.obs[['projection_hodge_raw', 'projection_hodge']].head())
    print('Selectable c range:', projector.scalar.valid_range)
    if args.c is None:
        print('Visualize annotated.h5ad outside this script, then rerun with --resume DIR --c VALUE.')
        return
    projected_field = projector.get_projected_field(c=args.c)
    x = np.asarray(adata.X[:1].toarray() if hasattr(adata.X, 'toarray') else adata.X[:1])[0]
    y = projected_field.surface_points(x)
    v, diagnostics = projected_field.evaluate(x)  # Same callable is valid for any supported (D,) or (N,D).
    np.testing.assert_allclose(projected_field(x), v)
    print('Surface point:', y)
    print('Projected velocity:', v)
    print('Diagnostics:', diagnostics)


if __name__ == '__main__':
    main()
