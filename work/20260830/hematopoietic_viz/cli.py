"""Shared CLI arguments for single-run and batch post-hoc entry points."""

from __future__ import annotations

from .runner import HematopoieticVizOptions, parse_timesteps


def add_common_arguments(parser):
    parser.add_argument("--timesteps", default="0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--neighbor-pcs", type=int, default=40)
    parser.add_argument("--superclass-column", default="")
    parser.add_argument("--superclass", action="append", default=[])
    parser.add_argument("--celltype-column", default="celltype")
    parser.add_argument("--no-h5ad", action="store_true")
    parser.add_argument("--no-paga", action="store_true")
    parser.add_argument("--force", action="store_true")


def options_from_args(args):
    return HematopoieticVizOptions(
        timesteps=parse_timesteps(args.timesteps),
        batch_size=args.batch_size,
        seed=args.seed,
        noise_seed=args.noise_seed,
        device=args.device,
        n_jobs=args.n_jobs,
        pca_components=args.pca_components,
        neighbors=args.neighbors,
        neighbor_pcs=args.neighbor_pcs,
        superclass_column=args.superclass_column,
        superclasses=tuple(args.superclass),
        celltype_column=args.celltype_column,
        sample_path=getattr(args, "sample_path", ""),
        save_h5ad=not args.no_h5ad,
        paga=not args.no_paga,
        force=args.force,
    )


__all__ = ["add_common_arguments", "options_from_args"]
