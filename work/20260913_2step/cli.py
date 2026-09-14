"""CLI parsing is safe: workflows execute only after an explicit command."""

import argparse
from pathlib import Path
from .common import (
    CONDITIONS,
    LEGACY_CONDITIONS,
    SUITE,
    build_diffusion,
    file_hash,
    run_id,
    seed_all,
)


def parser(command):
    p = argparse.ArgumentParser(
        description=f"Isolated 20260913 two-stage suite: {command}", allow_abbrev=False
    )
    if command == "train":
        p.add_argument(
            "--condition", choices=CONDITIONS + LEGACY_CONDITIONS, required=True
        )
        p.add_argument(
            "--campaign",
            required=True,
            help="one safe directory component below suite/runs",
        )
        p.add_argument("--data", help="Stage 1 only: existing preprocessed h5ad")
        p.add_argument(
            "--edge-tsv", help="Stage 1 only: unchanged source TF-target TSV"
        )
        p.add_argument("--device", default="auto")
        p.add_argument(
            "--resume-checkpoint",
            help="same-campaign intermediate raw checkpoint; continues in a new run directory",
        )
        p.add_argument("--ot-epsilon", type=float)
        p.add_argument("--ot-max-iterations", type=int)
        p.add_argument("--ot-tolerance", type=float)
        p.add_argument(
            "--source-cache", help="canonical generated x50 training cache directory"
        )
        for name in (
            "trajectory-batch-size",
            "trajectory-ode-steps",
            "trajectory-samples-per-path",
            "real-ot-points",
            "trajectory-checkpoint-block",
            "gradient-diagnostic-interval",
        ):
            p.add_argument("--" + name, type=int)
        p.add_argument("--trajectory-dt", type=float)
    elif command == "cache_x50":
        p.add_argument("--campaign", required=True)
        p.add_argument("--role", choices=["train", "evaluation"], default="train")
        p.add_argument("--cache-size", type=int)
        p.add_argument("--seed", type=int)
        p.add_argument("--start-t", type=int, default=50)
        p.add_argument("--batch-size", type=int, default=50)
        p.add_argument("--device", default="cuda")
    elif command == "occupation":
        p.add_argument("--checkpoint", required=True)
        p.add_argument(
            "--source-cache",
            required=True,
            help="independently generated evaluation x50 cache",
        )
        p.add_argument("--data")
        p.add_argument("--device", default="cpu")
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--ot-max-iterations", type=int)
        p.add_argument("--sw-projections", type=int)
        p.add_argument("--sw-points", type=int)
        p.add_argument("--eval-seed", type=int)
        p.add_argument("--eval-trajectories", type=int)
    elif command == "plot_occupation":
        p.add_argument(
            "--input",
            nargs="+",
            required=True,
            help="one or more completed occupation analysis directories",
        )
    elif command == "sample":
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--num-samples", type=int)
        p.add_argument("--batch-size", type=int)
        p.add_argument("--post-ode-dt", type=float)
        p.add_argument("--max-state-norm", type=float)
        p.add_argument("--device", default="cpu")
    elif command in ("analyze", "embed"):
        p.add_argument("--trajectory", required=True)
        p.add_argument(
            "--data",
            help="optional relocated identical h5ad; exact gene order is checked",
        )
        p.add_argument("--device", default="cpu")
        p.add_argument(
            "--ot-max-iterations",
            type=int,
            help="explicit evaluation solver cap; epsilon/tolerance unchanged",
        )
        p.set_defaults(action=command)
    elif command == "plot":
        p.add_argument(
            "--input", required=True, help="completed analyze/ or embed/ run directory"
        )
    else:
        raise ValueError(command)
    return p


def main(command, argv=None):
    args = parser(command).parse_args(argv)
    if command in ("train", "cache_x50"):
        if Path(args.campaign).name != args.campaign or args.campaign in (
            ".",
            "..",
            "",
        ):
            raise ValueError("campaign must be one safe path component")
        if command == "cache_x50":
            from .training.source_cache import cache_command

            cache_command(args)
        else:
            from .training.runner import train

            train(args)
    elif command == "occupation":
        from .analysis.occupation import occupation_command

        occupation_command(args)
    elif command == "plot_occupation":
        from .analysis.occupation_plotting import plot_occupation

        plot_occupation(args)
    elif command == "sample":
        from .training.checkpoints import restore
        from .sampling.trajectory import sample_to_disk

        model, meta = restore(args.checkpoint, args.device)
        config = dict(meta["effective_config"])
        for key in ("post_ode_dt", "max_state_norm"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        seed_all(config["seed"])
        output = SUITE / "results" / config["condition"] / run_id()
        provenance = {
            **meta,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": file_hash(args.checkpoint),
        }
        sample_to_disk(
            model,
            build_diffusion(config),
            output,
            config,
            count=args.num_samples
            if args.num_samples is not None
            else config["num_samples"],
            batch_size=args.batch_size
            if args.batch_size is not None
            else config["sample_batch_size"],
            device=args.device,
            provenance=provenance,
        )
        print(f"TRAJECTORY_DIR={output}")
    elif command in ("analyze", "embed"):
        from .analysis.runner import analyze

        analyze(args)
    else:
        from .analysis.plotting import plot

        plot(args)
