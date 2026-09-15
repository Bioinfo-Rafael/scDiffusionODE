"""Explicit isolated worker commands; importing never starts a workflow."""
import argparse
from pathlib import Path
from .common import CONDITIONS, STAGE1, build_diffusion, file_hash, result_root, run_id, seed_all


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    sub = parser.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train")
    train.add_argument("--campaign", required=True)
    train.add_argument("--condition", choices=CONDITIONS + [STAGE1], required=True)
    train.add_argument("--device", default="cpu")
    sample = sub.add_parser("sample")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--device", default="cpu")
    sample.add_argument("--num-samples", type=int)
    sample.add_argument("--batch-size", type=int)
    for action in ("analyze", "embed"):
        p = sub.add_parser(action)
        p.add_argument("--trajectory", required=True)
        p.add_argument("--data")
        p.add_argument("--device", default="cpu")
    plot = sub.add_parser("plot")
    plot.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    if args.action == "train":
        from .training.runner import train
        train(args)
    elif args.action == "sample":
        from .training.checkpoints import restore
        from .sampling.trajectory import sample_to_disk
        model, meta = restore(args.checkpoint, args.device)
        config = meta["effective_config"]
        seed_all(config["seed"])
        output = result_root(config) / config["condition"] / run_id()
        sample_to_disk(model, build_diffusion(config), output, config,
                       count=args.num_samples if args.num_samples is not None else config["num_samples"],
                       batch_size=args.batch_size if args.batch_size is not None else config["sample_batch_size"],
                       device=args.device, provenance={**meta, "checkpoint": str(Path(args.checkpoint).resolve()),
                                                      "checkpoint_sha256": file_hash(args.checkpoint)})
        print(f"TRAJECTORY_DIR={output}")
    elif args.action in ("analyze", "embed"):
        from .analysis.runner import analyze
        analyze(args)
    else:
        from .analysis.plotting import plot
        plot(args)


if __name__ == "__main__":
    main()
