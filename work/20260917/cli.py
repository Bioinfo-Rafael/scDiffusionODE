"""Train, sample and evaluate without editing individual condition files."""
import argparse
from pathlib import Path
from .common import CONDITIONS, MODELS, LOSSES, read_json, effective_config


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest="action", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True, help="resolved campaign config")
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cuda")
    sample = commands.add_parser("sample")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--device", default="cuda")
    for action in ("analyze", "embed"):
        p = commands.add_parser(action)
        p.add_argument("--trajectory", required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--data")
    commands.add_parser("plot").add_argument("--input", required=True)
    commands.add_parser("summarize").add_argument("--campaign", required=True)
    args = parser.parse_args(argv)
    if args.action == "train":
        from .training.runner import train
        checkpoint = train(read_json(args.config), Path(args.output), args.device)
        print(f"EMA_CHECKPOINT={checkpoint}", flush=True)
    elif args.action == "sample":
        from .sampling.run import sample
        sample(args.checkpoint, args.device)
    elif args.action in ("analyze", "embed"):
        from .analysis.runner import analyze
        analyze(args)
    elif args.action == "plot":
        from .analysis.plotting import plot
        plot(args)
    else:
        from .scripts.summarize import summarize
        summarize(args.campaign)


if __name__ == "__main__":
    main()
