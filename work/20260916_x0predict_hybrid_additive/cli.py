"""Single worker CLI; native START_X sampling plus unchanged legacy evaluation."""
import argparse
from pathlib import Path
from .common import CONDITIONS, build_diffusion, file_hash, result_root, run_id, seed_all


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest="action", required=True)
    train = commands.add_parser("train")
    train.add_argument("--campaign", required=True)
    train.add_argument("--condition", choices=CONDITIONS, required=True)
    train.add_argument("--device", default="cuda")
    sample = commands.add_parser("sample")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--device", default="cuda")
    for action in ("analyze", "embed"):
        p = commands.add_parser(action)
        p.add_argument("--trajectory", required=True)
        p.add_argument("--device", default="cpu")
        p.set_defaults(data=None)
    commands.add_parser("plot").add_argument("--input", required=True)
    args = parser.parse_args(argv)
    if args.action == "train":
        from .training.runner import train
        train(args)
    elif args.action == "sample":
        from .training.checkpoints import restore
        from .sampling.trajectory import sample_to_disk
        from .analysis.diagnostics import BranchRecorder, RecordingCellOnly
        model, meta = restore(args.checkpoint, args.device)
        config = meta["effective_config"]
        recorder = BranchRecorder(deduplicate=True)
        if hasattr(model, "ode_model"):
            model.diagnostic_sink = recorder
        else:
            model = RecordingCellOnly(model, recorder)
        seed_all(config["seed"])
        output = result_root(config) / config["condition"] / run_id()
        sample_to_disk(model, build_diffusion(config), output, config, count=config["num_samples"],
                       batch_size=config["sample_batch_size"], device=args.device,
                       provenance=dict(meta, checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=file_hash(args.checkpoint)))
        recorder.save(output)
        print(f"TRAJECTORY_DIR={output}")
    elif args.action in ("analyze", "embed"):
        from .analysis.runner import analyze
        analyze(args)
    else:
        from .analysis.plotting import plot
        plot(args)


if __name__ == "__main__":
    main()
