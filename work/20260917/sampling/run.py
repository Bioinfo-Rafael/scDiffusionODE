from pathlib import Path
from ..common import build_diffusion, file_hash, result_root, run_id, seed_all
from ..training.checkpoints import restore
from .trajectory import sample_to_disk
from ..analysis.diagnostics import BranchRecorder, RecordingCellOnly


def sample(checkpoint, device):
    model, meta = restore(checkpoint, device)
    config = meta["effective_config"]
    recorder = BranchRecorder(deduplicate=True)
    if hasattr(model, "ode_model"):
        model.diagnostic_sink = recorder
    else:
        model = RecordingCellOnly(model, recorder)
    seed_all(config["seed"])
    output = result_root(config) / config["condition"] / "sampling" / run_id()
    sample_to_disk(model, build_diffusion(config), output, config, count=config["num_samples"],
                   batch_size=config["sample_batch_size"], device=device,
                   provenance=dict(meta, checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=file_hash(checkpoint)))
    recorder.save(output)
    print(f"TRAJECTORY_DIR={output}", flush=True)
    return output
