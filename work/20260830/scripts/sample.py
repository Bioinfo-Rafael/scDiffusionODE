#!/usr/bin/env python3
"""Restore a trained wrapper and sample in eval mode (CellUnet branch only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for path in (REPO_ROOT, SUITE_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import choose_sampling_checkpoint, read_json, write_json  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", default="")
    args = parser.parse_args(argv)
    import scanpy as sc
    import random
    import torch
    from guided_diffusion import dist_util
    from guided_diffusion.script_util import create_gaussian_diffusion
    from ODE.ode_20260609_mathmlp import clean_state_dict
    from models import build_model_from_config

    run = Path(args.run_dir).resolve()
    run.relative_to((SUITE_ROOT / "runs").resolve())
    config = read_json(run / "exp_config.json")
    checkpoint = Path(args.model_path).resolve() if args.model_path else choose_sampling_checkpoint(run, config["ema_rate"])
    checkpoint.relative_to((run / "checkpoints").resolve())
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    device = torch.device(device_name)
    seed = int(config.get("seed", 1234))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    adata = sc.read_h5ad(config["data_dir"], backed="r")
    try:
        genes = [str(value) for value in adata.var["gene_name"].tolist()]
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    diffusion = create_gaussian_diffusion(
        steps=int(config["diffusion_steps"]),
        noise_schedule=config["noise_schedule"],
        learn_sigma=bool(config["learn_sigma"]),
        use_kl=bool(config["use_kl"]),
        predict_xstart=bool(config["predict_xstart"]),
        rescale_timesteps=bool(config["rescale_timesteps"]),
        rescale_learned_sigmas=bool(config["rescale_learned_sigmas"]),
        timestep_respacing=config["timestep_respacing"],
    )
    model = build_model_from_config(config, genes, diffusion.num_timesteps, device)
    state = clean_state_dict(dist_util.load_state_dict(str(checkpoint), map_location="cpu"))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()  # Wrapper guarantees that this disables ODE forward.
    ode_calls = []
    ode_hook = model.ode_model.register_forward_hook(lambda *args: ode_calls.append(1))
    sample_fn = diffusion.ddim_sample_loop if config["use_ddim"] else diffusion.p_sample_loop
    chunks = []
    remaining = int(config["num_samples"])
    with torch.no_grad():
        while remaining:
            size = min(int(config["sample_batch_size"]), remaining)
            kwargs = {"clip_denoised": bool(config["clip_denoised"])}
            if not config["use_ddim"]:
                kwargs["start_time"] = diffusion.num_timesteps
            sample, _ = sample_fn(model, (size, len(genes)), **kwargs)
            chunks.append(sample.cpu().numpy().astype(np.float32, copy=False))
            remaining -= size
    ode_hook.remove()
    if ode_calls:
        raise RuntimeError("ODE forward was unexpectedly called during eval sampling")
    output_dir = run / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"samples_{checkpoint.stem}"
    output = output_dir / f"{stem}.npz"
    suffix = 1
    while output.exists():
        output = output_dir / f"{stem}_{suffix:03d}.npz"
        suffix += 1
    np.savez_compressed(output, cell_gen=np.concatenate(chunks))
    write_json(output.with_suffix(".json"), {
        "checkpoint": str(checkpoint),
        "sample_path": str(output),
        "ode_forward_call_count": len(ode_calls),
    })
    print(f"SAMPLE_PATH='{output}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
