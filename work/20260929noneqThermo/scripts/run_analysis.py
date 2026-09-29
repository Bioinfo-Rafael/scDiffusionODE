#!/usr/bin/env python3
"""Thin Stage1 entry point into the shared 20260821 analysis."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
PRIOR = ROOT / "work/20260821_noeqThermo"
for path in (ROOT, HERE, PRIOR / "src"):
    sys.path.insert(0, str(path))


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", help="original 20260915 Stage1 EMA used by 20260916")
    source.add_argument("--campaign", help="20260916 runs/additive_* directory; uses canonical Stage1")
    p.add_argument("--data", help="h5ad path; default: checkpoint metadata data_dir")
    p.add_argument("--timesteps", nargs="+", type=int, default=None)
    p.add_argument("--config", help="20260929 entry configuration JSON")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--output-dir")
    p.add_argument("--dry-run", action="store_true", help="validate checkpoint/data/config without writing outputs")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from cellunet_adapter import CellUNetVelocity, restore_stage1
    from noeqthermo.config import load_config
    from noeqthermo.pipeline import run_pipeline
    import scanpy as sc

    settings = json.loads(Path(args.config or HERE / f"configs/{args.mode}.json").read_text())
    if settings["mode"] != args.mode:
        raise ValueError("entry configuration mode does not match --mode")
    config = load_config(args.mode, HERE / settings["analysis_config"])
    config["plot"]["shared_scales"] = True
    timesteps = args.timesteps if args.timesteps is not None else settings["timesteps"]
    if (not timesteps or len(set(timesteps)) != len(timesteps)
            or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in timesteps)):
        raise ValueError("timesteps must be distinct nonnegative integers")
    source_config = json.loads((ROOT / "work/20260916_x0predict_hybrid_additive/configs/base.json").read_text())
    checkpoint = args.checkpoint or source_config["stage1_checkpoint"]
    model, meta, checkpoint, file_hash = restore_stage1(checkpoint, args.campaign)
    data = Path(args.data or meta["effective_config"]["data_dir"]).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"h5ad not found: {data}; pass --data")
    data_hash = file_hash(data)
    if data_hash != meta["data_sha256"]:
        raise ValueError("h5ad SHA256 differs from Stage1 training data")
    adata = sc.read_h5ad(data, backed="r")
    try:
        genes = list(adata.var["gene_name"].astype(str)) if "gene_name" in adata.var else list(adata.var_names.astype(str))
        if genes != meta["gene_names"]:
            raise ValueError("h5ad gene order differs from Stage1 checkpoint")
        for key in ("Superclass", config["selection"]["celltype_key"]):
            if key not in adata.obs:
                raise ValueError(f"missing h5ad.obs column: {key}")
        if not (adata.obs["Superclass"].astype(str) == "Erythropoietic").any():
            raise ValueError("no Erythropoietic cells")
    finally:
        adata.file.close()
    import importlib
    importlib.import_module("work.20260916_x0predict_hybrid_additive.common").seed_all(config["seed"])
    velocity = CellUNetVelocity(model, genes, settings["batch_size"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output_dir or HERE / "outputs" / f"{args.mode}_{stamp}").expanduser().resolve()
    if not output.is_relative_to(HERE / "outputs"):
        raise ValueError(f"output must be below {HERE / 'outputs'}")
    inputs = [dict(
        name=f"t_{t:04d}", timestep=t, config=meta["effective_config"],
        run_dir=checkpoint.parent, checkpoint=checkpoint, data_path=data,
        plot_label=f"Stage1 V(x)=CellUNet(x,t), raw START_X; t={t}"
                   + (" (outside training range 0-999)" if t > 999 else ""),
    ) for t in timesteps]
    # Protect SDE resume against changes to inputs, numerical settings or implementation.
    source_paths = set()
    for directory in (HERE, PRIOR / "src", ROOT / "work/20260817_vector_field_analysis",
                      ROOT / "work/20260915_x0predict", ROOT / "work/20260916_x0predict_hybrid_additive",
                      ROOT / "guided_diffusion", ROOT / "ODE"):
        source_paths.update(p for p in directory.rglob("*.py") if "outputs" not in p.parts)
    identity = dict(
        checkpoint_sha256=file_hash(checkpoint), data_sha256=data_hash,
        timesteps=timesteps, device=args.device, batch_size=settings["batch_size"],
        config={k: v for k, v in config.items() if not k.startswith("_")},
        sources={str(p.relative_to(ROOT)): file_hash(p) for p in sorted(source_paths)},
    )
    summary = dict(checkpoint=str(checkpoint), data=str(data), timesteps=timesteps,
                   output_dir=str(output), definition="V_t(x)=CellUNet(x,t), raw START_X output",
                   outside_training_range=[t for t in timesteps if t > 999], sampling_used=False)
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return 0
    result = run_pipeline([checkpoint.parent], config, output, device_name=args.device,
                          prepared_inputs=inputs, velocity_provider=velocity, resume_identity=identity)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
