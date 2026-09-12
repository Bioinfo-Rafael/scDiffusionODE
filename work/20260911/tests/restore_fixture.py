"""Real source model classes + synthetic EMA roundtrip, no training or source writes.

Not a test of a user's trained artifact: those are absent locally. Runs in a fresh
process per suite. Only the source run-root constant is redirected into a temporary
fixture under this work directory; all loaders and model classes are unchanged.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import torch

WORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WORK), str(WORK.parents[1])]
from src.model_runs import discover_suite, restore, select_run
from src.source_imports import activate_suite, analysis_helpers, import_file, umap_core


def main(suite: str) -> None:
    common, _ = activate_suite(suite)
    helpers = analysis_helpers(suite)
    from models import build_model_from_config

    torch.set_num_threads(1)
    experiment = (
        "standard_hybrid_lincomb__hill_after_linear"
        if suite == "20260803"
        else "linear_centered_signed_hill"
    )
    config = common.load_experiment_config(common.CONFIG_ROOT / f"{experiment}.json")
    with tempfile.TemporaryDirectory(dir=WORK / "results") as tmp:
        root = Path(tmp)
        config.update(
            field_hidden=16,
            time_dim=8,
            cell_unet_hidden_num=[32, 24, 16, 16],
            data_dir=str(root / "data.h5ad"),
            edge_tsv_path=str(root / "edges.tsv"),
        )
        genes = ["source", "target", "g2", "g3", "g4"]
        ad.AnnData(
            np.ones((4, len(genes)), dtype=np.float32),
            var=pd.DataFrame({"gene_name": genes}, index=genes),
        ).write_h5ad(config["data_dir"])
        Path(config["edge_tsv_path"]).write_text("from\tto\nsource\ttarget\n")
        run = root / "runs" / experiment / "fixture"
        checkpoints = run / "train/checkpoints/segment_000/attempt_000"
        checkpoints.mkdir(parents=True)
        checkpoint = checkpoints / "ema_0.9999_030000.pt"
        diffusion = helpers.build_diffusion(config)
        model = build_model_from_config(config, genes, diffusion.num_timesteps, "cpu").eval()
        # Distinct wrapped EMA state verifies clean_state_dict and strict loading.
        with torch.no_grad():
            next(model.parameters()).add_(0.123)
        state = {"module." + k: v for k, v in model.state_dict().items()}
        torch.save({"ema": {"state_dict": state}}, checkpoint)
        common.write_json(run / "exp_config.json", config)
        manifest = {
            "ema_checkpoint_path": str(checkpoint),
            "stages": {"train": {"status": "completed"}},
        }
        common.write_json(run / "manifest.json", manifest)
        before = {str(p): common.file_sha256(p) for p in root.rglob("*") if p.is_file()}
        with patch.object(common, "RUNS_ROOT", root / "runs"):
            selected = select_run(suite, run)
            restored, actual_diffusion, loaded_genes, device = restore(selected, "cpu")
            assert actual_diffusion.num_timesteps == 1000 and loaded_genes == genes
            assert not restored.training and str(device) == "cpu"
            for key, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
            x, t = torch.randn(2, len(genes)), torch.tensor([0, 999])
            with torch.no_grad():
                torch.testing.assert_close(restored(x, t), model(x, t), rtol=0, atol=0)
            assert len(discover_suite(suite)["experiments"][experiment]["candidates"]) == 1
            other = "20260816" if suite == "20260803" else "20260803"
            try:
                activate_suite(other)
            except RuntimeError:
                pass
            else:
                raise AssertionError("source suite collision was not rejected")
        with patch.object(common, "RUNS_ROOT", root / "runs"):
            sys.path.insert(0, str(WORK / "scripts"))
            command = import_file("_posthoc_cli_restore", WORK / "scripts/run_one_model.py")
            with patch.object(
                sys,
                "argv",
                [
                    "run_one_model.py",
                    "--suite",
                    suite,
                    "--run-dir",
                    str(run),
                    "--restore-only",
                    "--device",
                    "cpu",
                ],
            ):
                command.main()
            metadata = command.sampling_metadata(
                type("Args", (), {"suite": suite, "batch_size": 2})(),
                selected,
                dict(config),
                genes,
                actual_diffusion,
                device,
                1234,
                "fixture",
            )
            assert metadata["gene_order_hash"] == umap_core().gene_order_hash(genes)
            assert metadata["snapshot_table"][-1]["reverse_step"] == 1000
            assert metadata["actual_diffusion_timesteps"][-1] == 0
        after = {str(p): common.file_sha256(p) for p in root.rglob("*") if p.is_file()}
        assert before == after, "restore modified fixture source files"
        print(
            json.dumps(
                {
                    "status": "fixture_restore_passed",
                    "suite": suite,
                    "experiment": experiment,
                    "genes": len(genes),
                    "diffusion_steps": 1000,
                    "source_files_unchanged": True,
                }
            )
        )


if __name__ == "__main__":
    main(sys.argv[1])
