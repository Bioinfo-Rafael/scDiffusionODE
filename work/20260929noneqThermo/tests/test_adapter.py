"""Verify raw output semantics and literal time values on a real CellUNet."""
import sys
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from cellunet_adapter import CellUNetVelocity, restore_stage1
from guided_diffusion.cell_model import Cell_Unet


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = Cell_Unet(input_dim=8, hidden_num=[16, 8, 4, 4]).eval()
        self.genes = [f"g{i}" for i in range(8)]
        self.X = np.random.default_rng(12).normal(size=(11, 8)).astype(np.float32)
        self.provider = CellUNetVelocity(self.model, self.genes, batch_size=3)

    def test_literal_forward_for_all_times(self):
        for t in (100, 900, 1000):
            actual, metadata = self.provider(self.X, self.genes, {"timestep": t}, "cpu")
            with torch.no_grad():
                expected = self.model(torch.from_numpy(self.X), torch.full((11, 1), t)).numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
            self.assertEqual(metadata["timestep"], t)
            self.assertEqual(metadata["outside_training_timestep_range"], t == 1000)
            self.assertFalse(metadata["sampling_used"])

    def test_reject_gene_order_and_nonfinite(self):
        with self.assertRaisesRegex(ValueError, "gene order"):
            self.provider(self.X, self.genes[::-1], {"timestep": 100}, "cpu")
        self.X[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.provider(self.X, self.genes, {"timestep": 100}, "cpu")


class CanonicalCheckpointTests(unittest.TestCase):
    def test_restore_campaign_stage1_and_reject_changed_checkpoint(self):
        common = importlib.import_module("work.20260916_x0predict_hybrid_additive.common")
        models = importlib.import_module("work.20260916_x0predict_hybrid_additive.models")
        config = common.effective_config(common.STAGE1)
        config["cell_unet_hidden_num"] = [16, 8, 4, 4]
        genes = [f"g{i}" for i in range(8)]
        model = models.build_model(config, genes)
        with tempfile.TemporaryDirectory(prefix="test_noeq_", dir=common.SUITE) as temporary:
            campaign = Path(temporary)
            (campaign / "stage1").mkdir()
            checkpoint = campaign / "stage1/ema_0.9999_030000.pt"
            metadata = dict(effective_config=config, gene_names=genes,
                            checkpoint_kind="ema", step=30000)
            torch.save(dict(state_dict=model.state_dict(), metadata=metadata), checkpoint)
            record = dict(checkpoint=str(checkpoint), checkpoint_sha256=common.file_hash(checkpoint),
                          cellunet_hash=common.state_hash(model))
            (campaign / "canonical_stage1.json").write_text(json.dumps(record))
            restored, meta, path, _ = restore_stage1(campaign=campaign)
            self.assertEqual(path, checkpoint)
            self.assertEqual(common.state_hash(restored), common.state_hash(model))
            self.assertFalse(restored.training)
            self.assertEqual(meta["step"], 30000)
            with checkpoint.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "changed/escaped"):
                restore_stage1(campaign=campaign)


if __name__ == "__main__":
    unittest.main(verbosity=2)
