"""Verify each 20260802 config builds a Hybrid Softmax model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
WORK_0707 = REPO_ROOT / "work" / "20260707_lincomb"
for path in (REPO_ROOT, WORK_0707, WORK_0707 / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import build_model_from_config, load_experiment_config  # noqa: E402


CONFIG_DIR = REPO_ROOT / "work" / "20260802" / "configs"


class NormModes20260802Test(unittest.TestCase):
    def test_all_configs_build_with_softmax_coefficients(self):
        for config_path in sorted(CONFIG_DIR.glob("*.json")):
            with self.subTest(config=config_path.name):
                config = load_experiment_config(config_path)
                if config.get("t_s") == "auto":
                    config["t_s"] = 616
                config.update({"use_mask_reg": False, "K": 3, "field_hidden": 16})
                model = build_model_from_config(config, ["g0", "g1", "g2", "g3"], 1000, "cpu").eval()
                field = model.ode_model
                coefficients = field.get_gate_values(torch.randn(2, 4), torch.tensor([0, 616]))["coefficients"]
                self.assertTrue(torch.allclose(coefficients.sum(dim=-1), torch.ones(2), atol=1e-6))
                if config["hybrid_norm_mode"] == "ratio_reg":
                    self.assertEqual(field.ratio_reg_weight, 5.0)
                if config["hybrid_norm_mode"] == "normed_learned_scale":
                    self.assertTrue(hasattr(model, "log_scale"))
                if config["hybrid_norm_mode"] == "scale_model":
                    self.assertIsNotNone(model.scale_model)
                    self.assertEqual(config["ode_input_source"], "none")


if __name__ == "__main__":
    unittest.main()
