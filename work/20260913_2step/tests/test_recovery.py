"""Synthetic CPU solver and checkpoint continuation regressions; no experiment jobs."""

import copy
import contextlib
import io
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
PKG = "work.20260913_2step"
common = importlib.import_module(PKG + ".common")
models = importlib.import_module(PKG + ".models")
cp = importlib.import_module(PKG + ".training.checkpoints")
recovery = importlib.import_module(PKG + ".training.recovery")
sinkhorn = importlib.import_module(PKG + ".losses.sinkhorn")
spec = importlib.util.spec_from_file_location(
    "recovery_launcher", common.SUITE / "scripts/run_all.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
torch.set_num_threads(1)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        common.seed_all(1234)
        parent = common.SUITE / ".test_tmp"
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.campaign = common.new_dir(self.root / "runs" / "old_campaign")
        self.genes = ["a", "b", "c", "d"]
        edges = self.root / "edges.tsv"
        edges.write_text("from\tto\na\tb\n")
        c1 = common.effective_config("stage1_cellunet")
        c1.update(
            cell_unet_hidden_num=[8, 8, 4, 4],
            edge_tsv_path=str(edges),
            field_hidden=8,
            time_dim=8,
        )
        self.stage1 = models.build_model(c1, self.genes)
        first = (
            common.new_dir(self.campaign / "stage1_cellunet" / "first" / "checkpoints")
            / "ema_0.9999_030000.pt"
        )
        cp.save_checkpoint(
            first,
            self.stage1.state_dict(),
            {"effective_config": c1, "checkpoint_kind": "ema", "step": 30000},
        )
        self.canonical = {
            "checkpoint": str(first),
            "checkpoint_sha256": common.file_hash(first),
            "cellunet_hash": common.state_hash(self.stage1),
            "gene_order_hash": "synthetic-gene-order",
            "edge_tsv_sha256": common.file_hash(edges),
        }
        common.write_json(self.campaign / "canonical_stage1.json", self.canonical)
        self.condition = "hill_after_linear_ot_soft"
        self.config = common.effective_config(self.condition)
        self.config.update(
            cell_unet_hidden_num=[8, 8, 4, 4],
            edge_tsv_path=str(edges),
            field_hidden=8,
            time_dim=8,
        )
        self.config["ot"]["max_iterations"] = 200  # real failure used the old cap
        self.model = models.build_model(self.config, self.genes)
        models.freeze_from_stage1(self.model, self.stage1.state_dict())
        self.model.train()
        self.optimizer = models.optimizer_for(self.model, self.config)
        self.ema = copy.deepcopy(self.model.state_dict())
        self.update(self.model, self.optimizer, self.ema)
        self.optimizer.param_groups[0]["lr"] = 1e-4 * (1 - 4999 / 30000)
        self.run = common.new_dir(self.campaign / self.condition / "old_run")
        self.checkpoints = common.new_dir(self.run / "checkpoints")
        self.metadata = {
            "effective_config": self.config,
            "gene_names": self.genes,
            "gene_order_hash": self.canonical["gene_order_hash"],
            "step": 5000,
            "originating_stage1": self.canonical,
            "checkpoint_kind": "raw",
            "frozen_cellunet_hash_before": self.canonical["cellunet_hash"],
        }
        self.raw = self.checkpoints / "model005000.pt"
        cp.save_checkpoint(self.raw, self.model.state_dict(), self.metadata)
        cp.save_checkpoint(
            self.checkpoints / "ema_0.9999_005000.pt",
            self.ema,
            {**self.metadata, "checkpoint_kind": "ema"},
        )
        with (self.checkpoints / "opt005000.pt").open("xb") as f:
            torch.save(self.optimizer.state_dict(), f)

    def update(self, model, optimizer, ema):
        optimizer.zero_grad(set_to_none=True)
        sum(p.square().sum() for p in model.ode_model.parameters()).backward()
        optimizer.step()
        models.update_ema(ema, model, 0.9999)

    def bundle(self):
        return recovery.load_bundle(
            self.raw, self.campaign, self.condition, self.canonical
        )

    def test_resume_preserves_raw_ema_optimizer_and_next_update(self):
        before = {str(p): common.file_hash(p) for p in self.checkpoints.iterdir()}
        bundle = self.bundle()
        restored = models.build_model(
            self.config, self.genes, state=bundle["raw"]["state_dict"]
        )
        optimizer = models.optimizer_for(restored, self.config)
        ema = recovery.restore_training_state(
            bundle, restored, optimizer, self.canonical["cellunet_hash"]
        )
        self.assertEqual(common.state_hash(restored), common.state_hash(self.model))
        self.assertEqual(common.state_hash(ema), common.state_hash(self.ema))
        self.assertEqual(
            optimizer.param_groups[0]["lr"], self.optimizer.param_groups[0]["lr"]
        )
        self.update(self.model, self.optimizer, self.ema)
        self.update(restored, optimizer, ema)
        self.assertEqual(common.state_hash(restored), common.state_hash(self.model))
        self.assertEqual(common.state_hash(ema), common.state_hash(self.ema))
        self.assertEqual(bundle["provenance"]["step"], 5000)
        self.assertFalse(bundle["provenance"]["bitwise_continuation"])
        models.assert_frozen(restored, optimizer, self.canonical["cellunet_hash"])
        self.assertEqual(
            before, {str(p): common.file_hash(p) for p in self.checkpoints.iterdir()}
        )

    def test_resume_changes_only_solver_cap(self):
        new = recovery.resume_config(
            self.bundle(), common.effective_config(self.condition)
        )
        self.assertEqual(new["ot"]["max_iterations"], 2000)
        new["ot"]["max_iterations"] = 200
        self.assertEqual(new, self.config)

    def test_select_latest_complete_bundle_ignores_partial_save(self):
        cp.save_checkpoint(
            self.checkpoints / "model010000.pt",
            self.model.state_dict(),
            {**self.metadata, "step": 10000},
        )
        selected = recovery.select_training(
            self.campaign, self.condition, self.canonical
        )
        self.assertEqual(selected["resume_checkpoint"], str(self.raw))
        self.assertEqual(selected["step"], 5000)

    def test_completed_training_is_reused(self):
        path = self.checkpoints / "ema_0.9999_030000.pt"
        cp.save_checkpoint(
            path, self.ema, {**self.metadata, "checkpoint_kind": "ema", "step": 30000}
        )
        common.write_json(
            self.run / "completed.json",
            {"checkpoint": str(path), "checkpoint_sha256": common.file_hash(path)},
        )
        selected = recovery.select_training(
            self.campaign, self.condition, self.canonical
        )
        self.assertEqual(selected, {"completed_checkpoint": str(path)})

    def test_corrupt_or_wrong_origin_bundle_rejected(self):
        with self.assertRaises(ValueError):
            recovery.load_bundle(
                self.raw,
                self.campaign,
                self.condition,
                {**self.canonical, "checkpoint_sha256": "wrong"},
            )
        with self.assertRaises(ValueError):
            recovery.load_bundle(
                self.raw, self.campaign.parent / "other", self.condition, self.canonical
            )
        with self.raw.open("ab") as f:
            f.write(b"bad")
        with self.assertRaises(ValueError):
            self.bundle()

    def test_recovery_plan_skips_four_completed_trainings(self):
        args = launcher.parser().parse_args(["--campaign", "old_campaign"])
        manifest = {"campaign": "old_campaign", "steps": launcher.plan(args)}

        def selection(campaign, condition, canonical):
            if condition.endswith("_recon_soft"):
                return {"completed_checkpoint": "/fixture/" + condition + ".pt"}
            if condition == self.condition:
                return {"resume_checkpoint": str(self.raw), "step": 5000}
            return {"start_fresh": True}

        with (
            patch.object(launcher, "SUITE", self.root),
            patch.object(recovery, "select_training", side_effect=selection),
        ):
            steps, artifacts, _ = launcher.recovery_steps(manifest)
        self.assertEqual(len(steps), 47)
        self.assertEqual(steps[0]["name"], "x50.train_cache")
        training = [s for s in steps if s["name"].endswith(".train")]
        self.assertEqual(len(training), 3)
        self.assertTrue(all("_trajectory_ot_soft.train" in s["name"] for s in training))
        self.assertTrue(all("--resume-checkpoint" not in s["argv"] for s in training))
        self.assertEqual(len(artifacts), 4)
        for step in steps:
            if step["name"].endswith(".analyze"):
                self.assertIn("--ot-max-iterations", step["argv"])

    def test_recovery_dry_run_uses_selected_plan_without_launching(self):
        selections = {"stage1_cellunet": {"completed_checkpoint": "fixture"}}
        with (
            patch.object(
                launcher, "recovery_steps", return_value=([], {}, selections)
            ) as selected,
            patch.object(launcher.subprocess, "Popen") as spawn,
        ):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(
                    launcher.main(["--resume-campaign", "old_campaign", "--dry-run"]), 0
                )
        selected.assert_called_once()
        spawn.assert_not_called()
        self.assertIn("RECOVERY_SELECTION=", output.getvalue())
        self.assertNotIn("train.py", output.getvalue())

    def test_completed_analysis_includes_finished_ot_and_never_partial_training(self):
        args = launcher.parser().parse_args(
            ["--analyze-completed-campaign", "old_campaign"]
        )
        args.campaign = args.analyze_completed_campaign
        manifest = dict(
            campaign=args.campaign,
            steps=launcher.plan(args),
            analysis_only=True,
            completed_analysis=True,
        )
        finished = set(launcher.CONDITIONS[1:5])

        def select(campaign, condition, canonical, *, completed_only=False):
            self.assertTrue(completed_only)
            return (
                {"completed_checkpoint": "/fixture/" + condition + ".pt"}
                if condition in finished
                else {"missing_completed_checkpoint": True}
            )

        with (
            patch.object(launcher, "SUITE", self.root),
            patch.object(recovery, "select_training", side_effect=select),
            patch.object(
                recovery, "load_bundle", side_effect=AssertionError("partial bundle")
            ),
        ):
            steps, artifacts, selections = launcher.recovery_steps(manifest)
        self.assertEqual(len(steps), 31)
        self.assertEqual(len(artifacts), 5)
        self.assertEqual(sum(s["name"].endswith(".sample") for s in steps), 5)
        self.assertEqual(sum(s["name"].endswith(".occupation") for s in steps), 4)
        self.assertFalse(any(s["name"].endswith(".train") for s in steps))
        self.assertFalse(
            any(s["name"].startswith(tuple(launcher.CONDITIONS[5:])) for s in steps)
        )
        available = set(artifacts)
        for step in steps:
            self.assertNotIn("train.py", [Path(arg).name for arg in step["argv"]])
            for arg in step["argv"]:
                if arg.startswith("@"):
                    self.assertIn(arg, available)
            available.update(step["capture"].values())
        self.assertEqual(steps[-1]["name"], "occupation.comparison_plot")
        self.assertEqual(sum(arg.startswith("@") for arg in steps[-1]["argv"]), 4)

    def test_analysis_only_plan_has_twenty_postprocessing_steps(self):
        args = launcher.parser().parse_args(["--analyze-campaign", "old_campaign"])
        args.campaign = args.analyze_campaign
        steps = launcher.plan(args)
        self.assertEqual(len(steps), 20)
        self.assertFalse(
            any("train.py" in arg for step in steps for arg in step["argv"])
        )
        self.assertFalse(any("_ot_soft" in step["name"] for step in steps))
        manifest = {"campaign": "old_campaign", "steps": steps, "analysis_only": True}

        def selection(campaign, condition, canonical, *, completed_only=False):
            self.assertTrue(completed_only)
            self.assertTrue(condition.endswith("_recon_soft"))
            return {"completed_checkpoint": "/fixture/" + condition + ".pt"}

        with (
            patch.object(launcher, "SUITE", self.root),
            patch.object(recovery, "select_training", side_effect=selection) as select,
        ):
            actual, artifacts, _ = launcher.recovery_steps(manifest)
        self.assertEqual(actual, steps)
        self.assertEqual(select.call_count, 3)
        self.assertEqual(len(artifacts), 4)

    def test_analysis_only_refuses_missing_completed_condition(self):
        args = launcher.parser().parse_args(["--analyze-campaign", "old_campaign"])
        manifest = {
            "campaign": "old_campaign",
            "steps": launcher.plan(args),
            "analysis_only": True,
        }
        with (
            patch.object(launcher, "SUITE", self.root),
            patch.object(
                recovery,
                "select_training",
                return_value={"missing_completed_checkpoint": True},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "no training will be started"):
                launcher.recovery_steps(manifest)

    def test_completed_only_does_not_load_partial_bundle(self):
        with patch.object(
            recovery, "load_bundle", side_effect=AssertionError("partial bundle loaded")
        ):
            self.assertEqual(
                recovery.select_training(
                    self.campaign, self.condition, self.canonical, completed_only=True
                ),
                {"missing_completed_checkpoint": True},
            )

    def test_extended_cap_converges_without_relaxing_tolerance(self):
        generator = torch.Generator().manual_seed(2)
        x = torch.randn((8, 3), generator=generator, dtype=torch.float64)
        y = x + 0.2 * torch.randn((8, 3), generator=generator, dtype=torch.float64)
        with self.assertRaises(FloatingPointError):
            sinkhorn.entropic_ot(x, y, epsilon=0.1, max_iterations=200, tolerance=1e-5)
        _, info = sinkhorn.entropic_ot(
            x, y, epsilon=0.1, max_iterations=2000, tolerance=1e-5
        )
        self.assertGreater(info["iterations"], 200)
        self.assertLessEqual(info["marginal_residual"], 1e-5)

    def test_checkpointed_iterations_match_unrolled_gradients(self):
        generator = torch.Generator().manual_seed(2)
        x = torch.randn(
            (8, 3), generator=generator, dtype=torch.float64
        ).requires_grad_()
        y = x.detach() + 0.2 * torch.randn(
            (8, 3), generator=generator, dtype=torch.float64
        )
        kwargs = dict(epsilon=0.1, max_iterations=2000, tolerance=1e-5)
        actual, _ = sinkhorn.entropic_ot(x, y, **kwargs)
        actual_grad = torch.autograd.grad(actual, x)[0]
        with patch.object(
            sinkhorn, "checkpoint", side_effect=lambda fn, *args, **kw: fn(*args)
        ):
            expected, _ = sinkhorn.entropic_ot(x, y, **kwargs)
            expected_grad = torch.autograd.grad(expected, x)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
