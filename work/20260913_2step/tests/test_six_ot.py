"""Launcher tests with mocked jobs; no real training/sampling or plots."""

import contextlib
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PKG = "work.20260913_2step"
common = importlib.import_module(PKG + ".common")
six = importlib.import_module(PKG + ".scripts.six_ot")
launcher = importlib.import_module(PKG + ".scripts.run_all")
cp = importlib.import_module(PKG + ".training.checkpoints")
recovery = importlib.import_module(PKG + ".training.recovery")


class SixTests(unittest.TestCase):
    def setUp(self):
        parent = common.SUITE / ".test_tmp"
        parent.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.args = launcher.parser().parse_args(
            ["--run-six-ot", "--stage1-campaign", "old", "--device", "cpu"]
        )

    def test_exact_matrix_dependency_order_and_no_stage1_training(self):
        steps = six.plan(self.args, "hybrid500_ot_fixture")
        self.assertEqual(len(steps), 45)
        trains = [s for s in steps if s["name"].endswith(".train")]
        self.assertEqual(len(trains), 6)
        self.assertTrue(all("_ot_soft.train" in s["name"] for s in trains))
        self.assertFalse(
            any("stage1_cellunet" in s["name"] or "recon" in s["name"] for s in steps)
        )
        available = set()
        for step in steps:
            for arg in step["argv"]:
                if arg.startswith("@"):
                    self.assertIn(arg, available)
            available.update(step["capture"].values())

    def test_failure_continues_and_resume_reuses_completed_outputs(self):
        name = "hybrid500_ot_fixture"
        steps = six.plan(self.args, name)
        # Exercise the whole dependency graph with tiny text artifacts.
        root = self.path / "launches" / name
        first = root / "a"
        first.mkdir(parents=True)
        manifest = dict(six_ot=True, campaign=name, device="cpu", steps=steps)
        common.write_json(first / "launch.json", manifest)
        calls = []

        def fake(step, artifacts, launch):
            calls.append(step["name"])
            if launch == first and step["name"] == "hill_after_linear_ot_soft.sample":
                raise RuntimeError("synthetic failure")
            for key in step["capture"].values():
                output = common.new_dir(launch / (step["name"] + "_output"))
                common.write_json(output / "completed.json", {"status": "completed"})
                artifacts[key] = str(output)

        with (
            patch.object(cp, "canonical_stage1", return_value=({}, {})),
            patch.object(
                recovery, "select_training", return_value={"start_fresh": True}
            ),
            patch.object(recovery, "verified_checkpoint"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(six.execute(manifest, first, run_step=fake), 1)
            self.assertIn("shifted_hill_rho_trajectory_ot_soft.umap_plot", calls)
            self.assertNotIn("hill_after_linear_ot_soft.analyze", calls)
            before = {p: p.read_bytes() for p in first.rglob("*") if p.is_file()}
            second = common.new_dir(root / "b")
            common.write_json(second / "launch.json", manifest)
            calls.clear()
            self.assertEqual(six.execute(manifest, second, run_step=fake), 0)
            self.assertIn("hill_after_linear_ot_soft.sample", calls)
            self.assertNotIn("centered_signed_hill_ot_soft.sample", calls)
            self.assertFalse(any(n.endswith(".train") for n in calls))
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_schedule_recovery_and_output_isolation(self):
        condition = common.LEGACY_CONDITIONS[0]
        old = common.effective_config(condition)
        new = common.effective_config(condition, "hybrid500_ot_v1")
        with self.assertRaisesRegex(ValueError, "interpolation"):
            recovery.check_objective(old, new)
        recovery.check_objective(new, new)
        new["output_campaign"] = "hybrid500_ot_fixture"
        self.assertEqual(
            common.result_root(new), common.SUITE / "results/hybrid500_ot_fixture"
        )
        self.assertEqual(common.result_root(old), common.SUITE / "results")
        new["output_campaign"] = "../old"
        with self.assertRaises(ValueError):
            common.result_root(new)

    def test_register_stage1_is_read_only_and_never_overwrites(self):
        source, target = self.path / "old", self.path / "new"
        source.mkdir()
        target.mkdir()
        (source / "sentinel").write_text("unchanged")
        with patch.object(
            cp, "canonical_stage1", return_value=({}, {"checkpoint": "old.pt"})
        ):
            cp.register_stage1(source, target)
            with self.assertRaises(FileExistsError):
                cp.register_stage1(source, target)
        self.assertEqual((source / "sentinel").read_text(), "unchanged")
        self.assertEqual(list(source.iterdir()), [source / "sentinel"])

    def test_launch_dry_run_and_fresh_campaign_isolation(self):
        config_dir = self.path / "configs"
        config_dir.mkdir()
        (config_dir / "hybrid500_ot.json").write_text(
            (common.SUITE / "configs/hybrid500_ot.json").read_text()
        )
        source = self.path / "runs" / "old"
        source.mkdir(parents=True)
        sentinel = source / "sentinel"
        sentinel.write_text("read only")
        data = self.path / "data.fixture"
        data.write_text("synthetic")
        payload = {
            "metadata": {
                "effective_config": {"data_dir": str(data), "edge_tsv_path": str(data)}
            }
        }
        record = {"checkpoint": "fixture.pt", "checkpoint_sha256": "fixture"}
        with (
            patch.object(common, "SUITE", self.path),
            patch.object(cp, "canonical_stage1", return_value=(payload, record)),
            patch.object(six.subprocess, "Popen") as popen,
            patch.object(common, "git_commit", return_value="synthetic"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            popen.return_value.pid = 123
            self.args.dry_run = True
            before = sorted(str(p) for p in self.path.rglob("*"))
            six.launch(self.args)
            self.assertEqual(before, sorted(str(p) for p in self.path.rglob("*")))
            popen.assert_not_called()
            self.args.dry_run = False
            six.launch(self.args)
            six.launch(self.args)
            campaigns = list((self.path / "runs").glob("hybrid500_ot_*"))
            self.assertEqual(len(campaigns), 2)
            for campaign in campaigns:
                self.assertTrue((campaign / "canonical_stage1.json").is_file())
                self.assertEqual(len([p for p in campaign.iterdir() if p.is_dir()]), 6)
            self.assertEqual(popen.call_count, 2)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(sentinel.read_text(), "read only")
            self.assertEqual(list(source.iterdir()), [sentinel])
