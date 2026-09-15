"""Small synthetic CPU regression tests. No real input, GPU, UMAP, or rendering."""

import copy
import csv
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

PKG = "work.20260913_2step"
common = importlib.import_module(PKG + ".common")
models = importlib.import_module(PKG + ".models")
cp = importlib.import_module(PKG + ".training.checkpoints")
recovery = importlib.import_module(PKG + ".training.recovery")
traj = importlib.import_module(PKG + ".training.trajectory")
cache = importlib.import_module(PKG + ".training.source_cache")
sink = importlib.import_module(PKG + ".losses.sinkhorn")
sw = importlib.import_module(PKG + ".analysis.sliced_wasserstein")
occ = importlib.import_module(PKG + ".analysis.occupation")
runner = importlib.import_module(PKG + ".training.runner")
cli = importlib.import_module(PKG + ".cli")
torch.set_num_threads(1)


class LinearField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))

    def forward(self, x, t):
        assert torch.count_nonzero(t) == 0
        return self.a * x


class FakeDiffusion:
    num_timesteps = 1000
    betas = np.linspace(0.0001, 0.02, 1000)

    def __init__(self):
        self.calls, self.first = [], None

    def p_sample(self, model, x, t, clip_denoised):
        assert not torch.is_grad_enabled()
        assert not clip_denoised
        assert not hasattr(model, "ode_model")
        if self.first is None:
            self.first = x.clone()
        self.calls.append(int(t[0]))
        return {"sample": x * 0.999}


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        common.seed_all(1234)
        base = common.SUITE / ".test_tmp"
        base.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.genes = ["a", "b", "c", "d"]
        self.edges = self.path / "edges.tsv"
        self.edges.write_text("from\tto\na\tb\n")
        self.stage1 = models.build_model(self.small("stage1_cellunet"), self.genes)
        self.canonical = dict(
            checkpoint="/fixture/ema.pt",
            checkpoint_sha256="sha",
            cellunet_hash=common.state_hash(self.stage1),
            gene_order_hash="genes",
            edge_tsv_sha256=common.file_hash(self.edges),
        )

    def small(self, condition):
        c = common.effective_config(condition)
        c.update(
            cell_unet_hidden_num=[8, 8, 4, 4],
            field_hidden=8,
            time_dim=8,
            edge_tsv_path=str(self.edges),
            total_steps=2,
            save_interval=1,
            log_interval=1,
        )
        if c["objective"] == "trajectory_ot":
            c["trajectory_ot"].update(
                ode_steps=4,
                trajectory_batch_size=3,
                real_target_points=5,
                checkpoint_block=2,
                gradient_diagnostic_interval=1,
            )
        return c

    def hybrid(self, condition=None):
        c = self.small(condition or common.CONDITIONS[4])
        model = models.build_model(c, self.genes)
        models.freeze_from_stage1(model, self.stage1.state_dict())
        return c, model

    def test_shared_parameters_match_source_unroll_and_gradients(self):
        for name in common.CONDITIONS[5:]:
            c, model = self.hybrid(name)
            model.ode_model.target_chunk_size = 2
            start = torch.randn(3, 4)
            results = []
            for backend in ("source", "shared_parameters_v1"):
                model.zero_grad(set_to_none=True)
                common.seed_all(42)
                states = traj.integrate(
                    model.ode_model,
                    start,
                    steps=5,
                    dt=0.001,
                    checkpoint_block=2,
                    field_backend=backend,
                )
                loss = states.square().mean()
                # Diagnostics followed by normal backward must also work with
                # a shared physical-parameter graph and nested checkpointing.
                params = list(model.ode_model.parameters())
                grads = torch.autograd.grad(
                    loss, params, retain_graph=True, allow_unused=True
                )
                loss.backward()
                results.append((states.detach(), loss.detach(), grads))
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
            for a, b in zip(results[0][2], results[1][2]):
                if a is not None:
                    torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-8)
            self.assertTrue(all(p.grad is None for p in model.ml_model.parameters()))

    def test_shared_parameters_rebuilt_after_optimizer_update(self):
        _, model = self.hybrid(common.CONDITIONS[5])
        model.ode_model.target_chunk_size = 2
        start = torch.randn(3, 4)
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            fast = traj.integrate(
                model.ode_model,
                start,
                steps=4,
                dt=0.001,
                field_backend="shared_parameters_v1",
            )
            plain = traj.integrate(model.ode_model, start, steps=4, dt=0.001)
            torch.testing.assert_close(fast, plain, rtol=0, atol=0)
            fast.square().mean().backward()
            with torch.no_grad():
                for p in model.ode_model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-0.01)

    def test_canonical_conditions_and_legacy_separation(self):
        self.assertEqual(
            common.CONDITIONS[4:], [f"{f}_trajectory_ot_soft" for f in common.FAMILIES]
        )
        for name in common.CONDITIONS[4:]:
            c = common.effective_config(name)
            self.assertEqual(c["objective_schema_version"], traj.SCHEMA)
            self.assertEqual(
                c["trajectory_ot"]["trajectory_batch_size"]
                * c["trajectory_ot"]["samples_per_trajectory"],
                512,
            )
            self.assertEqual(c["trajectory_ot"]["real_target_points"], 512)
        for name in common.LEGACY_CONDITIONS:
            self.assertEqual(common.effective_config(name)["objective"], "ot")
        self.assertEqual(
            common.effective_config(common.CONDITIONS[1])["ot"].get(
                "epsilon_scaling", False
            ),
            False,
        )

    def test_legacy_checkpoint_rejected_even_if_renamed(self):
        current = common.effective_config(common.CONDITIONS[4])
        old = common.effective_config(common.LEGACY_CONDITIONS[0])
        old["condition"] = current["condition"]
        with self.assertRaisesRegex(ValueError, "incompatible objective"):
            recovery.check_objective(old, current)
        with self.assertRaisesRegex(ValueError, "incompatible objective"):
            recovery.resume_config(
                {"raw": {"metadata": {"effective_config": old}}}, current
            )
        raw = common.new_dir(self.path / "old") / "model005000.pt"
        cp.save_checkpoint(raw, self.stage1.state_dict(), {"effective_config": old})
        with self.assertRaisesRegex(ValueError, "incompatible objective"):
            recovery.load_bundle(raw, self.path, current["condition"], self.canonical)

    def test_integrator_shape_states_and_all_step_gradient(self):
        ode = LinearField()
        x = torch.ones(3, 4, dtype=torch.float64)
        states = traj.integrate(ode, x, steps=11, dt=0.1, checkpoint_block=3)
        self.assertEqual(tuple(states.shape), (3, 12, 4))
        torch.testing.assert_close(states[:, -1], x * (1 + 0.1 * ode.a) ** 11)
        states[:, -1].sum().backward()
        torch.testing.assert_close(
            ode.a.grad, torch.tensor(12 * 11 * 0.1 * (1.02) ** 10, dtype=torch.float64)
        )

    def test_checkpointed_and_plain_states_loss_gradients_all_families(self):
        for name in common.CONDITIONS[4:]:
            c, model = self.hybrid(name)
            source, real = torch.randn(3, 4), torch.randn(5, 4)
            results = []
            for block in (0, 2):
                model.zero_grad(set_to_none=True)
                c["trajectory_ot"]["checkpoint_block"] = block
                states = traj.integrate(
                    model.ode_model, source, steps=4, dt=0.001, checkpoint_block=block
                )
                loss, values, info, diagnostic = traj.trajectory_loss(
                    model, source, real, c, step=1
                )
                self.assertTrue(diagnostic)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                loss.backward()
                results.append(
                    (
                        states.detach(),
                        loss.detach(),
                        [
                            p.grad.clone() if p.grad is not None else None
                            for p in model.ode_model.parameters()
                        ],
                    )
                )
            torch.testing.assert_close(results[0][0], results[1][0])
            torch.testing.assert_close(results[0][1], results[1][1])
            for a, b in zip(results[0][2], results[1][2]):
                if a is not None:
                    torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-8)
            self.assertTrue(all(p.grad is None for p in model.ml_model.parameters()))

    def test_trajectory_loss_no_hybrid_or_cellunet_calls_and_frozen_ema(self):
        c, model = self.hybrid()
        before = common.state_hash(model.ml_model)
        opt = models.optimizer_for(model, c)
        ema = copy.deepcopy(model.state_dict())
        with (
            patch.object(model, "forward", side_effect=AssertionError("Hybrid called")),
            patch.object(
                model.ml_model, "forward", side_effect=AssertionError("ML called")
            ),
        ):
            loss, values, _, _ = traj.trajectory_loss(
                model, torch.randn(3, 4), torch.randn(5, 4), c, step=1
            )
            self.assertGreater(values["grad_ot"], 0)
            self.assertGreater(values["grad_soft"], 0)
            loss.backward()
            opt.step()
            models.update_ema(ema, model, 0.9999)
        models.assert_frozen(model, opt, before)
        self.assertEqual(
            common.state_hash(
                {
                    k.removeprefix("ml_model."): v
                    for k, v in ema.items()
                    if k.startswith("ml_model.")
                }
            ),
            before,
        )

    def test_gradient_diagnostics_preserve_accumulated_grads_and_frequency(self):
        c, model = self.hybrid()
        c["trajectory_ot"]["gradient_diagnostic_interval"] = 100
        params = [p for p in model.ode_model.parameters() if p.requires_grad]
        for p in params:
            p.grad = torch.ones_like(p)
        _, v, _, flag = traj.trajectory_loss(
            model, torch.randn(3, 4), torch.randn(5, 4), c, step=100
        )
        self.assertTrue(flag)
        self.assertGreater(v["grad_ratio_ot_to_soft"], 0)
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in params))
        with patch.object(
            traj, "gradient_norm", side_effect=AssertionError("unrequested diagnostics")
        ):
            _, v, _, flag = traj.trajectory_loss(
                model, torch.randn(3, 4), torch.randn(5, 4), c, step=2
            )
        self.assertFalse(flag)
        self.assertIsNone(v["grad_ot"])

    def test_stratified_default_exact_count_bins_and_new_step_choices(self):
        c = common.effective_config(common.CONDITIONS[4])["trajectory_ot"]
        x = torch.arange(128 * 101 * 2).reshape(128, 101, 2).float().requires_grad_()
        points, times = traj.occupation_sample(x, c, np.random.default_rng(7))
        self.assertEqual(tuple(points.shape), (512, 2))
        self.assertEqual(traj.temporal_edges(100, 4), [0, 25, 50, 75, 101])
        for k, (lo, hi) in enumerate(zip([0, 25, 50, 75], [25, 50, 75, 101])):
            self.assertTrue(np.all((times[:, k] >= lo) & (times[:, k] < hi)))
        _, other = traj.occupation_sample(x, c, np.random.default_rng(8))
        self.assertFalse(np.array_equal(times, other))
        points.sum().backward()
        self.assertEqual(int(torch.count_nonzero(x.grad)), 512 * 2)

    def test_independent_target_stream_sparse_and_without_replacement(self):
        from scipy.sparse import csr_matrix

        real = np.arange(1024 * 4).reshape(1024, 4)
        target, ids = traj.independent_batch(csr_matrix(real), 512, 1237, 1)
        self.assertEqual(tuple(target.shape), (512, 4))
        self.assertEqual(len(set(ids)), 512)
        _, source_ids = traj.independent_batch(real, 512, 1236, 1)
        self.assertFalse(np.array_equal(source_ids, ids))
        _, next_ids = traj.independent_batch(real, 512, 1237, 2)
        self.assertFalse(np.array_equal(next_ids, ids))
        _, same = traj.independent_batch(real, 512, 1237, 1)
        np.testing.assert_array_equal(ids, same)
        small, _ = traj.independent_batch(real[:3], 512, 1237, 1)
        self.assertEqual(len(small), 512)

    def test_cache_gaussian_start_exact_t50_semantics_and_no_overwrite(self):
        d = FakeDiffusion()
        out = cache.generate_cache(
            self.stage1,
            d,
            self.path / "cache",
            count=3,
            batch_size=3,
            genes=self.genes,
            canonical=self.canonical,
            seed=1235,
            role="train",
            device="cpu",
        )
        states, meta = cache.load_cache(out, self.canonical, role="train")
        self.assertEqual(d.calls, list(range(999, 50, -1)))
        generator = torch.Generator().manual_seed(1235)
        torch.testing.assert_close(d.first, torch.randn((3, 4), generator=generator))
        np.testing.assert_allclose(states, d.first.numpy() * 0.999**949, rtol=3e-5)
        self.assertEqual(meta["checkpoint_sha256"], "sha")
        self.assertEqual(meta["reverse_updates"], 949)
        self.assertEqual(meta["start_diffusion_t"], 50)
        before = common.file_hash(out / "x50.npy")
        with self.assertRaises(FileExistsError):
            cache.generate_cache(
                self.stage1,
                FakeDiffusion(),
                out,
                count=3,
                batch_size=3,
                genes=self.genes,
                canonical=self.canonical,
                seed=1,
                role="train",
                device="cpu",
            )
        self.assertEqual(before, common.file_hash(out / "x50.npy"))
        with self.assertRaisesRegex(ValueError, "provenance"):
            cache.load_cache(out, {**self.canonical, "checkpoint_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "provenance"):
            cache.load_cache(out, self.canonical, role="evaluation")

    def test_cache_rejects_noncanonical_model(self):
        with torch.no_grad():
            next(self.stage1.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, "canonical Stage1"):
            cache.generate_cache(
                self.stage1,
                FakeDiffusion(),
                self.path / "bad",
                count=3,
                batch_size=3,
                genes=self.genes,
                canonical=self.canonical,
                seed=1,
                role="train",
                device="cpu",
            )
        self.assertFalse((self.path / "bad").exists())

    def test_epsilon_schedule_exact_target_and_invalid(self):
        self.assertEqual(
            sink.epsilon_schedule(0.1, enabled=True), [1.6, 0.8, 0.4, 0.2, 0.1]
        )
        self.assertEqual(sink.epsilon_schedule(0.13, enabled=True)[-1], 0.13)
        self.assertEqual(sink.epsilon_schedule(0.1), [0.1])
        for kw in ({"factor": 1.0}, {"factor": 0}, {"start": 0.05}):
            with self.assertRaises(ValueError):
                sink.epsilon_schedule(0.1, enabled=True, **kw)

    def test_scaled_direct_sinkhorn_values_gradients_and_permutation(self):
        x = torch.randn(6, 3, dtype=torch.float64, requires_grad=True) * 0.3
        y = torch.randn(7, 3, dtype=torch.float64) * 0.3
        c = common.effective_config(common.CONDITIONS[4])["ot"]
        c["tolerance"] = 1e-9
        value, info = sink.sinkhorn_divergence(
            x, y, c, return_info=True, cost_diagnostics=True
        )
        direct = sink.sinkhorn_divergence(x, y, {**c, "epsilon_scaling": False})
        torch.testing.assert_close(value, direct, atol=1e-8, rtol=1e-6)
        a = torch.autograd.grad(value, x, retain_graph=True)[0]
        b = torch.autograd.grad(direct, x)[0]
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
        for xx, yy in ((x.flip(0), y), (x, y.flip(0))):
            torch.testing.assert_close(
                value, sink.sinkhorn_divergence(xx, yy, c), atol=1e-9, rtol=1e-7
            )
        self.assertEqual(info["cross"]["scales"][-1]["epsilon"], 0.1)
        self.assertTrue(
            all(np.isfinite(v) for v in info["cross"]["cost_statistics"].values())
        )
        self.assertTrue(
            all(
                v["marginal_residual"] <= c["tolerance"]
                for v in info["cross"]["scales"]
            )
        )

    def test_cost_statistics_reference_and_zero_cost(self):
        cost = torch.arange(10, dtype=torch.float64).reshape(2, 5)
        stats = sink.cost_statistics(cost, 0.1)
        self.assertEqual(stats["cost_mean"], 4.5)
        self.assertEqual(stats["cost_median"], 4.5)
        self.assertEqual(stats["cost_max_over_epsilon"], 90)
        self.assertAlmostEqual(stats["cost_q90"], 8.1)
        self.assertTrue(
            all(np.isfinite(v) for v in sink.cost_statistics(cost * 0, 0.1).values())
        )

    def test_sw_identical_shifted_deterministic_permutation_and_subsampling(self):
        x = np.random.default_rng(7).normal(size=(30, 4))
        for n in (30, 13):
            self.assertLess(
                sw.sliced_wasserstein(x, x, points=n)["sliced_wasserstein2"], 1e-12
            )
            a = sw.sliced_wasserstein(x, x + 1, points=n)
            self.assertGreater(a["sliced_wasserstein2"], 0)
            self.assertEqual(a, sw.sliced_wasserstein(x[::-1], (x + 1)[::-1], points=n))
            self.assertEqual(a, sw.sliced_wasserstein(x, x + 1, points=n))
            self.assertAlmostEqual(
                a["sliced_wasserstein2"] ** 2, a["sliced_wasserstein2_squared"]
            )
        self.assertEqual(sw.sliced_wasserstein(x, x[:7])["points_used"], 7)

    def test_occupation_evaluation_separate_endpoint_csv_small_cpu(self):
        c, model = self.hybrid()
        c["evaluation"].update(
            num_trajectories=3,
            ode_steps=4,
            sliced_wasserstein_points=12,
            sliced_wasserstein_projections=8,
            sinkhorn_points=5,
        )
        c["analysis_pairs"] = 30
        out = common.new_dir(self.path / "evaluation")
        provenance = dict(
            condition=c["condition"],
            checkpoint="/fixture/ema.pt",
            source_cache={"path": "/fixture/eval_cache"},
            training_config=c,
        )
        rows = occ.evaluate_occupation(
            model,
            np.random.default_rng(1).normal(size=(5, 4)).astype("float32"),
            np.random.default_rng(2).normal(size=(15, 4)).astype("float32"),
            c,
            out,
            provenance=provenance,
            device="cpu",
            batch_size=2,
        )
        self.assertEqual([r["distribution"] for r in rows], ["occupation", "endpoint"])
        self.assertEqual(rows[0]["occupation_points_total"], 15)
        self.assertEqual(rows[0]["occupation_points_used"], 12)
        self.assertEqual(rows[0]["sinkhorn_points_used"], 5)
        self.assertEqual(rows[1]["occupation_points_used"], 0)
        self.assertEqual(rows[1]["generated_points_used"], 3)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue((out / "occupation_analysis.csv").exists())
        self.assertFalse(list(out.glob("*.png")))

    def test_synthetic_runner_cache_reuse_logging_and_compatible_resume(self):
        config_dir = common.new_dir(self.path / "configs")
        (config_dir / "trajectory_defaults.json").write_text(
            (common.SUITE / "configs/trajectory_defaults.json").read_text()
        )
        campaign = common.new_dir(self.path / "runs" / "synthetic")
        first = (
            common.new_dir(campaign / "stage1_cellunet" / "first" / "checkpoints")
            / "ema.pt"
        )
        c1 = self.small("stage1_cellunet")
        cp.save_checkpoint(
            first,
            self.stage1.state_dict(),
            dict(
                effective_config=c1,
                checkpoint_kind="ema",
                step=2,
                gene_names=self.genes,
            ),
        )
        canonical = {
            **self.canonical,
            "checkpoint": str(first),
            "checkpoint_sha256": common.file_hash(first),
        }
        common.write_json(campaign / "canonical_stage1.json", canonical)
        cache_args = cli.parser("cache_x50").parse_args(
            [
                "--campaign",
                "synthetic",
                "--cache-size",
                "6",
                "--batch-size",
                "3",
                "--device",
                "cpu",
            ]
        )
        with (
            patch.object(cache, "SUITE", self.path),
            patch.object(cache, "build_diffusion", return_value=FakeDiffusion()),
        ):
            source = cache.cache_command(cache_args)
            with patch.object(
                cache, "generate_cache", side_effect=AssertionError("cache regenerated")
            ):
                self.assertEqual(str(cache.cache_command(cache_args)), str(source))
        before = common.file_hash(source / "x50.npy")
        c = self.small(common.CONDITIONS[4])
        real = SimpleNamespace(
            X=np.random.default_rng(2).normal(size=(8, 4)).astype("float32"), n_obs=8
        )
        args = cli.parser("train").parse_args(
            [
                "--campaign",
                "synthetic",
                "--condition",
                c["condition"],
                "--device",
                "cpu",
            ]
        )
        with (
            patch.object(runner, "SUITE", self.path),
            patch.object(runner, "effective_config", return_value=copy.deepcopy(c)),
            patch.object(runner, "load_real", return_value=(real, self.genes, {})),
            patch.object(
                runner,
                "umap_core",
                return_value=SimpleNamespace(gene_order_hash=lambda _: "genes"),
            ),
        ):
            args.trajectory_save_interval = 1
            args.trajectory_log_interval = 1
            run = runner.train(args)
            with (run / "losses.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["grad_ot"] and row["cost_mean"] for row in rows))
            raw = run / "checkpoints/model000001.pt"
            bundle = recovery.load_bundle(raw, campaign, c["condition"], canonical)
            self.assertEqual(
                bundle["raw"]["metadata"]["objective_schema_version"], traj.SCHEMA
            )
            self.assertEqual(
                bundle["raw"]["metadata"]["trajectory_source_cache"]["path"],
                str(source),
            )
            old_hashes = {
                str(f): common.file_hash(f) for f in run.rglob("*") if f.is_file()
            }
            args.resume_checkpoint = str(raw)
            resumed = runner.train(args)
            self.assertNotEqual(run, resumed)
            with (resumed / "losses.csv").open() as f:
                self.assertEqual([row["step"] for row in csv.DictReader(f)], ["2"])
            self.assertEqual(
                old_hashes,
                {str(f): common.file_hash(f) for f in run.rglob("*") if f.is_file()},
            )
        self.assertEqual(before, common.file_hash(source / "x50.npy"))


if __name__ == "__main__":
    unittest.main()
