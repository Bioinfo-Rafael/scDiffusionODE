# Verification record — 2026-09-12 JST

Environment: macOS, Python 3.9 (`scdiffusion` conda environment), PyTorch 2.5.1,
NumPy 1.24.0, Scanpy 1.9.3. All computations below ran on CPU.

## Automated tests

```bash
python -m unittest discover -s work/20260911/tests -v
```

**16 tests passed, 6.253 seconds** in the final functional test run.

| Coverage | Result |
| --- | --- |
| 20 snapshots, increasing completed reverse steps, final `(1000,0)` | Passed |
| Actual 1000-step progressive sampler vs ordinary `p_sample_loop`, tiny denoiser, 2 trajectories × 3 genes | Saved final sample exactly equal; final pred_xstart allclose |
| Batch boundaries and stable trajectory IDs, including partial final batch | Passed, 6 trajectories split 4+2 |
| Incomplete generator | Rejected and `status=failed` recorded |
| `[2000,20,3] → [20,100,20,3]`, order and Eq. 7 feature mean | Passed |
| Nonfinite clean predictions | Rejected |
| Eq. 8 synthetic discontinuous two-line curve | Recovered snapshot 8 with SSE < 1e-20; confirms boundary inclusion |
| 10% endpoint exclusion | Exactly 16 candidates, indices 2..17 |
| Even-group median | Index 7.5 retained, step 425 / t575, not a saved snapshot |
| Stage 1 fit isolation | 20 separate AnnData objects and 20 fit calls, one snapshot each |
| Stage 2 explicit selection | Only `[7,9,3]` concatenated in requested order; one fit; missing/duplicate/invalid selections rejected |
| Source restore fixture, 20260803 | `standard_hybrid_lincomb__hill_after_linear`, real source classes, 5 genes, wrapped EMA, strict restore and forward equality |
| Source restore fixture, 20260816 | `linear_centered_signed_hill`, real source classes, 5 genes, wrapped EMA, strict restore and forward equality |
| Source fixture immutability | Hashes of every fixture input unchanged after restore, CLI restore-only and metadata assembly |
| Module collision | Loading the other model suite in the same process rejected |
| Coordinator | Missing/ambiguous runs start no jobs; preflight precedes sampling; dry-run/restore-only modes checked |
| CLI | All six commands accept `--help`; Stage 2 requires explicit selection |

The EMA fixture files are generated from randomly initialized real source model
classes with a small deterministic parameter perturbation, **without training**.
They are not the user's trained 30,000-step artifacts. Both source `runs/` trees
are absent in this local checkout; real checkpoint restore was therefore **not
performed**. Use `run_all.py --restore-only --device cpu` in the remote environment
for that check. Source run-root constants are redirected only within fixture test
processes; no files are created in the source suites.

## Real Scanpy Stage 1 smoke

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python work/20260911/tests/smoke_pipeline.py \
  --output work/20260911/results/smoke_20260912
```

Passed: **20 real independent PCA/neighbors/UMAP fits** and 20 snapshot PNG/CSV
pairs, contact sheet, breakpoint CSV/JSON, two group-fit PNGs and summary PNG.
Fixture: 48 synthetic reference cells × 12 genes; only the 40 Erythropoietic cells
were selected, excluding 8 Immune cells; 10 generated rows per snapshot; N=2/S=5.
Existing UMAP dimension caps applied. Final snapshot UMAP and group-fit plot were
visually inspected: titles, snapshot/progress/timestep, legend and axes render
correctly. Outputs are under ignored `results/` and are not committed.

Stage 2 was only tested using an injected fit spy; no Stage 2 UMAP was executed.
No trained-model GPU sampling, production 2000×1000 run, or training was executed.

## Static / repository checks

- Ruff check and format check passed for all 19 Python files.
- Python compilation passed.
- `git diff --check` passed.
- New tracked files are confined to `work/20260911/`.
- Existing changes under `work/20260801/hybrid_ts_soft_weight_curve/` are unrelated
  user work and are excluded from this commit.
- `results/.gitignore` excludes heavy arrays, figures, logs, fixtures and local
  development tooling; source checkpoints/data remain covered by repo policy.
