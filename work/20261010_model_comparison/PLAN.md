# Implementation and execution phases

1. Repository audit: enumerate all 34 historical work directories, ODE, guided_diffusion and 10/07 benchmark sources; record classes, equations, actual loss/update settings, source hashes and existing local changes.
2. Shared infrastructure: reuse exact 10/09 HVG preparation, validate shared IDs/linear scale, map Mouse GRN IDs, validate final Stage1 bundle, wire native DDPM and official normalized full-trained three-fold benchmark.
3. Pilot: A01 + B01–04 + C01–04 + D01–04 + E01–04; separate families, shared Stage1, observable loss/gradients and frozen-state hashes.
4. Extensions: compatible source fields/gates/compositions, EPSILON/START_X OT, same-initialization freeze controls. Registry records unsupported combinations and required adapters.
5. Validation: source identity, synthetic forward/backward, algebra and gradient invariants, full runner resume equivalence, ID-aligned exports, sampler, benchmark CLI, visualization/summary contracts and dry-runs. Large real-data/GPU runs are deliberately left for explicit remote execution.

See `audit/test_results.txt` and `audit/VALIDATION_REPORT.md` for actual outcomes. A planned step or generated command is not evidence that biological evaluation succeeded.
