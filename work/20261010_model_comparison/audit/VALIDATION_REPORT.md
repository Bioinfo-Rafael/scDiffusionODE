# Validation report

- Local test suite: 70 passed, 4 warnings in 22.82 seconds; see `test_results.txt`.
- D ODE-SGD control: all four resolved configurations differ from their original D configuration only by `ode_optimizer=SGD`. Tests verify exact branch updates against separate PyTorch AdamW/SGD, dual optimizer serialization, lambda=0 and lambda=0.1 bitwise raw/EMA equivalence after resume, and velocity exports. Python 3.9 launcher dry-run passed. No production GPU training was run locally.
- 2026-10-12 checkpoint relocation fix: when a latest pointer references a missing path, try only the same filename beside that pointer. Hash validation remains enforced; tests cover relocated pointers with and without hashes, corruption, absent files, and explicit-path behavior. Remote checkpoint contents remain unverified locally.
- 2026-10-12 portability fix: runtime provenance hashes only Git-tracked historical inventory entries. A regression test checks identical hashes with local untracked tooling present or absent, and continued failure for a missing tracked dependency.
- Coverage includes synthetic training/checkpoint resume, model adapters, source losses, sampling, velocity export, benchmark contracts, and visualization fixtures.
- Four warnings remain: three NumPy matmul warnings from the original sliced-Wasserstein implementation on synthetic inputs, and one AnnData string-index conversion warning. Tests verify finite outputs.
- Historical tracked source protection: 1,030 files checked against the initial workspace baseline; no changes.
- Real MouseGastrulation data, production GPU training, and actual benchmark scores were not available locally and have not been validated. Synthetic benchmark artifacts validate integration and plotting, not scientific results.
- The isolated local test environment is under ignored `runs/`; existing training and benchmark environments were not modified.
- Remote clean main may differ from the initial local protected baseline because that baseline includes pre-existing user edits. Capture a remote baseline with `scripts/verify_sources.py --capture` under `runs/` before execution when checking remote source preservation.
