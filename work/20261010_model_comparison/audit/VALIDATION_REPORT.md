# Validation report

- Local test suite: 64 passed, 4 warnings in 18.82 seconds; see `test_results.txt`.
- Coverage includes synthetic training/checkpoint resume, model adapters, source losses, sampling, velocity export, benchmark contracts, and visualization fixtures.
- Four warnings remain: three NumPy matmul warnings from the original sliced-Wasserstein implementation on synthetic inputs, and one AnnData string-index conversion warning. Tests verify finite outputs.
- Historical tracked source protection: 1,030 files checked against the initial workspace baseline; no changes.
- Real MouseGastrulation data, production GPU training, and actual benchmark scores were not available locally and have not been validated. Synthetic benchmark artifacts validate integration and plotting, not scientific results.
- The isolated local test environment is under ignored `runs/`; existing training and benchmark environments were not modified.
- Remote clean main may differ from the initial local protected baseline because that baseline includes pre-existing user edits. Capture a remote baseline with `scripts/verify_sources.py --capture` under `runs/` before execution when checking remote source preservation.
