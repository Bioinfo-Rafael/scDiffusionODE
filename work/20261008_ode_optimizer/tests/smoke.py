#!/usr/bin/env python3
"""Run the small CPU-only regression and full four-condition visualization smoke."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['ODE_OPTIMIZER_SMOKE']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['NUMBA_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import unittest
from test_suite import CoreTests, PipelineSmoke
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (CoreTests,PipelineSmoke)])
result=unittest.TextTestRunner(verbosity=2).run(suite)
from common import SUITE, write_json
write_json(SUITE/'validation/smoke_result.json',dict(success=result.wasSuccessful(),
    tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
    skipped=len(result.skipped),device='cpu',training_updates_per_condition=3,
    reverse_updates_per_trajectory=1000))
raise SystemExit(0 if result.wasSuccessful() else 1)
