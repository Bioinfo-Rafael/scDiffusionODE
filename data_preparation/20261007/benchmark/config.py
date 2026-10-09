"""Fixed Data 3 protocol; sources and deliberate deviations are in README.md."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data' / 'benchmark'
SOURCE = HERE.parent / 'data' / 'MouseGastrulation.h5ad'
EVAL_PYTHON = DATA / '.venv-eval' / 'bin' / 'python'
VENDOR = HERE / 'vendor' / 'VeloEV'
VELOEV_COMMIT = '719aafa3aebe488bf557c484ca700b78a0b90c95'
BENCHMARK_COMMIT = 'b8bc1312e8dc54ab5347159f447707bc06404ad2'
PYTHON_VERSION = '3.12.14'
FULL_CELLS = 89267
ERYTHROID_CELLS = 9815
ERYTHROID_CELLTYPES = (
    'Blood progenitors 1', 'Blood progenitors 2',
    'Erythroid1', 'Erythroid2', 'Erythroid3',
)
# Supplementary Note S10, Data 3, p. 8. Dataset labels omit the space before 1/2/3.
CELL_TYPE_TRANSITIONS = tuple(zip(ERYTHROID_CELLTYPES[:-1], ERYTHROID_CELLTYPES[1:]))
STAGE_TO_DAY = {
    'E6.5': 6.5, 'E6.75': 6.75, 'E7.0': 7.0, 'E7.25': 7.25,
    'E7.5': 7.5, 'E7.75': 7.75, 'E8.0': 8.0, 'E8.25': 8.25, 'E8.5': 8.5,
}
# Paper Eq. (3): adjacent observed developmental time groups in ascending order.
# The erythroid subset may lack early stages: use observed groups, record actual pairs.
SEED = 1234
N_PCS = 30
N_NEIGHBORS = 30
N_HVG = 2000
EXPRESSION_SCALE = 'spliced_independent_normalize_total_1e4'
PROTOCOL = 'mouse_gastrulation_normalized_full_train_ery_eval_v1'


def subprocess_env():
    """Fixed environment for scientific workers, with no inherited Python packages."""
    import os
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS'):
        env[key] = '1'
    env.update(PYTHONHASHSEED=str(SEED), MPLBACKEND='Agg', PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    return env


def launch_isolated(script):
    """Standard-library-only entry point, safe from a training environment."""
    import subprocess
    import sys
    if '--_worker' in sys.argv:
        sys.argv.remove('--_worker')
        return
    if not EVAL_PYTHON.is_file():
        raise SystemExit(f'Run bash {HERE / "setup_env.sh"} first')
    raise SystemExit(subprocess.call([str(EVAL_PYTHON), str(Path(script).resolve()),
                                     '--_worker', *sys.argv[1:]], env=subprocess_env()))
