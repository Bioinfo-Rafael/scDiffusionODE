"""Benchmark eligibility gate: standalone START_X cannot satisfy ds/dt."""
from common import *
import csv
import subprocess

REASON = ('CellUNet Stage 1 predicts x_start under an unconditional denoising MSE. '
          'Neither that prediction nor its reconstruction displacement defines biological ds/dt. '
          'Diffusion timestep is not developmental time. No justified conversion is available.')


def evaluate(c, *, prepare_reference=False):
    run = paths(c)
    report = dict(status='not_applicable', benchmark_executed=False, reason=REASON,
        model='CellUNet Stage 1 START_X', metrics={k: None for k in ('CBDir', 'ICVCoh', 'CTO', 'TSC')},
        benchmark_entrypoint=str(BENCH.HERE / 'run.py'), evaluation_python=str(BENCH.EVAL_PYTHON),
        benchmark_reference=str(BENCH.DATA / 'erythroid.h5ad'),
        benchmark_results_root=str(BENCH.DATA / 'results'), reference_requested=prepare_reference)
    if prepare_reference:
        # Existing entry point enforces its own venv and refuses overwrite.
        command = [sys.executable, str(BENCH.HERE / 'prepare.py')]
        with (run / 'logs/benchmark_prepare.log').open('a') as log:
            try:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            except Exception as exc:
                report.update(status='failed', reference_error=str(exc))
                write_json(run / 'metrics/benchmark_status.json', report)
                raise
    write_json(run / 'metrics/benchmark_status.json', report)
    with (run / 'metrics/applicability.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric', 'value', 'status', 'reason'])
        for metric in report['metrics']:
            w.writerow([metric, '', 'not_applicable', REASON])
    (run / 'logs/benchmark.log').write_text('NOT_APPLICABLE: ' + REASON + '\nNo VeloEV run was launched.\n')
    print('NOT_APPLICABLE:', REASON)
    return 2


def main():
    p = parser(__doc__)
    p.add_argument('--prepare-reference', action='store_true', help='Optional existing isolated benchmark prepare.py; refuses existing reference')
    args = p.parse_args()
    raise SystemExit(evaluate(config(args.config), prepare_reference=args.prepare_reference))

if __name__ == '__main__':
    cli(main)
