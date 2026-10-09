"""Evaluate the stipulated CellUNet velocity with the unchanged VeloEV entry point."""
from common import *
import csv
import shutil
import subprocess


def evaluate(c, *, method=None, prepare_reference=False):
    run = paths(c)
    t_index = c['field_timestep']
    stem = f'cellunet_direct_t{t_index}'
    prediction = run / 'predictions' / f'{stem}.h5ad'
    field_meta = json.loads((run / 'predictions' / f'{stem}_metadata.json').read_text())
    require(prediction.is_file() and sha256(prediction) == field_meta['output_sha256'],
            'direct velocity prediction missing or changed')
    require(field_meta['definition'] == 'ds_dt := CellUNet(clean_X, diffusion_t)'
            and field_meta['timestep'] == t_index,
            'benchmark input is not the stipulated direct field')
    checkpoint = resolve_checkpoint(c)
    require(sha256(checkpoint) == field_meta['checkpoint_sha256'],
            'prediction differs from current checkpoint')
    latest = json.loads((run / 'checkpoints/latest.json').read_text())
    training_meta = json.loads((run / 'data/metadata.json').read_text())
    require(field_meta['training_sha256'] == training_meta['training_sha256'],
            'prediction differs from training data')
    genes = run / 'data/gene_ids.txt'
    require(ids(genes.read_text().splitlines()) == training_meta['gene_ids'],
            'benchmark gene manifest differs from training data')
    require(digest_ids(training_meta['gene_ids']) == field_meta['gene_ids_sha256']
            and digest_ids(training_meta['cell_ids']) == field_meta['cell_ids_sha256'],
            'prediction ID manifest differs from training data')
    method = method or f'cellunet_stage1_direct_t{t_index}_{field_meta["checkpoint_sha256"][:12]}'
    require(method and all(ch.isalnum() or ch in '_.-' for ch in method), 'invalid method name')
    reference = BENCH.DATA / 'erythroid.h5ad'
    result = BENCH.DATA / 'results' / method
    local_status = run / 'metrics' / f'{stem}_benchmark_status.json'
    report = dict(status='running', method=method, prediction=str(prediction),
        prediction_sha256=field_meta['output_sha256'], reference=str(reference),
        result=str(result), checkpoint=str(checkpoint), checkpoint_step=latest['step'],
        velocity_definition=field_meta['definition'], diffusion_timestep=t_index,
        model_time_direction='forward by experimental definition',
        time_source='VeloEV velocity_pseudotime fallback; no model time supplied',
        benchmark_entrypoint=str(BENCH.HERE / 'run.py'),
        reference_requested=prepare_reference)
    log_path = run / 'logs' / f'{stem}_benchmark.log'
    try:
        if not reference.is_file():
            require(not reference.exists(), 'benchmark reference already exists; do not overwrite')
            with log_path.open('a') as log:
                subprocess.run([sys.executable, str(BENCH.HERE / 'prepare.py')],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        require(reference.is_file(), 'benchmark reference was not created')
        if result.exists():
            previous = json.loads((result / 'metadata.json').read_text())
            require(previous['status'] == 'complete'
                    and previous['inputs']['prediction']['sha256'] == field_meta['output_sha256']
                    and previous['inputs']['reference']['sha256'] == sha256(reference),
                    'existing benchmark result does not match this prediction/reference; use --method')
            require((result / 'metrics.csv').is_file(), 'completed benchmark has no metrics.csv')
        else:
            command = [sys.executable, str(BENCH.HERE / 'run.py'),
                       '--prediction', str(prediction), '--genes', str(genes), '--method', method]
            with log_path.open('a') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        completed = json.loads((result / 'metadata.json').read_text())
        require(completed['status'] == 'complete', 'VeloEV did not complete')
        local_metrics = run / 'metrics' / f'{stem}_metrics.csv'
        if local_metrics.exists():
            require(sha256(local_metrics) == sha256(result / 'metrics.csv'),
                    'local metrics differ from completed benchmark')
        else:
            shutil.copyfile(result / 'metrics.csv', local_metrics)
        with local_metrics.open(newline='') as f:
            scores = list(csv.DictReader(f))
        require(len(scores) == 1 and all(k in scores[0] for k in ('cbdir', 'icvcoh', 'cto', 'tsc')),
                'four benchmark metrics missing')
        report.update(status='complete', metrics=scores[0], metrics_file=str(local_metrics),
                      benchmark_metadata=str(result / 'metadata.json'), benchmark_log=str(result / 'run.log'))
        write_json(local_status, report)
        print(f'VeloEV complete: {local_metrics}')
        return 0
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', log=str(log_path))
        write_json(local_status, report)
        raise


def main():
    p = parser(__doc__)
    p.add_argument('--method', help='VeloEV result directory name; use a fresh name to retry a failed run')
    p.add_argument('--prepare-reference', action='store_true', help='Prepare reference if absent; existing reference is never overwritten')
    args = p.parse_args()
    raise SystemExit(evaluate(config(args.config), method=args.method,
                              prepare_reference=args.prepare_reference))


if __name__ == '__main__':
    cli(main)
