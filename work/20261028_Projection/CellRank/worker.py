"""Official CellRank worker: arrays and IDs only; no model/checkpoint or required Torch."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from CellRank.subprocess_bridge import SCHEMA, digest, run_in_process, runtime_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        from CellRank import cellrank_core  # Verify real imports, not only installed metadata.
        print(json.dumps(runtime_info()))
        return
    if args.directory is None: parser.error('Request directory required')
    import numpy as np
    from scipy import sparse
    directory = args.directory
    request = json.loads((directory/'request.json').read_text())
    if request['schema'] != SCHEMA or digest(directory/'input.npz') != request['input_sha256']:
        raise ValueError('CellRank worker input integrity failure')
    with np.load(directory/'input.npz', allow_pickle=False) as saved:
        x, v = saved['X'], saved['V']
    result = run_in_process(x, v, request['cell_ids'], request['gene_ids'],
                            request['terminal_cells'], request['settings'])
    np.savez_compressed(directory/'result.npz', times=result['times'], probabilities=result['probabilities'],
                        cell_ids=np.asarray(result['cell_ids'], dtype=str), gene_ids=np.asarray(result['gene_ids'], dtype=str))
    sparse.save_npz(directory/'transition.npz', result['transition'])
    meta = {key: result[key] for key in ('time_columns', 'probability_columns', 'transition_checks', 'absorption_checks', 'runtime')}
    meta.update(schema=SCHEMA, request_sha256=digest(directory/'request.json'),
                files_sha256={name: digest(directory/name) for name in ('result.npz', 'transition.npz')})
    (directory/'response.json').write_text(json.dumps(meta, allow_nan=False, indent=2)+'\n')


if __name__ == '__main__': main()
