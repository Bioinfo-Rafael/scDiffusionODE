"""Download only. Run this script on the remote machine."""
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import shutil
import zipfile

import anndata as ad
import requests

RECORD = '18051944'
ARCHIVE = '03_Gastrulation_erythroid.h5ad.zip'


def benchmark(data_dir):
    target = data_dir / ARCHIVE.removesuffix('.zip')
    if target.exists():
        return target
    response = requests.get(f'https://zenodo.org/api/records/{RECORD}', timeout=120)
    response.raise_for_status()
    record = response.json()
    matches = [f for f in record['files'] if f['key'] == ARCHIVE]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one {ARCHIVE}: found {len(matches)}')
    info = matches[0]
    archive = data_dir / ARCHIVE
    partial = archive.with_suffix(archive.suffix + '.part')
    algorithm, expected = info['checksum'].split(':', 1)
    digest = hashlib.new(algorithm)
    with requests.get(info['links']['self'], stream=True, timeout=(60, 300)) as r:
        r.raise_for_status()
        with partial.open('wb') as out:
            for chunk in r.iter_content(8 * 1024 * 1024):
                out.write(chunk)
                digest.update(chunk)
    if digest.hexdigest() != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError('Zenodo checksum mismatch; rerun download')
    partial.replace(archive)
    temp = target.with_suffix('.h5ad.part')
    with zipfile.ZipFile(archive) as z:
        members = [i for i in z.infolist() if not i.is_dir()
                   and Path(i.filename).name == target.name
                   and '__MACOSX' not in i.filename.split('/')]
        if len(members) != 1:
            raise RuntimeError('Archive does not contain exactly one expected h5ad')
        # Copy just the selected member; never extract arbitrary archive paths.
        with z.open(members[0]) as src, temp.open('wb') as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
    temp.replace(target)
    (data_dir / 'zenodo_record.json').write_text(json.dumps(record, indent=2))
    return target


def full_dataset(data_dir):
    target = data_dir / 'gastrulation_full.h5ad'
    if target.exists():
        return target
    import scvelo as scv
    cache = data_dir / 'scvelo_cache'
    cache.mkdir(exist_ok=True)
    # file_path changes storage location only. No additional preprocessing.
    full = scv.datasets.gastrulation(file_path=str(cache / 'gastrulation.h5ad'))
    temp = target.with_suffix('.part.h5ad')
    # Preserve string columns instead of AnnData writer's implicit categorization.
    full.write_h5ad(temp, convert_strings_to_categoricals=False)
    temp.replace(target)
    (data_dir / 'full_source.json').write_text(json.dumps({
        'loader': 'scvelo.datasets.gastrulation',
        'scvelo_version': importlib.metadata.version('scvelo'),
        'note': 'Saved returned object; loader may make var_names unique.',
    }, indent=2))
    del full
    gc.collect()
    return target


def report(path):
    data = ad.read_h5ad(path, backed='r')
    try:
        print(json.dumps({'file_path': str(path), 'file_size_bytes': path.stat().st_size,
                          'shape': data.shape, 'n_obs': data.n_obs, 'n_vars': data.n_vars}))
    finally:
        data.file.close()


def main():
    data_dir = Path(__file__).resolve().parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    report(benchmark(data_dir))
    report(full_dataset(data_dir))


if __name__ == '__main__':
    main()
