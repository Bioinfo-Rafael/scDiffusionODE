"""Small output helpers; no model or statistical decisions live here."""
import hashlib
import json
from pathlib import Path
import re


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def group_folder(index, label):
    return f'{index:02d}_' + re.sub(r'[^A-Za-z0-9_-]+', '_', str(label)).strip('_')


def groups(labels):
    import numpy as np
    yield 'all', 'ALL', np.arange(len(labels))
    for i, label in enumerate(sorted(set(labels))):
        yield group_folder(i, label), label, np.flatnonzero(labels == label)


def log(message):
    from datetime import datetime, timezone, timedelta
    print(f'[{datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")}] {message}', flush=True)
