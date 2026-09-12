"""Small result-file helpers; completed trajectory validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .settings import Settings
from .source_imports import WORK


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if WORK.resolve() not in path.parents:
        raise ValueError(f"outputs must be below {WORK}: {path}")
    return path


def load_trajectory(result: Path) -> tuple[np.ndarray, list[dict], dict, Settings]:
    from .trajectory import snapshot_table

    result = result_path(result)
    meta = json.loads((result / "metadata.json").read_text())
    sampling = json.loads((result / "trajectory/sampling_metadata.json").read_text())
    if sampling.get("status") != "completed":
        raise ValueError(f"sampling is incomplete: {result}")
    settings = Settings(**meta["settings"])
    with (result / "trajectory/snapshot_table.csv").open(newline="") as handle:
        table = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    expected = snapshot_table(settings, sampling["timestep_map"])
    if table != expected:
        raise ValueError("snapshot table does not match recorded diffusion schedule")
    array = np.load(result / "trajectory/pred_xstart.npy", mmap_mode="r", allow_pickle=False)
    shape = (settings.trajectories, settings.M, meta["gene_count"])
    if array.shape != shape or array.dtype != np.float32:
        raise ValueError(f"expected float32 trajectory {shape}, got {array.shape}, {array.dtype}")
    return array, table, meta, settings
