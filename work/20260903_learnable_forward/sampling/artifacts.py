"""Validated, provenance-complete sample archive I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.common import gene_order_sha256, write_json


REQUIRED_SAMPLE_METADATA = (
    "checkpoint_path",
    "checkpoint_step",
    "ema_rate",
    "forward_model",
    "random_seed",
    "sampler_type",
    "reverse_steps",
    "decoder_sampling_mode",
)


def save_sample_archive(
    path: str | Path,
    cell_gen,
    genes: Sequence[str],
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Atomically save numerical samples, ordered genes, and scalar provenance."""

    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("sample archive path must end in .npz")
    cells = np.asarray(cell_gen, dtype=np.float32)
    ordered_genes = [str(gene) for gene in genes]
    if cells.ndim != 2 or cells.shape[1] != len(ordered_genes):
        raise ValueError("cell_gen and ordered gene names have incompatible shapes")
    if not np.isfinite(cells).all() or len(set(ordered_genes)) != len(ordered_genes):
        raise ValueError("samples must be finite and ordered genes unique")
    missing = [name for name in REQUIRED_SAMPLE_METADATA if name not in metadata]
    if missing:
        raise KeyError("sample metadata is missing: " + ", ".join(missing))
    payload = dict(metadata)
    payload.update(
        {
            "sample_path": str(destination.resolve()),
            "sample_count": int(cells.shape[0]),
            "dimension": int(cells.shape[1]),
            "ordered_gene_names": ordered_genes,
            "gene_order_sha256": gene_order_sha256(ordered_genes),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        cell_gen=cells,
        ordered_gene_names=np.asarray(ordered_genes, dtype=str),
        gene_order_sha256=np.asarray(payload["gene_order_sha256"]),
        checkpoint_path=np.asarray(str(payload["checkpoint_path"])),
        checkpoint_step=np.asarray(int(payload["checkpoint_step"]), dtype=np.int64),
        ema_rate=np.asarray(str(payload["ema_rate"])),
        forward_model=np.asarray(str(payload["forward_model"])),
        random_seed=np.asarray(int(payload["random_seed"]), dtype=np.int64),
        sampler_type=np.asarray(str(payload["sampler_type"])),
        reverse_steps=np.asarray(int(payload["reverse_steps"]), dtype=np.int64),
        decoder_sampling_mode=np.asarray(str(payload["decoder_sampling_mode"])),
    )
    temporary.replace(destination)
    sidecar = destination.with_suffix(".json")
    write_json(sidecar, payload)
    return destination, sidecar


def load_sample_archive(path: str | Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    source = Path(path)
    sidecar = source.with_suffix(".json")
    if not source.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"sample archive or sidecar is missing: {source}")
    with np.load(source, allow_pickle=False) as archive:
        cells = np.asarray(archive["cell_gen"], dtype=np.float32)
        genes = [str(value) for value in archive["ordered_gene_names"].tolist()]
        embedded_hash = str(archive["gene_order_sha256"].item())
    with sidecar.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_hash = gene_order_sha256(genes)
    if embedded_hash != expected_hash or metadata.get("gene_order_sha256") != expected_hash:
        raise ValueError("sample gene-order hash mismatch")
    if cells.ndim != 2 or cells.shape[1] != len(genes) or not np.isfinite(cells).all():
        raise ValueError("invalid cell_gen payload")
    return cells, genes, metadata


__all__ = [
    "REQUIRED_SAMPLE_METADATA",
    "load_sample_archive",
    "save_sample_archive",
]
