"""Combined Erythropoietic/Immune palette using the historical lineage table."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
HISTORICAL_PALETTE_SOURCE = (
    REPO_ROOT
    / "work/20260215_embryonic/20260224_084326_Lamda5/velocity_by_Superclass.py"
)


def _historical_lineages():
    """Load the exact past LINEAGES table without importing by package name."""

    spec = importlib.util.spec_from_file_location(
        "historical_velocity_by_superclass_20260224", HISTORICAL_PALETTE_SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load historical palette: {HISTORICAL_PALETTE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LINEAGES


def _gradient_hex(cmap_name: str, count: int, lo: float = 0.25, hi: float = 0.9):
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(value)) for value in np.linspace(lo, hi, max(count, 1))]


def apply_combined_hematopoietic_palette(
    adata,
    *,
    selected_superclasses=("Erythropoietic", "Immune"),
    celltype_column: str = "celltype",
) -> dict[str, str]:
    """Apply the historical lineage order/colors, deduplicated across superclasses."""

    if celltype_column not in adata.obs:
        raise KeyError(f"missing celltype column: {celltype_column}")
    lineages = _historical_lineages()
    adata.obs[celltype_column] = adata.obs[celltype_column].astype("category")
    adata.obs[celltype_column] = adata.obs[celltype_column].cat.remove_unused_categories()
    present = [str(value) for value in adata.obs[celltype_column].cat.categories]
    mapping: dict[str, str] = {}
    ordered: list[str] = []
    for superclass in selected_superclasses:
        for _lineage_name, sequence, cmap in lineages.get(str(superclass), []):
            selected = [name for name in sequence if name in present and name not in ordered]
            for name, color in zip(selected, _gradient_hex(cmap, len(selected))):
                mapping[name] = color
            ordered.extend(selected)
    leftovers = [name for name in present if name not in ordered]
    fallback = plt.get_cmap("tab20").colors
    for index, name in enumerate(sorted(leftovers)):
        mapping[name] = mcolors.to_hex(fallback[index % len(fallback)])
    category_order = ordered + sorted(leftovers)
    adata.obs[celltype_column] = adata.obs[celltype_column].cat.reorder_categories(
        category_order, ordered=True
    )
    adata.uns[f"{celltype_column}_colors"] = np.asarray(
        [mapping[name] for name in category_order], dtype=str
    )
    return mapping


__all__ = [
    "HISTORICAL_PALETTE_SOURCE",
    "apply_combined_hematopoietic_palette",
]
