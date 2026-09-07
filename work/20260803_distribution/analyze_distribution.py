#!/usr/bin/env python3
"""Compare real/generated value distributions by real-expression gene quintile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import ks_2samp, wasserstein_distance


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_SOURCE_ROOT = REPO_ROOT / "work" / "20260803_ODE_hill_exp"
OUTPUT_ROOT = HERE / "outputs"
FAMILIES = ("ode_only_lincomb", "standard_hybrid_lincomb")
ODE_TYPES = ("hill_after_linear", "racipe", "exp")
EXPERIMENTS = tuple(f"{family}__{ode}" for family in FAMILIES for ode in ODE_TYPES)
SAMPLE_KEYS = ("cell_gen", "samples", "generated", "x")
EMA_RE = re.compile(r"^ema_(?P<rate>.+)_(?P<step>\d+)\.pt$")


@dataclass
class RunRecord:
    experiment: str
    model_family: str
    ode_type: str
    run_dir: str
    config_path: str
    manifest_path: str
    checkpoint_path: str
    checkpoint_step: int
    sample_path: str = ""
    sample_manifest_path: str = ""
    sample_source: str = ""


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def assert_finite(name: str, values: Any) -> None:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got {array.dtype}")
    finite = np.isfinite(array)
    if not bool(finite.all()):
        raise FloatingPointError(f"{name} contains {int((~finite).sum())}/{array.size} NaN/Inf")


def checkpoint_step(path: Path) -> int:
    match = EMA_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"expected EMA checkpoint, got {path}")
    return int(match.group("step"))


def _complete_checkpoint(run_dir: Path, config: Mapping[str, Any], override: str = "") -> Path:
    candidates = [Path(override).expanduser().resolve()] if override else list(
        run_dir.glob("train/checkpoints/segment_*/*/ema_*_*.pt")
    )
    valid: list[Path] = []
    expected_rate = str(float(str(config.get("ema_rate", "0.9999")).split(",")[0]))
    for ema in candidates:
        match = EMA_RE.fullmatch(ema.name)
        if not ema.is_file() or not match or str(float(match.group("rate"))) != expected_rate:
            continue
        step_text = match.group("step")
        if (ema.parent / f"model{step_text}.pt").is_file() and (ema.parent / f"opt{step_text}.pt").is_file():
            valid.append(ema.resolve())
    if not valid:
        raise FileNotFoundError(f"no complete raw/optimizer/EMA checkpoint bundle: {run_dir}")
    selected = max(valid, key=lambda path: (checkpoint_step(path), path.stat().st_mtime_ns))
    target = int(config.get("total_steps", config.get("lr_anneal_steps", 0)))
    if target and checkpoint_step(selected) < target:
        raise RuntimeError(f"latest complete checkpoint step {checkpoint_step(selected)} < target {target}")
    return selected


def parse_checkpoint_overrides(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--checkpoint expects EXPERIMENT=/absolute/path/ema_...pt")
        experiment, path = value.split("=", 1)
        if experiment not in EXPERIMENTS:
            raise ValueError(f"unsupported checkpoint experiment: {experiment}")
        result[experiment] = path
    return result


def select_experiments(family: str, ode: str, experiments: Sequence[str]) -> tuple[str, ...]:
    requested = set(experiments)
    unknown = requested.difference(EXPERIMENTS)
    if unknown:
        raise ValueError(f"unknown experiments: {sorted(unknown)}")
    selected = tuple(
        name for name in EXPERIMENTS
        if (not family or name.startswith(family + "__"))
        and (not ode or name.endswith("__" + ode))
        and (not requested or name in requested)
    )
    if not selected:
        raise ValueError("filters select no experiments")
    return selected


def discover_runs(
    source_root: Path,
    selected: Sequence[str],
    batch_id: str,
    overrides: Mapping[str, str],
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for expected in selected:
        expected_family, expected_ode = expected.split("__", 1)
        candidates: list[Path] = []
        direct = source_root / "runs" / expected / batch_id
        if (direct / "manifest.json").is_file():
            candidates.append(direct)
        for manifest_path in source_root.rglob("manifest.json"):
            run_dir = manifest_path.parent
            if run_dir not in candidates and run_dir.name == batch_id:
                candidates.append(run_dir)
        matches: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
        for run_dir in candidates:
            manifest_path = run_dir / "manifest.json"
            config_path = run_dir / "exp_config.json"
            if not manifest_path.is_file() or not config_path.is_file():
                continue
            manifest, config = read_json(manifest_path), read_json(config_path)
            identity = (
                str(config.get("experiment", manifest.get("experiment", ""))),
                str(config.get("model_family", manifest.get("model_family", ""))),
                str(config.get("ode_type", manifest.get("ode_type", ""))),
            )
            if identity == (expected, expected_family, expected_ode):
                matches.append((run_dir.resolve(), manifest, config))
        if not matches:
            raise FileNotFoundError(f"no config-verified run for {expected}, batch={batch_id}")
        completed = [item for item in matches if item[1].get("stages", {}).get("train", {}).get("status") == "completed"]
        if not completed:
            raise RuntimeError(f"no completed training run for {expected}")
        run_dir, manifest, config = max(completed, key=lambda item: item[0].stat().st_mtime_ns)
        checkpoint = _complete_checkpoint(run_dir, config, overrides.get(expected, ""))
        records.append(RunRecord(
            experiment=expected, model_family=expected_family, ode_type=expected_ode,
            run_dir=str(run_dir), config_path=str(run_dir / "exp_config.json"),
            manifest_path=str(run_dir / "manifest.json"), checkpoint_path=str(checkpoint),
            checkpoint_step=checkpoint_step(checkpoint),
        ))
    return records


def _sample_array(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        keys = [key for key in SAMPLE_KEYS if key in archive.files]
        if not keys:
            raise KeyError(f"no generated array key in {path}: {archive.files}")
        array = np.asarray(archive[keys[0]], dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"sample must be 2-D: {path} {array.shape}")
    assert_finite(f"generated sample {path}", array)
    return array


def _matching_existing_sample(record: RunRecord, config: Mapping[str, Any]) -> tuple[Path, Path] | None:
    run_dir = Path(record.run_dir)
    checkpoint = Path(record.checkpoint_path).resolve()
    candidates: list[Path] = []
    run_manifest = read_json(Path(record.manifest_path))
    if run_manifest.get("sample_manifest_path"):
        candidates.append(Path(str(run_manifest["sample_manifest_path"])))
    candidates.extend(sorted((run_dir / "samples").glob("*.manifest.json"), reverse=True))
    for manifest_path in candidates:
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        recorded = manifest.get("checkpoint") or manifest.get("checkpoint_path")
        sample_value = manifest.get("sample_path")
        if not recorded or not sample_value:
            continue
        sample_path = Path(str(sample_value)).expanduser()
        try:
            same_checkpoint = Path(str(recorded)).expanduser().resolve() == checkpoint
        except OSError:
            same_checkpoint = False
        if not same_checkpoint or not sample_path.is_file():
            continue
        array = _sample_array(sample_path)
        if array.shape != (int(config["num_samples"]), int(config.get("n_genes", array.shape[1]))):
            continue
        if int(manifest.get("seed", -1)) != int(config["seed"]):
            continue
        if int(manifest.get("diffusion_steps", -1)) != int(config["diffusion_steps"]):
            continue
        return sample_path.resolve(), manifest_path.resolve()
    return None


def _gene_names(adata: Any) -> list[str]:
    if "gene_name" not in adata.var.columns:
        raise KeyError("AnnData .var lacks gene_name")
    genes = [str(value) for value in adata.var["gene_name"].tolist()]
    if len(genes) != adata.n_vars or len(set(genes)) != len(genes):
        raise ValueError("gene_name must be unique and one-to-one with variables")
    return genes


def _dense(value: Any) -> np.ndarray:
    if sparse.issparse(value):
        value = value.toarray()
    array = np.asarray(value, dtype=np.float32)
    assert_finite("real-data chunk", array)
    return array


def _source_matrix(adata: Any, layer: str | None) -> Any:
    if layer:
        if layer not in adata.layers:
            raise KeyError(f"configured layer not found: {layer}")
        return adata.layers[layer]
    return adata.X


def matrix_storage(matrix: Any) -> str:
    if sparse.issparse(matrix):
        return f"scipy_sparse_{matrix.format}"
    storage_format = getattr(matrix, "format", None)
    if storage_format in ("csr", "csc") or "SparseDataset" in type(matrix).__name__:
        return f"backed_sparse_{storage_format or type(matrix).__name__}"
    return f"dense_{type(matrix).__name__}"


def gene_order_sha256(genes: Sequence[str]) -> str:
    payload = "".join(f"{index}\t{name}\n" for index, name in enumerate(genes))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def real_gene_means(matrix: Any, n_cells: int, n_genes: int, chunk_rows: int) -> np.ndarray:
    sums = np.zeros(n_genes, dtype=np.float64)
    for start in range(0, n_cells, chunk_rows):
        chunk = _dense(matrix[start:min(start + chunk_rows, n_cells), :])
        sums += chunk.sum(axis=0, dtype=np.float64)
    means = sums / n_cells
    assert_finite("real gene means", means)
    return means


def assign_groups(means: np.ndarray, genes: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    indices = np.arange(len(means), dtype=np.int64)
    order = np.lexsort((indices, -means))
    groups = [np.asarray(group, dtype=np.int64) for group in np.array_split(order, 5)]
    rows: list[dict[str, Any]] = []
    rank = np.empty(len(means), dtype=np.int64); rank[order] = np.arange(1, len(means) + 1)
    for group_number, group in enumerate(groups):
        for within, gene_index in enumerate(group, start=1):
            rows.append({
                "group": f"group_{group_number}", "group_number": group_number,
                "gene_index": int(gene_index), "gene_name": genes[gene_index],
                "real_mean_expression": float(means[gene_index]),
                "rank_within_group": within, "rank_all_genes": int(rank[gene_index]),
            })
    assignment = pd.DataFrame(rows)
    summaries = []
    for group_number, group in enumerate(groups):
        values = means[group]
        summaries.append({
            "group": f"group_{group_number}", "group_number": group_number,
            "gene_count": len(group), "mean_expression_min": float(values.min()),
            "mean_expression_max": float(values.max()),
            "mean_expression_median": float(np.median(values)),
        })
    return assignment, pd.DataFrame(summaries), groups


def real_group_values(matrix: Any, n_cells: int, genes: np.ndarray, chunk_rows: int) -> np.ndarray:
    chunks = []
    for start in range(0, n_cells, chunk_rows):
        chunk = _dense(matrix[start:min(start + chunk_rows, n_cells), :])[:, genes]
        chunks.append(chunk.reshape(-1))
    result = np.concatenate(chunks)
    expected = n_cells * len(genes)
    if result.size != expected:
        raise AssertionError(f"flatten size {result.size} != {expected}")
    return result


def summarize(values: np.ndarray) -> dict[str, float | int]:
    assert_finite("distribution values", values)
    work = values.astype(np.float64, copy=False)
    return {
        "value_count": int(values.size), "mean": float(work.mean()),
        "standard_deviation": float(work.std()), "median": float(np.median(work)),
        "zero_fraction": float(np.count_nonzero(values == 0) / values.size),
        "min": float(work.min()), "max": float(work.max()),
    }


def deterministic_sample(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0 or count >= values.size:
        return values
    rng = np.random.default_rng(seed)
    return values[np.sort(rng.choice(values.size, count, replace=False))]


def histogram_density(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts.astype(np.float64) / (values.size * np.diff(edges))


def _new_output() -> Path:
    now = datetime.now()
    parent = OUTPUT_ROOT / now.strftime("%Y%m%d")
    base = now.strftime("%H%M%S")
    destination = parent / base
    suffix = 1
    while destination.exists():
        destination = parent / f"{base}_{suffix:02d}"; suffix += 1
    destination.mkdir(parents=True)
    for name in ("figures", "tables", "metadata", "generated_samples"):
        (destination / name).mkdir()
    return destination


def _select_device(torch: Any, requested: str) -> Any:
    choice = requested
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(choice)


def generate_sample(record: RunRecord, config: Mapping[str, Any], genes: Sequence[str], output: Path, device_name: str) -> tuple[Path, Path]:
    import torch
    from guided_diffusion import dist_util
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
    from ODE.ode_20260609_mathmlp import clean_state_dict

    source_root = Path(record.run_dir).parents[2]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from models import build_model_from_config

    seed = int(config["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = _select_device(torch, device_name)
    diffusion_args = model_and_diffusion_defaults()
    for key in diffusion_args:
        if key in config: diffusion_args[key] = config[key]
    _, diffusion = create_model_and_diffusion(**diffusion_args)
    model = build_model_from_config(config, list(genes), diffusion.num_timesteps, device)
    state = clean_state_dict(dist_util.load_state_dict(record.checkpoint_path, map_location="cpu"))
    for name, tensor in state.items():
        if torch.is_tensor(tensor) and not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"checkpoint tensor non-finite: {name}")
    model.load_state_dict(state, strict=True); model.to(device).eval()
    sample_fn = diffusion.ddim_sample_loop if bool(config.get("use_ddim", False)) else diffusion.p_sample_loop
    chunks = []; generated = 0; total = int(config["num_samples"]); batch = int(config["sample_batch_size"])
    with torch.no_grad():
        while generated < total:
            count = min(batch, total - generated)
            kwargs = {"clip_denoised": bool(config.get("clip_denoised", False))}
            if not bool(config.get("use_ddim", False)): kwargs["start_time"] = diffusion.num_timesteps
            sample, _ = sample_fn(model, (count, len(genes)), **kwargs)
            if not bool(torch.isfinite(sample).all()): raise FloatingPointError("generated sample non-finite")
            chunks.append(sample.detach().cpu().numpy().astype(np.float32)); generated += count
    array = np.concatenate(chunks)
    sample_path = output / "generated_samples" / f"{record.experiment}_{Path(record.checkpoint_path).stem}.npz"
    np.savez_compressed(sample_path, cell_gen=array)
    manifest_path = sample_path.with_suffix(".manifest.json")
    write_json(manifest_path, {
        "experiment": record.experiment, "checkpoint": record.checkpoint_path,
        "checkpoint_step": record.checkpoint_step, "sample_path": str(sample_path),
        "shape": list(array.shape), "seed": seed, "diffusion_steps": diffusion.num_timesteps,
        "noise_schedule": config.get("noise_schedule"), "use_ddim": bool(config.get("use_ddim", False)),
        "clip_denoised": bool(config.get("clip_denoised", False)), "device": str(device),
    })
    return sample_path, manifest_path


def _plot_grid(histograms: Mapping[tuple[str, int, str], np.ndarray], edges_by_group: Sequence[np.ndarray], ymax: Sequence[float], records: Sequence[RunRecord], group_sizes: Sequence[int], output: Path, stem: str, nonzero: bool = False) -> None:
    rows = len(records); fig, axes = plt.subplots(rows, 5, figsize=(21, 3.2 * rows), squeeze=False, sharex="col", sharey="col")
    colors = {"real": "#2563eb", "generated": "#dc2626"}
    for row, record in enumerate(records):
        for group in range(5):
            ax = axes[row, group]; edges = edges_by_group[group]
            centers = (edges[:-1] + edges[1:]) / 2; widths = np.diff(edges)
            for kind in ("real", "generated"):
                ax.bar(centers, histograms[(record.experiment, group, kind)], width=widths,
                       color=colors[kind], alpha=0.42, linewidth=0, label=kind)
            ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0, ymax[group] * 1.04 if ymax[group] else 1)
            ax.set_title(f"{record.experiment}\ngroup_{group} | genes={group_sizes[group]} | step={record.checkpoint_step}", fontsize=8)
            if row == rows - 1: ax.set_xlabel("expression value" + (" (non-zero)" if nonzero else ""))
            if group == 0: ax.set_ylabel("probability density")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k], alpha=.42) for k in ("real", "generated")]
    fig.legend(handles, ("real", "generated"), loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output / "figures" / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(output / "figures" / f"{stem}.pdf", bbox_inches="tight"); plt.close(fig)


def validate_figure_pair(png: Path, pdf: Path) -> None:
    from PIL import Image

    if not png.is_file() or not pdf.is_file():
        raise FileNotFoundError(f"missing figure pair: {png}, {pdf}")
    with Image.open(png) as image:
        image.verify()
        if image.width < 1000 or image.height < 500:
            raise RuntimeError(f"unexpectedly small rendered figure: {png} {image.size}")
    if not pdf.read_bytes()[:4] == b"%PDF":
        raise RuntimeError(f"invalid PDF header: {pdf}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).expanduser().resolve()
    selected = select_experiments(args.family, args.ode, args.experiment)
    records = discover_runs(source_root, selected, args.batch_id, parse_checkpoint_overrides(args.checkpoint))
    configs = {record.experiment: read_json(Path(record.config_path)) for record in records}
    comparison_keys = ("data_dir", "ts_layer", "num_samples", "seed", "diffusion_steps", "noise_schedule", "use_ddim", "clip_denoised", "sample_batch_size")
    reference = {key: configs[records[0].experiment].get(key) for key in comparison_keys}
    mismatches = {record.experiment: {key: configs[record.experiment].get(key) for key in comparison_keys if configs[record.experiment].get(key) != reference[key]} for record in records}
    mismatches = {key: value for key, value in mismatches.items() if value}
    if mismatches: raise RuntimeError(f"real/sampling configuration mismatch: {mismatches}")
    data_path = Path(str(reference["data_dir"])).expanduser()
    if not data_path.is_file(): raise FileNotFoundError(data_path)
    if args.dry_run:
        sample_status = {}
        for record in records:
            match = _matching_existing_sample(record, configs[record.experiment])
            sample_status[record.experiment] = {
                "action": "reuse" if match else ("error_missing_plot_only" if args.plot_only else "generate"),
                "sample_path": str(match[0]) if match else "",
                "sample_manifest_path": str(match[1]) if match else "",
            }
        result = {"dry_run": True, "selected": list(selected), "data_path": str(data_path), "runs": [asdict(record) for record in records], "sampling_config": reference, "sample_status": sample_status}
        print(json.dumps(result, indent=2, ensure_ascii=False)); return result

    output = _new_output(); (output / "metadata" / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    adata = sc.read_h5ad(data_path, backed="r"); genes = _gene_names(adata); layer = reference["ts_layer"]
    matrix = _source_matrix(adata, layer); n_cells, n_genes = int(adata.n_obs), int(adata.n_vars)
    means = real_gene_means(matrix, n_cells, n_genes, args.chunk_rows)
    assignment, group_summary, groups = assign_groups(means, genes)
    assignment.to_csv(output / "tables" / "gene_group_assignments.csv", index=False)
    group_summary.to_csv(output / "tables" / "gene_group_summary.csv", index=False)

    generated: dict[str, np.ndarray] = {}
    for record in records:
        config = configs[record.experiment]; match = _matching_existing_sample(record, config)
        if match:
            sample_path, sample_manifest = match; record.sample_source = "reused"
        elif args.plot_only:
            raise FileNotFoundError(f"no checkpoint-matched existing sample for {record.experiment}")
        else:
            sample_path, sample_manifest = generate_sample(record, config, genes, output, args.device); record.sample_source = "generated_for_distribution"
        record.sample_path, record.sample_manifest_path = str(sample_path), str(sample_manifest)
        array = _sample_array(sample_path)
        if array.shape[1] != n_genes: raise ValueError(f"gene dimension mismatch: {record.experiment}")
        generated[record.experiment] = array

    low, high = (float(value) for value in args.display_percentiles.split(","))
    if not 0 <= low < high <= 100: raise ValueError("--display-percentiles must be LOW,HIGH within 0..100")
    summary_rows = []; histogram_meta = {}; histograms = {}; nonzero_histograms = {}; edges_by_group = []; nonzero_edges = []; ymax = []; nonzero_ymax = []
    zero_rows = []
    for group_number, gene_indices in enumerate(groups):
        real_values = real_group_values(matrix, n_cells, gene_indices, args.chunk_rows)
        model_values = {record.experiment: generated[record.experiment][:, gene_indices].reshape(-1) for record in records}
        combined = np.concatenate([real_values, *model_values.values()])
        left, right = np.percentile(combined.astype(np.float64, copy=False), [low, high])
        if np.count_nonzero(combined == 0): left, right = min(left, 0.0), max(right, 0.0)
        if left == right: left, right = left - 0.5, right + 0.5
        edges = np.linspace(left, right, args.bins + 1); edges_by_group.append(edges)
        nonzero_all = combined[combined != 0]
        if nonzero_all.size:
            nz_left, nz_right = np.percentile(nonzero_all.astype(np.float64, copy=False), [low, high])
            if nz_left == nz_right: nz_left, nz_right = nz_left - .5, nz_right + .5
        else: nz_left, nz_right = 0.0, 1.0
        nz_edges = np.linspace(nz_left, nz_right, args.bins + 1); nonzero_edges.append(nz_edges)
        del combined, nonzero_all
        common_count = min([args.plot_subsample if args.plot_subsample > 0 else 2**63-1, real_values.size, *(v.size for v in model_values.values())])
        real_plot = deterministic_sample(real_values, common_count, args.seed + group_number)
        real_hist = histogram_density(real_plot, edges); real_nz = real_plot[real_plot != 0]
        real_nz_hist = histogram_density(real_nz, nz_edges) if real_nz.size else np.zeros(args.bins)
        group_max = float(real_hist.max()); nz_max = float(real_nz_hist.max())
        real_stats = summarize(real_values)
        histogram_meta[f"group_{group_number}"] = {"bin_edges": edges.tolist(), "nonzero_bin_edges": nz_edges.tolist(), "display_percentiles": [low, high], "plot_value_count_per_distribution": int(common_count)}
        for model_index, record in enumerate(records):
            values = model_values[record.experiment]
            generated_plot = deterministic_sample(values, common_count, args.seed + 1000 + group_number * 31 + model_index)
            gen_hist = histogram_density(generated_plot, edges); gen_nz = generated_plot[generated_plot != 0]
            gen_nz_hist = histogram_density(gen_nz, nz_edges) if gen_nz.size else np.zeros(args.bins)
            histograms[(record.experiment, group_number, "real")] = real_hist
            histograms[(record.experiment, group_number, "generated")] = gen_hist
            nonzero_histograms[(record.experiment, group_number, "real")] = real_nz_hist
            nonzero_histograms[(record.experiment, group_number, "generated")] = gen_nz_hist
            group_max = max(group_max, float(gen_hist.max())); nz_max = max(nz_max, float(gen_nz_hist.max()))
            gen_stats = summarize(values); distance_count = min(args.distance_subsample, real_values.size, values.size)
            real_distance = deterministic_sample(real_values, distance_count, args.seed + 2000 + group_number)
            gen_distance = deterministic_sample(values, distance_count, args.seed + 3000 + group_number * 31 + model_index)
            row = {"model_family": record.model_family, "ode": record.ode_type, "experiment": record.experiment, "checkpoint": record.checkpoint_path, "checkpoint_step": record.checkpoint_step, "group": f"group_{group_number}", "gene_count": len(gene_indices), "distance_subsample_count": distance_count}
            for prefix, stats in (("real", real_stats), ("generated", gen_stats)):
                for key, value in stats.items(): row[f"{prefix}_{key}"] = value
            row["wasserstein_distance"] = float(wasserstein_distance(real_distance, gen_distance)); row["ks_statistic"] = float(ks_2samp(real_distance, gen_distance).statistic)
            row["real_display_outside_fraction"] = float(np.mean((real_values < left) | (real_values > right)))
            row["generated_display_outside_fraction"] = float(np.mean((values < left) | (values > right)))
            summary_rows.append(row); zero_rows.append({"experiment": record.experiment, "group": f"group_{group_number}", "real_zero_fraction": real_stats["zero_fraction"], "generated_zero_fraction": gen_stats["zero_fraction"]})
        ymax.append(group_max); nonzero_ymax.append(nz_max)

    pd.DataFrame(summary_rows).to_csv(output / "tables" / "distribution_summary.csv", index=False)
    pd.DataFrame(zero_rows).to_csv(output / "tables" / "zero_fraction.csv", index=False)
    stem = "distribution_real_vs_generated_6x5" if len(records) == 6 else f"distribution_real_vs_generated_{len(records)}x5"
    _plot_grid(histograms, edges_by_group, ymax, records, [len(g) for g in groups], output, stem)
    _plot_grid(nonzero_histograms, nonzero_edges, nonzero_ymax, records, [len(g) for g in groups], output, stem + "_nonzero", nonzero=True)

    fig, axes = plt.subplots(1, 5, figsize=(21, 3.8), squeeze=False)
    first = records[0].experiment
    for group in range(5):
        ax = axes[0, group]; edges = edges_by_group[group]; centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, histograms[(first, group, "real")], width=np.diff(edges), color="#2563eb", alpha=.55)
        ax.set(xlim=(edges[0], edges[-1]), ylim=(0, ymax[group] * 1.04 if ymax[group] else 1), title=f"group_{group} | genes={len(groups[group])}", xlabel="expression value")
        if group == 0: ax.set_ylabel("probability density")
    fig.tight_layout(); fig.savefig(output / "figures" / "real_distribution_by_expression_group.png", dpi=320, bbox_inches="tight"); fig.savefig(output / "figures" / "real_distribution_by_expression_group.pdf", bbox_inches="tight"); plt.close(fig)

    zero_frame = pd.DataFrame(zero_rows); fig, axes = plt.subplots(1, 5, figsize=(21, 4), sharey=True)
    for group in range(5):
        part = zero_frame[zero_frame.group == f"group_{group}"]; x = np.arange(len(part)); width=.38
        axes[group].bar(x-width/2, part.real_zero_fraction, width, color="#2563eb", alpha=.65, label="real")
        axes[group].bar(x+width/2, part.generated_zero_fraction, width, color="#dc2626", alpha=.65, label="generated")
        axes[group].set_xticks(x, [name.replace("__", "\n") for name in part.experiment], rotation=90, fontsize=7); axes[group].set_title(f"group_{group}"); axes[group].set_ylim(0, 1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2); fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(output / "figures" / "zero_fraction_by_group.png", dpi=320, bbox_inches="tight"); fig.savefig(output / "figures" / "zero_fraction_by_group.pdf", bbox_inches="tight"); plt.close(fig)

    for figure_stem in (
        stem, stem + "_nonzero", "real_distribution_by_expression_group",
        "zero_fraction_by_group",
    ):
        validate_figure_pair(
            output / "figures" / f"{figure_stem}.png",
            output / "figures" / f"{figure_stem}.pdf",
        )

    real_min = min(float(row["real_min"]) for row in summary_rows)
    real_max = max(float(row["real_max"]) for row in summary_rows)
    metadata = {
        "status": "completed", "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "output_dir": str(output), "runs": [asdict(record) for record in records],
        "real_data": {"path": str(data_path), "layer": layer, "cells": n_cells, "genes": n_genes, "storage": matrix_storage(matrix), "value_min": real_min, "value_max": real_max, "gene_order_sha256": gene_order_sha256(genes), "value_space": "stored AnnData .X/layer used directly; train_vae=True, preprocess=False; no inferred inverse transform"},
        "group_method": "real-only mean descending; ties by original gene index; numpy.array_split into five near-equal groups",
        "flatten_method": "all cell x genes-in-group values; no cell/gene averaging",
        "bins": args.bins, "histogram_normalization": "counts / (all plotted values including out-of-range * bin width)",
        "figure_layout": {"rows": len(records), "columns": 5, "subplots": len(records) * 5, "png_dpi": 320, "png_pdf_validated": True},
        "histograms": histogram_meta, "plot_subsample": args.plot_subsample, "distance_subsample": args.distance_subsample,
        "seed": args.seed, "chunk_rows": args.chunk_rows, "arguments": vars(args),
    }
    write_json(output / "metadata" / "run_metadata.json", metadata); write_json(output / "metadata" / "histogram_bins.json", histogram_meta)
    adata.file.close()
    print(f"OUTPUT_DIR='{output}'"); return metadata


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(allow_abbrev=False)
    value.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT)); value.add_argument("--batch-id", default="20260803_full_30000")
    value.add_argument("--family", choices=("", *FAMILIES), default=""); value.add_argument("--ode", choices=("", *ODE_TYPES), default="")
    value.add_argument("--experiment", nargs="*", default=[]); value.add_argument("--plot-only", action="store_true")
    value.add_argument("--bins", type=int, default=200); value.add_argument("--plot-subsample", type=int, default=1000000)
    value.add_argument("--distance-subsample", type=int, default=200000); value.add_argument("--seed", type=int, default=1234)
    value.add_argument("--checkpoint", action="append", default=[], metavar="EXPERIMENT=PATH")
    value.add_argument("--display-percentiles", default="0.1,99.9"); value.add_argument("--chunk-rows", type=int, default=4096)
    value.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); value.add_argument("--dry-run", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.bins < 2 or args.plot_subsample < 0 or args.distance_subsample < 1 or args.chunk_rows < 1: raise ValueError("invalid positive numeric CLI argument")
    run(args); return 0


if __name__ == "__main__":
    raise SystemExit(main())
