"""Save clean predictions and states from the existing progressive generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_csv, write_json
from .settings import Settings


def snapshot_table(settings: Settings, timestep_map: list[int] | None = None) -> list[dict]:
    mapping = list(range(settings.T)) if timestep_map is None else list(timestep_map)
    if (
        len(mapping) != settings.T
        or mapping[0] != 0
        or any(a >= b for a, b in zip(mapping, mapping[1:]))
    ):
        raise ValueError("timestep_map must start at zero and strictly increase with T entries")
    # Existing loop traverses range(start_time)[::-1], yielding AFTER p_sample.
    steps = range(settings.T // settings.M, settings.T + 1, settings.T // settings.M)
    table = [
        {"snapshot_index": k, "reverse_step": step, "diffusion_t": mapping[settings.T - step]}
        for k, step in enumerate(steps)
    ]
    assert len(table) == settings.M and table[-1]["reverse_step"] == settings.T
    return table


def sample_to_disk(
    model: Any,
    diffusion: Any,
    output: Path,
    settings: Settings,
    *,
    gene_count: int,
    batch_size: int,
    device: Any,
    clip_denoised: bool = False,
    save_states: bool = True,
) -> dict:
    if gene_count <= 0 or batch_size <= 0:
        raise ValueError("gene_count and batch_size must be positive")
    if diffusion.num_timesteps != settings.T:
        raise ValueError("diffusion step count differs from requested T")
    table = snapshot_table(settings, list(getattr(diffusion, "timestep_map", range(settings.T))))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "snapshot_table.csv", table)
    selected = {row["reverse_step"]: row["snapshot_index"] for row in table}
    return _sample_to_disk(
        model,
        diffusion,
        output,
        settings,
        table,
        selected,
        gene_count,
        batch_size,
        device,
        clip_denoised,
        save_states,
    )


def _sample_to_disk(
    model,
    diffusion,
    output,
    settings,
    table,
    selected,
    gene_count,
    batch_size,
    device,
    clip_denoised,
    save_states,
) -> dict:
    import torch

    shape = (settings.trajectories, settings.M, gene_count)
    arrays = {
        "pred_xstart": np.lib.format.open_memmap(
            output / "pred_xstart.npy", mode="w+", dtype="float32", shape=shape
        )
    }
    if save_states:
        arrays["sample"] = np.lib.format.open_memmap(
            output / "sample_state.npy", mode="w+", dtype="float32", shape=shape
        )
    metadata = {
        "status": "running",
        "shape": list(shape),
        "dtype": "float32",
        "batch_size": batch_size,
        "timestep_map": list(getattr(diffusion, "timestep_map", range(settings.T))),
        "snapshot_table": table,
        "save_states": save_states,
        "trajectory_order": "row id = n*S+j, contiguous batches, no shuffle/padding",
        "completed_trajectories": 0,
        "pred_xstart_semantics": "clean prediction at diffusion_t computed during this update",
        "sample_semantics": "actual reverse state after this update",
    }
    write_json(output / "sampling_metadata.json", metadata)
    final_max_error = 0.0
    try:
        with torch.no_grad():
            for start in range(0, settings.trajectories, batch_size):
                count = min(batch_size, settings.trajectories - start)
                seen, completed = 0, 0
                for completed, out in enumerate(
                    diffusion.p_sample_loop_progressive(
                        model,
                        (count, gene_count),
                        start_time=settings.T,
                        device=device,
                        clip_denoised=clip_denoised,
                    ),
                    start=1,
                ):
                    if completed > settings.T:
                        raise RuntimeError("progressive sampler returned too many updates")
                    if completed not in selected:
                        continue
                    k = selected[completed]
                    for key in ("pred_xstart", "sample"):
                        value = out[key].detach().cpu().numpy().astype(np.float32, copy=False)
                        if value.shape != (count, gene_count) or not np.isfinite(value).all():
                            raise FloatingPointError(
                                f"invalid {key} at batch {start}, completed update {completed}"
                            )
                        if key in arrays:
                            arrays[key][start : start + count, k, :] = value
                    seen += 1
                if completed != settings.T or seen != settings.M:
                    raise RuntimeError(
                        f"incomplete generator: {completed} updates / {seen} snapshots"
                    )
                error = float((out["pred_xstart"] - out["sample"]).abs().max().item())
                final_max_error = max(final_max_error, error)
                if not torch.allclose(out["sample"], out["pred_xstart"], rtol=1e-5, atol=1e-6):
                    raise AssertionError(
                        "unconditional final sample differs from final pred_xstart"
                    )
                for array in arrays.values():
                    array.flush()
                metadata["completed_trajectories"] = start + count
                write_json(output / "sampling_metadata.json", metadata)
                print(f"Trajectories {start + count}/{settings.trajectories}", flush=True)
        metadata.update(
            status="completed",
            final_reverse_step=settings.T,
            final_diffusion_t=table[-1]["diffusion_t"],
            final_pred_sample_max_abs_error=final_max_error,
            final_pred_sample_allclose=True,
            final_relation="at original t=0 stochastic noise is zero; posterior mean equals pred_xstart within float32 tolerance",
        )
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for array in arrays.values():
            array.flush()
        write_json(output / "sampling_metadata.json", metadata)
    return metadata
