"""20260911 independent fits plus one terminal joint fit, coordinates only."""

import numpy as np
from ..common import finite, umap_core, write_csv, write_json


def joint_indices(table):
    wanted = (950, 1000, 1050, 1100)
    lookup = {row["reverse_step"]: row["snapshot_index"] for row in table}
    if any(k not in lookup for k in wanted):
        raise ValueError(
            "joint UMAP requires a Stage-2 trajectory with 950/1000/1050/1100"
        )
    return [lookup[k] for k in wanted]


def label(row):
    return (
        f"Diffusion {row['reverse_step']}"
        if row["phase"] == "diffusion"
        else f"ODE +{row['ode_step']}"
    )


def snapshot_matrix(states, predictions, row):
    k = row["snapshot_index"]
    return np.asarray(
        predictions[:, k] if row["phase"] == "diffusion" else states[:, k]
    )


def embedding_inputs(real, states, predictions, table, selected, core=None):
    core = umap_core() if core is None else core
    generated = np.concatenate(
        [snapshot_matrix(states, predictions, table[k]) for k in selected], axis=0
    )
    combined = core.build_sampling_anndata(real, generated)
    combined.obs["time_label"] = ["Real Erythropoietic"] * real.n_obs + [
        label(table[k]) for k in selected for _ in range(len(states))
    ]
    combined.obs["trajectory_id"] = [-1] * real.n_obs + list(range(len(states))) * len(
        selected
    )
    return combined


def compute_embeddings(real, states, predictions, table, output, *, seed):
    core = umap_core()
    fits = [
        (f"independent_{row['reverse_step']:04d}", [row["snapshot_index"]])
        for row in table
    ]
    if len(table) == 22:
        fits.append(("joint_terminal", joint_indices(table)))
    metadata = []
    for name, selected in fits:
        combined = embedding_inputs(
            real, states, predictions, table, selected, core=core
        )
        parameters = core.compute_common_umap(combined, seed=seed)
        xy = finite("UMAP coordinates", np.asarray(combined.obsm["X_umap"]))
        rows = [
            {
                "UMAP1": float(a),
                "UMAP2": float(b),
                "time_label": str(time),
                "trajectory_id": int(cell),
            }
            for (a, b), time, cell in zip(
                xy, combined.obs["time_label"], combined.obs["trajectory_id"]
            )
        ]
        write_csv(output / f"{name}.csv", rows)
        metadata.append(
            {
                "fit": name,
                "snapshot_indices": selected,
                "labels": [label(table[k]) for k in selected],
                "parameters": parameters,
                "representation": "diffusion pred_xstart; post_ode state after update",
                "coordinate_alignment": "one fit per independent figure; independent fits are not aligned trajectories",
            }
        )
    write_json(output / "umap_metadata.json", metadata)
