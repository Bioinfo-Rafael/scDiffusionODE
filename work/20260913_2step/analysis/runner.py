import copy
from pathlib import Path
import shutil
import numpy as np
from ..common import (
    build_diffusion,
    confined,
    file_hash,
    load_real,
    new_dir,
    read_json,
    run_id,
    seed_all,
    write_json,
)
from ..training.checkpoints import restore
from .diagnostics import diagnostics
from .distributions import trajectory_metrics
from .umaps import compute_embeddings


def load_trajectory(path):
    path = confined(path)
    complete = read_json(path / "completed.json")
    if complete["status"] != "completed" or (path / "failed.json").exists():
        raise ValueError("refusing incomplete/failed trajectory")
    meta = read_json(path / "sampling_metadata.json")
    states = np.load(path / "sample_state.npy", mmap_mode="r")
    predictions = np.load(path / "pred_xstart.npy", mmap_mode="r")
    expected = (meta["count"], len(meta["snapshot_table"]), len(meta["gene_names"]))
    if states.shape != expected or predictions.shape != (expected[0], 20, expected[2]):
        raise ValueError("trajectory shapes differ from metadata")
    return meta, states, predictions


def analyze(args):
    trajectory = confined(args.trajectory)
    meta, states, predictions = load_trajectory(trajectory)
    config = copy.deepcopy(meta["effective_config"])
    ot_cap = getattr(args, "ot_max_iterations", None)
    if ot_cap is not None:
        if ot_cap < config["ot"]["max_iterations"]:
            raise ValueError("evaluation iteration cap must not decrease")
        config["ot"]["max_iterations"] = ot_cap
    if args.data:
        config["data_dir"] = str(Path(args.data).expanduser().resolve())
    seed = config["seed"]
    real, _, selection = load_real(config, meta["gene_names"], erythropoietic=True)
    output = new_dir(trajectory / args.action / run_id())
    write_json(
        output / "analysis_metadata.json",
        {
            "trajectory": str(trajectory),
            "selection": selection,
            "seed": seed,
            "checkpoint": meta["checkpoint"],
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "training_config": meta["effective_config"],
            "ot_iteration_cap_override": ot_cap,
            "effective_config": config,
            "sampling_config": meta["sampling_config"],
            "action": args.action,
            "noise_policy": "same seeded float64 CPU Gaussian stream reset at each diffusion timestep",
            "metric_aggregation": "per cell across genes, then mean and population std across cells",
        },
    )
    try:
        if args.action == "embed":
            compute_embeddings(
                real, states, predictions, meta["snapshot_table"], output, seed=seed
            )
        else:
            if file_hash(meta["checkpoint"]) != meta["checkpoint_sha256"]:
                raise ValueError("sampling checkpoint has changed")
            seed_all(seed)
            model, restored_meta = restore(meta["checkpoint"], args.device)
            if restored_meta["gene_names"] != meta["gene_names"]:
                raise ValueError("checkpoint/trajectory gene mismatch")
            diffusion = build_diffusion(config)
            # Reference stays in original X space; subselect before densification.
            rng = np.random.default_rng(seed)
            ids = np.sort(
                rng.choice(
                    real.n_obs, min(config["analysis_cells"], real.n_obs), replace=False
                )
            )
            x = real.X[ids]
            x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
            with (output / "reference_cell_ids.npz").open("xb") as f:
                np.savez(
                    f,
                    indices_in_erythropoietic_subset=ids,
                    obs_names=real.obs_names.to_numpy(dtype=str)[ids],
                )
            diagnostics(
                model,
                diffusion,
                x,
                output,
                batch_size=config["analysis_batch_size"],
                seed=seed,
                division_epsilon=config["division_epsilon"],
            )
            trajectory_metrics(
                states, x, meta["snapshot_table"], output, config, seed=seed
            )
            convergence = trajectory / "post_ode_convergence.csv"
            if convergence.exists():
                with (
                    convergence.open("rb") as source,
                    (output / convergence.name).open("xb") as target,
                ):
                    shutil.copyfileobj(source, target)
        write_json(output / "completed.json", {"status": "completed"})
    except BaseException as exc:
        write_json(output / "failed.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    print(f"ANALYSIS_DIR={output}")
    return output
