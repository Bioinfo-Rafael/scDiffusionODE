"""Independent x50 ODE occupation evaluation; normal Hybrid sampling is separate."""

from pathlib import Path
import numpy as np
import torch
from ..common import (
    SUITE,
    file_hash,
    load_real,
    new_dir,
    read_json,
    run_id,
    write_csv,
    write_json,
)
from ..training.checkpoints import canonical_stage1, restore
from ..training.source_cache import load_cache
from ..training.trajectory import integrate, occupation_sample, validate_trajectory
from ..losses.sinkhorn import sinkhorn_divergence
from .distributions import diversity
from .sliced_wasserstein import sliced_wasserstein


@torch.no_grad()
def evaluate_occupation(
    model, source, real, config, output, *, provenance, device, batch_size
):
    c = config["evaluation"]
    validate_trajectory(c)
    seed = c["seed"]
    npaths = min(c["num_trajectories"], len(source))
    if min(npaths, real.shape[0]) < 2 or batch_size < 1:
        raise ValueError(
            "occupation evaluation needs >=2 trajectories/real cells and positive batch size"
        )
    rng = np.random.default_rng(seed)
    source_ids = rng.choice(len(source), npaths, replace=False)
    temporal_rng = np.random.default_rng(seed + 1)
    points, endpoints, times = [], [], []
    model.eval()
    for start in range(0, npaths, batch_size):
        x = torch.as_tensor(
            np.array(source[source_ids[start : start + batch_size]], copy=True),
            device=device,
        )
        states = integrate(
            model.ode_model, x, steps=c["ode_steps"], dt=c["ode_dt"], checkpoint_block=0
        )
        sample, selected_times = occupation_sample(states, c, temporal_rng)
        points.append(sample.cpu().numpy())
        endpoints.append(states[:, -1].cpu().numpy())
        times.append(selected_times)
    points, endpoints = np.concatenate(points), np.concatenate(endpoints)
    real_count = min(
        real.shape[0],
        max(c["sliced_wasserstein_points"], c["sinkhorn_points"], len(endpoints)),
    )
    real_ids = np.random.default_rng(seed + 2).choice(
        real.shape[0], real_count, replace=False
    )
    target = real[real_ids]
    target = target.toarray() if hasattr(target, "toarray") else np.asarray(target)
    rows, scales, selected = [], [], {}
    for distribution, pool in (("occupation", points), ("endpoint", endpoints)):
        n = min(c["sliced_wasserstein_points"], len(pool), len(target))
        r = np.random.default_rng(seed + 3)
        gids, tids = (
            r.choice(len(pool), n, replace=False),
            r.choice(len(target), n, replace=False),
        )
        generated, reference = pool[gids], target[tids]
        sw = sliced_wasserstein(
            generated,
            reference,
            projections=c["sliced_wasserstein_projections"],
            points=n,
            seed=seed,
        )
        not_ = min(c["sinkhorn_points"], n)
        # The SW/diversity sample is larger than the explicitly bounded dense OT sample.
        ot_ids = np.random.default_rng(seed + 4).choice(n, not_, replace=False)
        distance, info = sinkhorn_divergence(
            torch.as_tensor(generated[ot_ids], device=device),
            torch.as_tensor(reference[ot_ids], device=device),
            config["ot"],
            return_info=True,
            cost_diagnostics=True,
        )
        for term, detail in info.items():
            scales.extend(
                dict(distribution=distribution, term=term, **scale)
                for scale in detail["scales"]
            )
        d = diversity(generated, pairs=config["analysis_pairs"], seed=seed)
        rows.append(
            dict(
                condition=provenance["condition"],
                checkpoint=provenance["checkpoint"],
                trajectory_start_source=provenance["source_cache"]["path"],
                reference_population="Erythropoietic",
                distribution=distribution,
                num_trajectories=npaths,
                ode_steps=c["ode_steps"],
                trajectory_dt=c["ode_dt"],
                occupation_points_total=npaths * (c["ode_steps"] + 1),
                occupation_points_stratified=len(points),
                occupation_points_used=n if distribution == "occupation" else 0,
                generated_points_used=n,
                real_points_used=n,
                sinkhorn_points_used=not_,
                sinkhorn_divergence=float(distance),
                sliced_wasserstein2=sw["sliced_wasserstein2"],
                sliced_wasserstein2_squared=sw["sliced_wasserstein2_squared"],
                mean_pairwise_diversity=d["mean"],
                median_pairwise_diversity=d["median"],
                seed=seed,
                train_trajectory_batch_size=provenance["training_config"]
                .get("trajectory_ot", {})
                .get("trajectory_batch_size"),
                train_samples_per_trajectory=provenance["training_config"]
                .get("trajectory_ot", {})
                .get("samples_per_trajectory"),
                eval_samples_per_trajectory=c["samples_per_trajectory"],
                eval_sw_projections=c["sliced_wasserstein_projections"],
                **info["cross"]["cost_statistics"],
            )
        )
        selected[distribution + "_generated_ids"] = gids
        selected[distribution + "_real_ids"] = real_ids[tids]
        selected[distribution + "_sinkhorn_ids"] = ot_ids
    write_csv(output / "occupation_analysis.csv", rows)
    write_csv(output / "occupation_sinkhorn_scales.csv", scales)
    with (output / "occupation_subsample_ids.npz").open("xb") as f:
        np.savez(
            f, source_ids=source_ids, temporal_indices=np.concatenate(times), **selected
        )
    return rows


def occupation_command(args):
    model, meta = restore(args.checkpoint, args.device)
    if not hasattr(model, "ode_model"):
        raise ValueError("occupation evaluation requires a Stage2 ODE branch")
    canonical = meta["originating_stage1"]
    # Validate canonical Stage1 itself as well as cache ancestry.
    _, checked = canonical_stage1(Path(canonical["checkpoint"]).parents[3])
    if checked["checkpoint_sha256"] != canonical["checkpoint_sha256"]:
        raise ValueError("evaluation canonical Stage1 differs from model ancestry")
    source, cache_meta = load_cache(args.source_cache, canonical, role="evaluation")
    if (
        cache_meta["gene_names"] != meta["gene_names"]
        or cache_meta["start_diffusion_t"] != 50
    ):
        raise ValueError("evaluation x50 gene order/state semantics mismatch")
    training_cache = meta.get("trajectory_source_cache")
    if training_cache and cache_meta["seed"] == training_cache["seed"]:
        raise ValueError("evaluation cache must use an independent seed")
    defaults = read_json(SUITE / "configs/trajectory_defaults.json")
    config = {
        **meta["effective_config"],
        "evaluation": dict(defaults["evaluation"]),
        "ot": dict(defaults["ot"]),
    }
    config["evaluation"]["batch_size"] = args.batch_size
    train = meta["effective_config"].get("trajectory_ot", {})
    for key in (
        "ode_steps",
        "ode_dt",
        "samples_per_trajectory",
        "temporal_bins",
        "temporal_sampling",
        "binning_rule",
    ):
        if key in train:
            config["evaluation"][key] = train[key]
    for arg, key in (
        ("sw_projections", "sliced_wasserstein_projections"),
        ("sw_points", "sliced_wasserstein_points"),
        ("eval_seed", "seed"),
        ("eval_trajectories", "num_trajectories"),
    ):
        if getattr(args, arg, None) is not None:
            config["evaluation"][key] = getattr(args, arg)
    if args.ot_max_iterations is not None:
        config["ot"]["max_iterations"] = args.ot_max_iterations
    if args.data:
        config["data_dir"] = args.data
    real, _, selection = load_real(config, meta["gene_names"], erythropoietic=True)
    output = new_dir(SUITE / "results" / config["condition"] / "occupation" / run_id())
    provenance = dict(
        condition=config["condition"],
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=file_hash(args.checkpoint),
        source_cache=cache_meta,
        training_config=meta["effective_config"],
        evaluation_config=config,
        selection=selection,
        training_loss_history=str(Path(args.checkpoint).parents[1] / "losses.csv"),
        semantics="independent generated x50 -> ODE-only t_cond=0; occupation vs endpoint; distinct from normal Hybrid inference",
    )
    write_json(output / "occupation_metadata.json", provenance)
    try:
        evaluate_occupation(
            model,
            source,
            real.X,
            config,
            output,
            provenance=provenance,
            device=args.device,
            batch_size=args.batch_size,
        )
        write_json(output / "completed.json", {"status": "completed"})
    except BaseException as exc:
        write_json(output / "failed.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    print(f"OCCUPATION_DIR={output}")
    return output
