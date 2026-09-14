"""Immutable Gaussian-start reverse-diffusion cache, shared across ODE families."""

from pathlib import Path
import numpy as np
import torch
from ..common import (
    SUITE,
    build_diffusion,
    confined,
    file_hash,
    finite,
    git_commit,
    new_dir,
    read_json,
    run_id,
    seed_all,
    state_hash,
    write_json,
)
from .checkpoints import canonical_stage1, restore

CACHE_SCHEMA = "stage1_generated_x50_v1"


def cache_pointer(campaign, role):
    if role not in ("train", "evaluation"):
        raise ValueError("cache role must be train or evaluation")
    return Path(campaign) / f"canonical_x50_{role}.json"


def load_cache(path, canonical, *, role=None):
    path = confined(path)
    meta = read_json(path / "cache_metadata.json")
    if (
        read_json(path / "completed.json")["status"] != "completed"
        or (path / "failed.json").exists()
    ):
        raise ValueError("cache is incomplete")
    if (
        meta["cache_schema"] != CACHE_SCHEMA
        or meta["checkpoint_sha256"] != canonical["checkpoint_sha256"]
        or meta["gene_order_hash"] != canonical["gene_order_hash"]
        or meta["cellunet_hash"] != canonical["cellunet_hash"]
        or (role is not None and meta["role"] != role)
    ):
        raise ValueError("x50 cache provenance mismatch")
    if file_hash(path / "x50.npy") != meta["states_sha256"]:
        raise ValueError("x50 cache states checksum mismatch")
    states = np.load(path / "x50.npy", mmap_mode="r", allow_pickle=False)
    if list(states.shape) != [meta["count"], len(meta["gene_names"])]:
        raise ValueError("x50 cache shape mismatch")
    for i in range(0, len(states), 256):
        finite("cached x50", states[i : i + 256])
    return states, {
        **meta,
        "path": str(path),
        "metadata_sha256": file_hash(path / "cache_metadata.json"),
    }


@torch.no_grad()
def generate_cache(
    model,
    diffusion,
    output,
    *,
    count,
    batch_size,
    genes,
    canonical,
    seed,
    role,
    device,
    start_t=50,
):
    if hasattr(model, "ode_model") or diffusion.num_timesteps != 1000:
        raise ValueError(
            "x50 generation requires pure Stage1 and full 1000-step diffusion"
        )
    if not 0 <= start_t < 999 or count < 1 or batch_size < 1:
        raise ValueError("invalid cache size/batch/start_t")
    if state_hash(model) != canonical["cellunet_hash"]:
        raise ValueError("cache model differs from canonical Stage1 EMA")
    output = new_dir(output)
    states = np.lib.format.open_memmap(
        output / "x50.npy", mode="w+", dtype="float32", shape=(count, len(genes))
    )
    model.eval()
    seed_all(seed)
    try:
        for start in range(0, count, batch_size):
            n = min(batch_size, count - start)
            x = torch.randn((n, len(genes)), device=device)
            # p_sample at index t maps x_t to x_{t-1}. Call 999..51 inclusive:
            # 949 updates, giving x_50, the INPUT to the next update at t=50.
            for index in range(999, start_t, -1):
                t = torch.full((n,), index, device=device, dtype=torch.long)
                x = finite(
                    "reverse-generated cache state",
                    diffusion.p_sample(model, x, t, clip_denoised=False)["sample"],
                )
            states[start : start + n] = finite("x50 float32", x.float()).cpu().numpy()
            states.flush()
            print(f"cached {start + n}/{count} x{start_t} states", flush=True)
        states.flush()
        write_json(
            output / "cache_metadata.json",
            dict(
                cache_schema=CACHE_SCHEMA,
                role=role,
                checkpoint=canonical["checkpoint"],
                checkpoint_sha256=canonical["checkpoint_sha256"],
                cellunet_hash=canonical["cellunet_hash"],
                gene_order_hash=canonical["gene_order_hash"],
                gene_names=genes,
                count=count,
                batch_size=batch_size,
                seed=seed,
                device=str(device),
                git_commit=git_commit(),
                start_diffusion_t=start_t,
                reverse_input_timesteps=[999, start_t + 1],
                reverse_updates=999 - start_t,
                state_semantics=f"x_{start_t}: input to p_sample(t={start_t}); after p_sample(t={start_t + 1}); Gaussian start, never forward-noised real cells",
                diffusion=dict(
                    steps=1000,
                    noise_schedule="linear",
                    timestep_respacing="",
                    clip_denoised=False,
                    source_p_sample_nw=0.5,
                    betas=np.asarray(diffusion.betas).tolist(),
                ),
                states_sha256=file_hash(output / "x50.npy"),
            ),
        )
        write_json(output / "completed.json", {"status": "completed"})
    except BaseException as exc:
        write_json(output / "failed.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    return output


def cache_command(args):
    campaign = confined(SUITE / "runs" / args.campaign)
    payload, canonical = canonical_stage1(campaign)
    defaults = read_json(SUITE / "configs/trajectory_defaults.json")
    cfg = defaults["trajectory_ot" if args.role == "train" else "evaluation"]
    count = args.cache_size if args.cache_size is not None else cfg["source_cache_size"]
    seed = args.seed if args.seed is not None else cfg["source_seed"]
    other_role = "evaluation" if args.role == "train" else "train"
    other_pointer = cache_pointer(campaign, other_role)
    if other_pointer.exists():
        _, other = load_cache(
            read_json(other_pointer)["path"], canonical, role=other_role
        )
        if seed == other["seed"]:
            raise ValueError(
                "evaluation source seed must differ from training cache seed"
            )
    pointer = cache_pointer(campaign, args.role)
    if pointer.exists():
        path = read_json(pointer)["path"]
        _, meta = load_cache(path, canonical, role=args.role)
        if (
            meta["count"],
            meta["seed"],
            meta["batch_size"],
            meta["start_diffusion_t"],
        ) != (count, seed, args.batch_size, args.start_t):
            raise ValueError(
                "existing canonical cache has different settings; never overwrite/redefine it"
            )
    else:
        model, meta = restore(canonical["checkpoint"], args.device)
        path = generate_cache(
            model,
            build_diffusion(payload["metadata"]["effective_config"]),
            campaign / "x50" / (args.role + "_" + run_id()),
            count=count,
            batch_size=args.batch_size,
            genes=meta["gene_names"],
            canonical=canonical,
            seed=seed,
            role=args.role,
            device=args.device,
            start_t=args.start_t,
        )
        write_json(pointer, {"path": str(path)})
    print(f"X50_CACHE={path}")
    return path
