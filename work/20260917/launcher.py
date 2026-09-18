"""Sequential fresh-process campaign; identical overrides for all ten conditions."""
import argparse
import subprocess
import sys
from pathlib import Path
from .common import (SUITE, ROOT, MODELS, LOSSES, CONDITIONS, STAGE1, effective_config, file_hash,
                     new_dir, read_json, run_id, write_json, source_provenance, load_real, umap_core)
from .training.checkpoints import load_stage1, save_checkpoint
from .scripts.summarize import summarize
from .src.progress import phase, report


def worker(campaign, label, arguments):
    log = campaign / (label + ".log")
    with phase(f"{label} log={log}"), log.open("x", buffering=1) as handle:
        child = subprocess.Popen(
            [sys.executable, "-u", "-m", "work.20260917.cli", *map(str, arguments)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            report(f"WORKER {label} pid={child.pid}")
            for line in child.stdout:
                handle.write(line)
                print(f"[{label}] {line}", end="", flush=True)
            code = child.wait()
            if code:
                raise RuntimeError(f"worker failed ({code}): {log}")
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
        finally:
            child.stdout.close()
    return log


def marker(log, prefix):
    for line in reversed(log.read_text().splitlines()):
        if line.startswith(prefix+"="):
            return Path(line.split("=",1)[1])
    raise ValueError(f"missing {prefix} in {log}")


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--model-type", choices=MODELS)
    p.add_argument("--aux-loss", choices=LOSSES)
    p.add_argument("--overrides", help="JSON common overrides; nested knn keys merge")
    p.add_argument("--stage1-checkpoint")
    p.add_argument("--data")
    p.add_argument("--edge-tsv")
    p.add_argument("--device", default="cuda")
    p.add_argument("--evaluation-device", default="cpu")
    p.add_argument("--training-steps", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="write all configs without data/checkpoint access")
    a = p.parse_args(argv)
    overrides = read_json(a.overrides) if a.overrides else {}
    if any(k in overrides for k in ("condition", "model_type", "aux_loss", "objective", "output_campaign")):
        raise ValueError("model/loss identities must be selected by CLI, not common overrides")
    for arg,key in ((a.stage1_checkpoint,"stage1_checkpoint"),(a.data,"data_dir"),(a.edge_tsv,"edge_tsv_path"),
                    (a.training_steps,"total_steps"),(a.epochs,"epochs")):
        if arg is not None:
            overrides[key] = arg
    if a.training_steps is not None and a.epochs is not None:
        p.error("choose either --epochs or --training-steps")
    # All config validation happens before creating a campaign.
    configs = {f"{m}_{loss}":effective_config(m,loss,overrides) for m in MODELS for loss in LOSSES}
    campaign = new_dir(SUITE / "results" / ("comparison_"+run_id()))
    new_dir(campaign / "configs")
    for name,c in configs.items():
        c["output_campaign"] = campaign.name
        write_json(campaign / "configs" / (name+".json"), c)
    chosen = [name for name,c in configs.items() if (a.model_type is None or c["model_type"] == a.model_type)
              and (a.aux_loss is None or c["aux_loss"] == a.aux_loss)]
    write_json(campaign / "campaign.json", dict(conditions=chosen, dry_run=a.dry_run, arguments=vars(a)))
    write_json(campaign / "source_sha256.json", source_provenance())
    print(f"CAMPAIGN={campaign}", flush=True)
    order = "training only" if a.train_only else "shared baseline sample/analyze/embed, then condition train/sample/analyze/embed"
    report(f"PLAN conditions={len(chosen)} order={order}")
    if a.dry_run:
        summarize(campaign)
        return
    c = next(iter(configs.values()))
    # Preflight once before any expensive training; strict source conventions retained.
    with phase("preflight: load and validate Stage1 checkpoint"):
        payload, origin = load_stage1(c["stage1_checkpoint"])
    with phase("preflight: verify dataset and GRN hashes"):
        if file_hash(c["data_dir"]) != origin["data_sha256"] or file_hash(c["edge_tsv_path"]) != origin["edge_tsv_sha256"]:
            raise ValueError("dataset/GRN differs from Stage1; supply a path to identical files")
    with phase("preflight: load dataset and validate gene order"):
        data, genes, _ = load_real(c, payload["metadata"]["gene_names"])
        if umap_core().gene_order_hash(genes) != origin["gene_order_hash"]:
            raise ValueError("gene order differs from Stage1")
    from .src.grn import load_grn
    with phase("preflight: map GRN to dataset columns"):
        src, _, _ = load_grn(genes, c["edge_tsv_path"])
        report(f"DATA cells={data.n_obs} genes={len(genes)} GRN_edges={len(src)}")
    if data.n_obs < c["batch_size"]:
        raise ValueError("not enough cells for one source batch")
    del data
    failures = []
    if not a.train_only:
        # Shared CellUNet-only baseline, original checkpoint state with local metadata.
        baseline_config = dict(c, condition=STAGE1)
        baseline_meta = dict(payload["metadata"], effective_config=baseline_config,
                             originating_stage1=origin)
        with phase("save shared baseline checkpoint"):
            path = save_checkpoint(campaign / "stage1_ema.pt", payload["state_dict"], baseline_meta)
        try:
            log = worker(campaign, "cellunet_only_sample", ["sample","--checkpoint",path,"--device",a.device])
            trajectory = marker(log, "TRAJECTORY_DIR")
            for action in ("analyze", "embed"):
                try:
                    worker(campaign, "cellunet_only_"+action, [action,"--trajectory",trajectory,"--device",a.evaluation_device])
                except Exception as exc:
                    failures.append(dict(condition=STAGE1, action=action, error=str(exc)))
        except Exception as exc:
            failures.append(dict(condition=STAGE1, action="sample", error=str(exc)))
    del payload
    for number, name in enumerate(chosen, start=1):
        report(f"CONDITION {number}/{len(chosen)} {name}")
        try:
            log = worker(campaign, name+"_train", ["train","--config",campaign/"configs"/(name+".json"),
                         "--output",campaign/name/"training","--device",a.device])
            checkpoint = marker(log, "EMA_CHECKPOINT")
            if not a.train_only:
                log = worker(campaign, name+"_sample", ["sample","--checkpoint",checkpoint,"--device",a.device])
                trajectory = marker(log, "TRAJECTORY_DIR")
                for action in ("analyze", "embed"):
                    try:
                        log = worker(campaign, name+"_"+action, [action,"--trajectory",trajectory,"--device",a.evaluation_device])
                        worker(campaign, name+"_"+action+"_plot", ["plot","--input",marker(log,"ANALYSIS_DIR")])
                    except Exception as exc:
                        failures.append(dict(condition=name, action=action, error=str(exc)))
        except Exception as exc:
            failures.append(dict(condition=name, error=str(exc)))
        summarize(campaign)
    write_json(campaign / "finished.json", dict(status="failed" if failures else "completed", failures=failures))
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; see {campaign / 'finished.json'}")
    report(f"CAMPAIGN COMPLETE {campaign}")


if __name__ == "__main__":
    main()
