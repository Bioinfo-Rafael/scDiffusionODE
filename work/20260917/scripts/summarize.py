"""Wide per-condition table plus full long-form legacy metrics; missing stays missing."""
import csv
from pathlib import Path
from ..common import CONDITIONS, confined, new_dir, read_json, run_id, write_csv


def summarize(campaign):
    campaign = confined(campaign)
    rows, metrics = [], []
    for condition in CONDITIONS:
        config = read_json(campaign / "configs" / f"{condition}.json")
        row = dict(condition=condition, model=config["model_type"], loss=config["aux_loss"], status="not_run")
        run = campaign / condition / "training"
        if (run / "failed.json").exists():
            row.update(status="failed", error=read_json(run / "failed.json")["error"])
        if (run / "completed.json").exists():
            row["status"] = "trained"
            report = read_json(run / "performance.json")
            row.update({k:v for k,v in report.items() if not isinstance(v, dict)})
            row.update(report["loss_means"])
        analyses = [p for p in sorted((campaign / condition / "sampling").glob("*/analyze/*"))
                    if (p / "completed.json").exists()]
        if analyses:
            row["status"] = "evaluated"
            for path in analyses[-1].glob("*.csv"):
                with path.open() as f:
                    for item in csv.DictReader(f):
                        metrics.append(dict(condition=condition, file=path.name, **item))
                        # Terminal diffusion endpoint, not post-ODE continuation.
                        if item.get("reverse_step") == "1000":
                            for key,value in item.items():
                                if key not in ("snapshot_index", "phase", "reverse_step"):
                                    row[path.stem+"__"+key] = value
                        if path.name == "true_x0_metrics.csv" and item.get("t") == "0" and item.get("series") == "hybrid":
                            row["true_x0_t0_"+item["metric"]] = item["mean"]
        rows.append(row)
    output = new_dir(campaign / "summaries" / run_id())
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(output / "comparison.csv", rows, fields)
    if metrics:
        fields = list(dict.fromkeys(key for row in metrics for key in row))
        write_csv(output / "all_metrics.csv", metrics, fields)
    print(f"SUMMARY={output / 'comparison.csv'}", flush=True)
    return output
