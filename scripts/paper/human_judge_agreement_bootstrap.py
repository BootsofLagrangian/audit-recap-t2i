#!/usr/bin/env python3
"""Judge-human agreement as estimate and bootstrap standard deviation.

Runs the human-study exporter's metric computation on the exported annotation
rows with its image-cluster bootstrap (10,000 resamples, seed 1477) and keeps
the standard deviation of each bootstrap distribution next to the estimate.
Usage: human_judge_agreement_bootstrap.py <export-dir> > judge_human_agreement_bootstrap.json
"""

from __future__ import annotations

import json
import statistics
import sys

import audit_recap_t2i.human_cbu.export as export
import audit_recap_t2i.human_cbu.metrics as metrics

_original = metrics.cluster_bootstrap_percentile
KEEP = ("estimate", "lower", "upper", "bootstrap_mean", "bootstrap_std", "n_records", "n_clusters", "n_resamples", "seed")


def with_std(records, statistic, **kwargs):
    kwargs["return_distribution"] = True
    out = _original(records, statistic, **kwargs)
    distribution = out.pop("distribution")
    out["bootstrap_mean"] = statistics.fmean(distribution) if distribution else None
    out["bootstrap_std"] = statistics.stdev(distribution) if len(distribution) > 1 else None
    return out


def main() -> int:
    metrics.cluster_bootstrap_percentile = with_std
    export.cluster_bootstrap_percentile = with_std
    export_dir = sys.argv[1]
    with open(f"{export_dir}/annotations.private.jsonl", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = export.compute_export_metrics(rows, status=None, bootstrap_resamples=10_000, bootstrap_seed=1477)
    out = {}
    for judge in ("qwen", "gemma"):
        summary = result["judge_human"][judge]
        out[judge] = {
            "overall": {k: summary["overall_cluster_bootstrap_95"].get(k) for k in KEEP},
            "by_surface": [
                {"surface_group": g.get("surface_group"), **{k: g["cluster_bootstrap_95"].get(k) for k in KEEP}}
                for g in summary["by_surface"]
            ],
        }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
