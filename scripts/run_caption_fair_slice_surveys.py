#!/usr/bin/env python3
"""Run caption_corpus_survey.py over paired fair-slice outputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SURVEY_SCRIPT = Path(__file__).resolve().parent / "caption_corpus_survey.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Survey caption fair slices")
    parser.add_argument("--fair-root", default="data/caption-fair-slices")
    parser.add_argument("--max-pairs", type=int, default=1_000_000)
    parser.add_argument("--comparison", action="append", default=[], help="Fair-slice comparison name to include")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--parallel-entries", type=int, default=4)
    parser.add_argument("--token-budgets", default="16,32,64,128,256")
    parser.add_argument("--top-ks", default="10,50,100,500,1000,5000")
    parser.add_argument("--ngram-orders", default="1,2,3,4")
    parser.add_argument("--repeat-ngram-orders", default="3,4,5,6,8")
    parser.add_argument("--budget-tokenizer-backend", choices=["lexical", "hf"], default="lexical")
    parser.add_argument("--budget-tokenizer-name", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_jobs(fair_root: Path, max_pairs: int, seed: int, comparisons: set[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for summary_path in sorted(fair_root.glob(f"*/n{max_pairs}_seed{seed}/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if comparisons and summary["comparison"] not in comparisons:
            continue
        outputs = summary["outputs"]
        for role, key in [("local", "local_jsonl"), ("reference", "reference_jsonl")]:
            path = Path(outputs[key])
            if not path.exists():
                continue
            jobs.append(
                {
                    "comparison": summary["comparison"],
                    "family": summary.get("family"),
                    "tier": summary.get("tier"),
                    "role": role,
                    "surface": summary["local_surface"] if role == "local" else summary["reference_surface"],
                    "input": str(path),
                    "paired_count": outputs["paired_count"],
                    "summary_path": str(summary_path),
                }
            )
    return jobs


def run_job(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SURVEY_SCRIPT),
        "--input",
        job["input"],
        "--max-records",
        str(args.max_records),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--token-budgets",
        args.token_budgets,
        "--top-ks",
        args.top_ks,
        "--ngram-orders",
        args.ngram_orders,
        "--repeat-ngram-orders",
        args.repeat_ngram_orders,
        "--budget-tokenizer-backend",
        args.budget_tokenizer_backend,
    ]
    if args.budget_tokenizer_name:
        command.extend(["--budget-tokenizer-name", args.budget_tokenizer_name])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return {
        "job": job,
        "summary": json.loads(completed.stdout),
    }


def nested_get(mapping: dict[str, Any], dotted: str) -> Any:
    current: Any = mapping
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def cell(value: Any) -> str:
    return "" if value is None else str(value)


def tsv_rows(results: list[dict[str, Any]]) -> list[str]:
    header = [
        "comparison",
        "role",
        "surface",
        "family",
        "tier",
        "paired_count",
        "records",
        "avg_tokens",
        "cov64",
        "cov128",
        "cov256",
        "prefix_ev64",
        "prefix_top100_64",
        "d3_64",
        "m3_top100_64",
        "viol64",
        "rep4_64",
        "full_cb_m3_top100",
        "full_within_d3_mean",
        "full_rep4",
        "full_viol_ind_per64",
        "full_viol_codes_per64",
        "control_hits_per64_full",
    ]
    rows = ["\t".join(header)]
    for result in results:
        job = result["job"]
        summary = result["summary"]
        full_length = summary.get("full_length_reference", {})
        lc64 = summary.get("length_controlled", {}).get("64", {})
        lc128 = summary.get("length_controlled", {}).get("128", {})
        lc256 = summary.get("length_controlled", {}).get("256", {})
        values = [
            job["comparison"],
            job["role"],
            job["surface"],
            job.get("family") or "",
            job.get("tier") or "",
            str(job["paired_count"]),
            str(summary.get("captions_loaded", "")),
            str(full_length.get("avg_lexical_tokens", "")),
            str(lc64.get("coverage_rate", "")),
            str(lc128.get("coverage_rate", "")),
            str(lc256.get("coverage_rate", "")),
            cell(
                nested_get(lc64, "prefix_content.effective_vocab_miller_madow")
                or nested_get(lc64, "prefix_content.effective_vocab")
            ),
            cell(nested_get(lc64, "prefix_content.top_k_mass")),
            cell(nested_get(lc64, "distinct_n.3")),
            cell(nested_get(lc64, "ngram_top_k_mass.3")),
            cell(lc64.get("violation_rate")),
            cell(lc64.get("repeated_4gram_rate")),
            cell(nested_get(full_length, "full_caption_normalized.caption_balanced_ngram_top_k_mass.3.100")),
            cell(nested_get(full_length, "full_caption_normalized.within_caption_distinct_n_mean.3")),
            cell(nested_get(full_length, "full_caption_normalized.repeated_ngram_rate_by_n.4")),
            cell(nested_get(full_length, "full_caption_normalized.violation_indicator_per_64_lexical_tokens_mean")),
            cell(nested_get(full_length, "full_caption_normalized.violation_codes_per_64_lexical_tokens_mean")),
            cell(nested_get(full_length, "debug_control_lexicon.hits_per_64_lexical_tokens")),
        ]
        rows.append("\t".join(values))
    return rows


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(Path(args.fair_root), args.max_pairs, args.seed, set(args.comparison))
    if not jobs:
        raise SystemExit("No fair-slice jobs found")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel_entries) as executor:
        futures = [executor.submit(run_job, job, args) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: (item["job"]["comparison"], item["job"]["role"]))
    payload = {
        "fair_root": args.fair_root,
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "max_records": args.max_records,
        "budget_tokenizer_backend": args.budget_tokenizer_backend,
        "budget_tokenizer_name": args.budget_tokenizer_name,
        "job_count": len(jobs),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tsv_path = output_path.with_suffix(".tsv")
    tsv_path.write_text("\n".join(tsv_rows(results)) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "tsv": str(tsv_path), "job_count": len(jobs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
