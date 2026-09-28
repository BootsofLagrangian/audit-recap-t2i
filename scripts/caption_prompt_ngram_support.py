#!/usr/bin/env python3
"""Measure caption overlap with prompt-only reference pools via hashed n-grams."""

from __future__ import annotations

import argparse
import json
import math
import re
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption/prompt hashed n-gram support metrics")
    parser.add_argument("--fair-root", default="data/caption-fair-slices")
    parser.add_argument("--max-pairs", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--role", choices=["local", "reference", "both"], default="both")
    parser.add_argument("--prompt-glob", default="data/prompt-pools/2026-04-24/*.filtered.jsonl")
    parser.add_argument("--budgets", default="64,128,full")
    parser.add_argument("--ngrams", default="1,2,3")
    parser.add_argument("--hash-buckets", type=int, default=2**18)
    parser.add_argument("--max-caption-records", type=int, default=None)
    parser.add_argument("--max-prompt-records", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1e-12)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fair_jobs(fair_root: Path, max_pairs: int, seed: int, role: str) -> list[dict[str, Any]]:
    jobs = []
    seen = set()
    for summary_path in sorted(fair_root.glob(f"*/n{max_pairs}_seed{seed}/summary.json")):
        summary = json_load(summary_path)
        outputs = summary["outputs"]
        roles = ["local", "reference"] if role == "both" else [role]
        for item_role in roles:
            path_key = "local_jsonl" if item_role == "local" else "reference_jsonl"
            surface = summary["local_surface"] if item_role == "local" else summary["reference_surface"]
            path = outputs[path_key]
            dedupe_key = (item_role, surface, path)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            jobs.append(
                {
                    "comparison": summary["comparison"],
                    "role": item_role,
                    "surface": surface,
                    "path": path,
                    "paired_count": outputs["paired_count"],
                }
            )
    return jobs


def parse_budgets(raw: str) -> list[int | None]:
    budgets: list[int | None] = []
    for item in raw.split(","):
        item = item.strip().lower()
        if item == "full":
            budgets.append(None)
        else:
            budgets.append(int(item))
    return budgets


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def hash_ngram(tokens: list[str], start: int, n: int, buckets: int) -> int:
    # crc32 is stable, fast, and sufficient for approximate distributional support.
    gram = " ".join(tokens[start : start + n]).encode("utf-8")
    return zlib.crc32(gram) % buckets


def add_ngrams(counts: dict[tuple[str, int], np.ndarray], tokens: list[str], budget_key: str, ngrams: list[int], buckets: int) -> None:
    for n in ngrams:
        if len(tokens) < n:
            continue
        arr = counts[(budget_key, n)]
        for i in range(0, len(tokens) - n + 1):
            arr[hash_ngram(tokens, i, n, buckets)] += 1


def read_jsonl_text(path: Path, field: str, max_records: int | None):
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_records is not None and count >= max_records:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get(field)
            if isinstance(text, str) and text.strip():
                count += 1
                yield text


def count_prompt_pool(path: Path, ngrams: list[int], buckets: int, max_records: int | None) -> dict[str, Any]:
    counts = {("prompt", n): np.zeros(buckets, dtype=np.int64) for n in ngrams}
    records = 0
    token_total = 0
    for prompt in read_jsonl_text(path, "prompt", max_records):
        tokens = tokenize(prompt)
        if not tokens:
            continue
        records += 1
        token_total += len(tokens)
        add_ngrams(counts, tokens, "prompt", ngrams, buckets)
    return {
        "name": path.name.removesuffix(".filtered.jsonl").removesuffix(".prompt_only.jsonl"),
        "path": str(path),
        "records": records,
        "mean_tokens": token_total / records if records else 0.0,
        "counts": {n: counts[("prompt", n)] for n in ngrams},
    }


def count_caption_surface(job: dict[str, Any]) -> dict[str, Any]:
    budgets: list[int | None] = job["budgets"]
    ngrams: list[int] = job["ngrams"]
    buckets: int = job["hash_buckets"]
    counts = {
        (budget_label(budget), n): np.zeros(buckets, dtype=np.int64)
        for budget in budgets
        for n in ngrams
    }
    records = 0
    eligible = {budget_label(budget): 0 for budget in budgets}
    token_total = 0
    for caption in read_jsonl_text(Path(job["path"]), "caption", job["max_records"]):
        tokens = tokenize(caption)
        if not tokens:
            continue
        records += 1
        token_total += len(tokens)
        for budget in budgets:
            label = budget_label(budget)
            use_tokens = tokens if budget is None else tokens[:budget]
            if budget is None or len(tokens) >= budget:
                eligible[label] += 1
            add_ngrams(counts, use_tokens, label, ngrams, buckets)
    return {
        "comparison": job["comparison"],
        "role": job["role"],
        "surface": job["surface"],
        "path": job["path"],
        "paired_count": job["paired_count"],
        "records": records,
        "mean_tokens": token_total / records if records else 0.0,
        "eligible": eligible,
        "counts": {f"{label}:{n}": arr for (label, n), arr in counts.items()},
    }


def budget_label(budget: int | None) -> str:
    return "full" if budget is None else str(budget)


def distribution(counts: np.ndarray, alpha: float) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return np.full_like(counts, 1.0 / len(counts), dtype=np.float64)
    probs = counts.astype(np.float64, copy=True)
    if alpha > 0:
        probs += alpha
    probs /= probs.sum()
    return probs


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def compare_counts(caption_counts: np.ndarray, prompt_counts: np.ndarray, alpha: float) -> dict[str, float]:
    caption_probs = distribution(caption_counts, alpha)
    prompt_probs = distribution(prompt_counts, alpha)
    midpoint = 0.5 * (caption_probs + prompt_probs)
    caption_support = caption_counts > 0
    prompt_support = prompt_counts > 0
    return {
        "caption_total_ngrams": float(caption_counts.sum()),
        "prompt_total_ngrams": float(prompt_counts.sum()),
        "caption_unique_buckets": float(np.count_nonzero(caption_support)),
        "prompt_unique_buckets": float(np.count_nonzero(prompt_support)),
        "overlap_buckets": float(np.count_nonzero(caption_support & prompt_support)),
        "caption_mass_on_prompt_support": float(caption_probs[prompt_support].sum()),
        "prompt_mass_on_caption_support": float(prompt_probs[caption_support].sum()),
        "kl_caption_to_prompt": kl_div(caption_probs, prompt_probs),
        "kl_prompt_to_caption": kl_div(prompt_probs, caption_probs),
        "jsd": 0.5 * kl_div(caption_probs, midpoint) + 0.5 * kl_div(prompt_probs, midpoint),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "prompt_pool",
        "comparison",
        "role",
        "surface",
        "records",
        "caption_mean_tokens",
        "budget",
        "budget_eligible_rate",
        "ngram_n",
        "caption_total_ngrams",
        "prompt_total_ngrams",
        "caption_unique_buckets",
        "prompt_unique_buckets",
        "overlap_buckets",
        "caption_mass_on_prompt_support",
        "prompt_mass_on_caption_support",
        "jsd",
        "kl_caption_to_prompt",
        "kl_prompt_to_caption",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


def main() -> int:
    args = parse_args()
    budgets = parse_budgets(args.budgets)
    ngrams = parse_ints(args.ngrams)
    prompt_paths = sorted(Path().glob(args.prompt_glob))
    if not prompt_paths:
        raise SystemExit(f"no prompt pools matched {args.prompt_glob}")
    prompt_pools = [
        count_prompt_pool(path, ngrams, args.hash_buckets, args.max_prompt_records)
        for path in prompt_paths
    ]
    caption_jobs = fair_jobs(Path(args.fair_root), args.max_pairs, args.seed, args.role)
    worker_jobs = [
        {
            **job,
            "budgets": budgets,
            "ngrams": ngrams,
            "hash_buckets": args.hash_buckets,
            "max_records": args.max_caption_records,
        }
        for job in caption_jobs
    ]
    caption_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(count_caption_surface, job) for job in worker_jobs]
        for future in as_completed(futures):
            caption_results.append(future.result())
    rows = []
    for caption in caption_results:
        for pool in prompt_pools:
            for budget in budgets:
                label = budget_label(budget)
                eligible = caption["eligible"][label] / caption["records"] if caption["records"] else 0.0
                for n in ngrams:
                    metrics = compare_counts(caption["counts"][f"{label}:{n}"], pool["counts"][n], args.alpha)
                    rows.append(
                        {
                            "prompt_pool": pool["name"],
                            "prompt_pool_path": pool["path"],
                            "prompt_pool_records": pool["records"],
                            "prompt_pool_mean_tokens": pool["mean_tokens"],
                            "comparison": caption["comparison"],
                            "role": caption["role"],
                            "surface": caption["surface"],
                            "path": caption["path"],
                            "paired_count": caption["paired_count"],
                            "records": caption["records"],
                            "caption_mean_tokens": caption["mean_tokens"],
                            "budget": label,
                            "budget_eligible_rate": eligible,
                            "ngram_n": n,
                            **metrics,
                        }
                    )
    rows.sort(key=lambda row: (row["prompt_pool"], row["surface"], row["role"], str(row["budget"]), row["ngram_n"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fair_root": args.fair_root,
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "role": args.role,
        "prompt_glob": args.prompt_glob,
        "budgets": [budget_label(budget) for budget in budgets],
        "ngrams": ngrams,
        "hash_buckets": args.hash_buckets,
        "max_caption_records": args.max_caption_records,
        "max_prompt_records": args.max_prompt_records,
        "alpha": args.alpha,
        "prompt_pools": [
            {key: value for key, value in pool.items() if key != "counts"}
            for pool in prompt_pools
        ],
        "results": rows,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_tsv(output.with_suffix(".tsv"), rows)
    print(json.dumps({"output": str(output), "tsv": str(output.with_suffix(".tsv")), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
