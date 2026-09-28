#!/usr/bin/env python3
"""Block-bootstrap prompt-support deltas for paired recap fair slices."""

from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-support block bootstrap for paired fair slices")
    parser.add_argument("--fair-root", default="data/caption-fair-slices")
    parser.add_argument("--max-pairs", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-glob", default="data/prompt-pools/2026-04-24-expanded/*.filtered.jsonl")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--ngram", type=int, default=2)
    parser.add_argument("--hash-buckets", type=int, default=2**18)
    parser.add_argument("--block-size", type=int, default=5000)
    parser.add_argument("--max-caption-records", type=int, default=250000)
    parser.add_argument("--max-prompt-records", type=int, default=1000000)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=1e-12)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def hash_ngram(tokens: list[str], start: int, n: int, buckets: int) -> int:
    return zlib.crc32(" ".join(tokens[start : start + n]).encode("utf-8")) % buckets


def add_ngrams(arr: np.ndarray, tokens: list[str], n: int) -> None:
    if len(tokens) < n:
        return
    buckets = len(arr)
    for index in range(0, len(tokens) - n + 1):
        arr[hash_ngram(tokens, index, n, buckets)] += 1


def distribution(counts: np.ndarray, alpha: float) -> np.ndarray:
    probs = counts.astype(np.float64, copy=True)
    if probs.sum() <= 0:
        return np.full(len(probs), 1.0 / len(probs), dtype=np.float64)
    if alpha > 0:
        probs += alpha
    probs /= probs.sum()
    return probs


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def jsd(counts: np.ndarray, prompt_counts: np.ndarray, alpha: float) -> float:
    p = distribution(counts, alpha)
    q = distribution(prompt_counts, alpha)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def prompt_mass_on_caption_support(counts: np.ndarray, prompt_counts: np.ndarray, alpha: float) -> float:
    prompt_probs = distribution(prompt_counts, alpha)
    return float(prompt_probs[counts > 0].sum())


def read_jsonl_field(path: Path, field: str, max_records: int | None):
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


def count_prompt(path: Path, n: int, buckets: int, max_records: int | None) -> tuple[np.ndarray, int]:
    counts = np.zeros(buckets, dtype=np.int64)
    records = 0
    for prompt in read_jsonl_field(path, "prompt", max_records):
        records += 1
        add_ngrams(counts, tokenize(prompt), n)
    return counts, records


def fair_summaries(fair_root: Path, max_pairs: int, seed: int) -> list[dict[str, Any]]:
    summaries = []
    for summary_path in sorted(fair_root.glob(f"*/n{max_pairs}_seed{seed}/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["_summary_path"] = str(summary_path)
        summaries.append(summary)
    return summaries


def block_counts(summary: dict[str, Any], budget: int, n: int, buckets: int, block_size: int, max_records: int | None) -> list[dict[str, Any]]:
    pairs_path = Path(summary["outputs"]["pairs_jsonl"])
    blocks = []
    local_counts = np.zeros(buckets, dtype=np.int64)
    reference_counts = np.zeros(buckets, dtype=np.int64)
    records = 0
    block_index = 0

    def flush() -> None:
        nonlocal local_counts, reference_counts, records, block_index
        if records == 0:
            return
        blocks.append(
            {
                "block_index": block_index,
                "records": records,
                "local_counts": local_counts,
                "reference_counts": reference_counts,
            }
        )
        block_index += 1
        local_counts = np.zeros(buckets, dtype=np.int64)
        reference_counts = np.zeros(buckets, dtype=np.int64)
        records = 0

    with pairs_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if max_records is not None and line_number >= max_records:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            local_tokens = tokenize(row.get("local_caption", ""))[:budget]
            reference_tokens = tokenize(row.get("reference_caption", ""))[:budget]
            add_ngrams(local_counts, local_tokens, n)
            add_ngrams(reference_counts, reference_tokens, n)
            records += 1
            if records >= block_size:
                flush()
    flush()
    return blocks


def bootstrap_ci(values: np.ndarray, reps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "bootstrap_reps": reps}
    if n == 1:
        value = float(values[0])
        return {"mean": value, "ci95_low": value, "ci95_high": value, "bootstrap_reps": reps}
    indices = rng.integers(0, n, size=(reps, n))
    sampled = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "bootstrap_reps": reps,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "prompt_pool",
        "comparison",
        "local_surface",
        "reference_surface",
        "metric",
        "blocks",
        "block_size",
        "records",
        "delta_mean_local_minus_reference",
        "ci95_low",
        "ci95_high",
        "stable_direction",
        "local_block_mean",
        "reference_block_mean",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


def main() -> int:
    args = parse_args()
    prompt_paths = sorted(Path().glob(args.prompt_glob))
    if not prompt_paths:
        raise SystemExit(f"no prompt pools matched {args.prompt_glob}")
    prompt_counts = {
        path.name.removesuffix(".filtered.jsonl"): count_prompt(path, args.ngram, args.hash_buckets, args.max_prompt_records)
        for path in prompt_paths
    }

    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "fair_root": args.fair_root,
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "prompt_glob": args.prompt_glob,
        "budget": args.budget,
        "ngram": args.ngram,
        "hash_buckets": args.hash_buckets,
        "block_size": args.block_size,
        "max_caption_records": args.max_caption_records,
        "max_prompt_records": args.max_prompt_records,
        "bootstrap_reps": args.bootstrap_reps,
        "method": "paired contiguous-block bootstrap over block-level metric deltas; prompt pools are fixed references",
        "prompt_pools": {
            name: {"records": int(records), "total_ngrams": float(counts.sum())}
            for name, (counts, records) in prompt_counts.items()
        },
        "results": [],
    }

    for summary in fair_summaries(Path(args.fair_root), args.max_pairs, args.seed):
        blocks = block_counts(summary, args.budget, args.ngram, args.hash_buckets, args.block_size, args.max_caption_records)
        records = sum(block["records"] for block in blocks)
        for prompt_name, (prompt_arr, prompt_records) in prompt_counts.items():
            local_jsd = np.array([jsd(block["local_counts"], prompt_arr, args.alpha) for block in blocks], dtype=np.float64)
            reference_jsd = np.array([jsd(block["reference_counts"], prompt_arr, args.alpha) for block in blocks], dtype=np.float64)
            local_mass = np.array(
                [prompt_mass_on_caption_support(block["local_counts"], prompt_arr, args.alpha) for block in blocks],
                dtype=np.float64,
            )
            reference_mass = np.array(
                [prompt_mass_on_caption_support(block["reference_counts"], prompt_arr, args.alpha) for block in blocks],
                dtype=np.float64,
            )
            metrics = {
                "jsd": local_jsd - reference_jsd,
                "prompt_mass_on_caption_support": local_mass - reference_mass,
            }
            for metric_name, delta_values in metrics.items():
                ci = bootstrap_ci(
                    delta_values,
                    args.bootstrap_reps,
                    seed=abs(hash((args.seed, summary["comparison"], prompt_name, metric_name))) % (2**32),
                )
                if metric_name == "jsd":
                    stable_direction = "local_lower" if ci["ci95_high"] < 0 else "reference_lower" if ci["ci95_low"] > 0 else "inconclusive"
                    local_mean = float(local_jsd.mean())
                    reference_mean = float(reference_jsd.mean())
                else:
                    stable_direction = "local_higher" if ci["ci95_low"] > 0 else "reference_higher" if ci["ci95_high"] < 0 else "inconclusive"
                    local_mean = float(local_mass.mean())
                    reference_mean = float(reference_mass.mean())
                row = {
                    "prompt_pool": prompt_name,
                    "prompt_records": prompt_records,
                    "comparison": summary["comparison"],
                    "local_surface": summary["local_surface"],
                    "reference_surface": summary["reference_surface"],
                    "metric": metric_name,
                    "blocks": len(blocks),
                    "block_size": args.block_size,
                    "records": records,
                    "delta_mean_local_minus_reference": ci["mean"],
                    "ci95_low": ci["ci95_low"],
                    "ci95_high": ci["ci95_high"],
                    "stable_direction": stable_direction,
                    "local_block_mean": local_mean,
                    "reference_block_mean": reference_mean,
                }
                rows.append(row)
                payload["results"].append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_tsv(output.with_suffix(".tsv"), rows)
    print(json.dumps({"output": str(output), "tsv": str(output.with_suffix(".tsv")), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
