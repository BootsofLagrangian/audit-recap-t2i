#!/usr/bin/env python3
"""Repeated prompt-support estimates from disjoint caption pools.

The script consumes prepared prompt-pool JSONL files whose prompt text has
already been normalized into a top-level ``prompt`` field. Raw upstream prompt
column names are resolved by ``scripts/prepare_prompt_pools.py`` before this
metric runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-support estimates over disjoint caption pools")
    parser.add_argument("--fair-root", default="data/caption-fair-slices")
    parser.add_argument("--max-pairs", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-glob", default="data/prompt-pools/2026-04-24-expanded/*.filtered.jsonl")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--ngram", type=int, default=2)
    parser.add_argument("--hash-buckets", type=int, default=2**18)
    parser.add_argument("--pool-count", type=int, default=3)
    parser.add_argument("--pool-fraction", type=float, default=0.25)
    parser.add_argument("--max-prompt-records", type=int, default=1_000_000)
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


def metric_row(counts: np.ndarray, prompt_counts: np.ndarray, alpha: float) -> dict[str, float]:
    p = distribution(counts, alpha)
    q = distribution(prompt_counts, alpha)
    m = 0.5 * (p + q)
    support = counts > 0
    prompt_support = prompt_counts > 0
    return {
        "jsd": 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m),
        "prompt_mass_on_caption_support": float(q[support].sum()),
        "caption_mass_on_prompt_support": float(p[prompt_support].sum()),
        "total_ngrams": float(counts.sum()),
        "unique_buckets": float(np.count_nonzero(support)),
    }


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


def pool_index(comparison: str, row: dict[str, Any], seed: int, pool_count: int, pool_fraction: float) -> int | None:
    key = row.get("pair_key") or row.get("pair_id") or row.get("public_lookup_key") or row.get("metadata", {}).get("stable_image_id")
    if key is None:
        return None
    raw = hashlib.blake2b(f"{seed}:{comparison}:{key}".encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(raw, "big") / 2**64
    width = pool_fraction
    for index in range(pool_count):
        low = index * width
        high = low + width
        if low <= value < high:
            return index
    return None


def count_caption_pools(summary: dict[str, Any], budget: int, n: int, buckets: int, pool_count: int, pool_fraction: float, seed: int) -> list[dict[str, Any]]:
    pools = [
        {
            "pool_index": index,
            "records": 0,
            "local_counts": np.zeros(buckets, dtype=np.int64),
            "reference_counts": np.zeros(buckets, dtype=np.int64),
        }
        for index in range(pool_count)
    ]
    pairs_path = Path(summary["outputs"]["pairs_jsonl"])
    with pairs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index = pool_index(summary["comparison"], row, seed, pool_count, pool_fraction)
            if index is None:
                continue
            pool = pools[index]
            local_tokens = tokenize(row.get("local_caption", ""))[:budget]
            reference_tokens = tokenize(row.get("reference_caption", ""))[:budget]
            add_ngrams(pool["local_counts"], local_tokens, n)
            add_ngrams(pool["reference_counts"], reference_tokens, n)
            pool["records"] += 1
    return pools


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "prompt_pool",
        "comparison",
        "local_surface",
        "reference_surface",
        "pool_index",
        "records",
        "role",
        "jsd",
        "prompt_mass_on_caption_support",
        "caption_mass_on_prompt_support",
        "total_ngrams",
        "unique_buckets",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


def main() -> int:
    args = parse_args()
    if args.pool_count * args.pool_fraction > 1.0:
        raise SystemExit("--pool-count * --pool-fraction must be <= 1.0 for disjoint pools")
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
        "pool_count": args.pool_count,
        "pool_fraction": args.pool_fraction,
        "max_prompt_records": args.max_prompt_records,
        "method": "deterministic disjoint hash pools over paired image/url rows; aggregate counts within each pool before computing JSD",
        "prompt_pools": {
            name: {"records": int(records), "total_ngrams": float(counts.sum())}
            for name, (counts, records) in prompt_counts.items()
        },
        "results": rows,
    }

    for summary in fair_summaries(Path(args.fair_root), args.max_pairs, args.seed):
        pools = count_caption_pools(
            summary,
            args.budget,
            args.ngram,
            args.hash_buckets,
            args.pool_count,
            args.pool_fraction,
            args.seed,
        )
        for prompt_name, (prompt_arr, _prompt_records) in prompt_counts.items():
            for pool in pools:
                for role, key in [("local", "local_counts"), ("reference", "reference_counts")]:
                    metrics = metric_row(pool[key], prompt_arr, args.alpha)
                    rows.append(
                        {
                            "prompt_pool": prompt_name,
                            "comparison": summary["comparison"],
                            "local_surface": summary["local_surface"],
                            "reference_surface": summary["reference_surface"],
                            "pool_index": pool["pool_index"],
                            "records": pool["records"],
                            "role": role,
                            **metrics,
                        }
                    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_tsv(output.with_suffix(".tsv"), rows)
    print(json.dumps({"output": str(output), "tsv": str(output.with_suffix(".tsv")), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
