#!/usr/bin/env python3
"""Summarize remaining CPU-only fair-slice diagnostics.

The script reads paired fair-slice `pairs.jsonl` files and emits:
- per-code violation rates copied from existing survey summaries
- paired-difference normal CIs for cheap per-row diagnostics
- a small paired manifest for CPU sanity checks of later GPU metrics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
LEADING_STOPWORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "image",
    "picture",
    "photo",
    "photograph",
    "illustration",
    "artwork",
    "scene",
    "shows",
    "showing",
    "depicts",
    "depicting",
    "features",
    "featuring",
    "presents",
    "displaying",
    "displays",
    "contains",
    "captures",
    "we",
    "can",
    "see",
    "in",
}
VIOLATION_CODES = [
    "D_markdown_leak",
    "E_thinking_leak",
    "F_repetition",
    "G_meta_instruction",
    "H_detection_meta_leak",
    "I_overlay_meta_leak",
    "J_meta_statement",
    "L_tag_verbatim",
    "M_numbered_list_leak",
    "N_stage_header_leak",
]
J_META_RE = re.compile(
    r"^(the|this) (image|picture|photo|illustration|artwork|scene) "
    r"(shows?|depicts?|features?|presents?|displays?|contains?|captures?)|"
    r"^in this (image|picture|photo|illustration)|"
    r"^we (can )?see ",
    re.IGNORECASE,
)
I_OVERLAY_RE = re.compile(r"overlay|annotation layer|green rectangle|labeled box", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU residual diagnostics for recap fair slices")
    parser.add_argument("--fair-root", default="data/caption-fair-slices")
    parser.add_argument("--max-pairs", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="artifacts/caption-survey/cpu_remaining_2026-04-24",
        help="Ignored artifact directory for generated summaries",
    )
    parser.add_argument("--sanity-per-comparison", type=int, default=512)
    parser.add_argument("--token-budget", type=int, default=64)
    parser.add_argument("--prefix-tokens", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--comparison", action="append", default=[], help="Optional comparison name filter")
    parser.add_argument(
        "--survey-json",
        action="append",
        default=[],
        help="Survey JSON for violation breakdown. Later files override duplicate comparisons.",
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def truncate_tokens(text: str, budget: int) -> list[str]:
    matches = list(TOKEN_RE.finditer(text))
    if len(matches) <= budget:
        return tokenize(text)
    prefix = text[: matches[budget - 1].end()]
    return tokenize(prefix)


def content_prefix(tokens: list[str], prefix_tokens: int) -> str:
    trimmed = list(tokens)
    while trimmed and trimmed[0] in LEADING_STOPWORDS:
        trimmed.pop(0)
    return " ".join(trimmed[:prefix_tokens])


def ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]


def has_repeated_ngram(tokens: list[str], n: int = 4) -> float:
    grams = ngrams(tokens, n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    return 1.0 if any(count >= 2 for count in counts.values()) else 0.0


def fast_residue_codes(text: str) -> set[str]:
    codes: set[str] = set()
    if J_META_RE.search(text):
        codes.add("J_meta_statement")
    if I_OVERLAY_RE.search(text):
        codes.add("I_overlay_meta_leak")
    return codes


def load_violation_rates(survey_paths: list[str]) -> dict[tuple[str, str], dict[str, float]]:
    rates: dict[tuple[str, str], dict[str, float]] = {}
    for raw_path in survey_paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("results", []):
            job = item.get("job", {})
            comparison = job.get("comparison")
            role = job.get("role")
            if not comparison or not role:
                continue
            lc64 = item.get("summary", {}).get("length_controlled", {}).get("64", {})
            code_rate = lc64.get("violation_code_rate", {})
            if isinstance(code_rate, dict):
                rates[(comparison, role)] = {str(key): float(value) for key, value in code_rate.items()}
    return rates


def find_pair_summaries(fair_root: Path, max_pairs: int, seed: int) -> list[Path]:
    return sorted(fair_root.glob(f"*/n{max_pairs}_seed{seed}/summary.json"))


def read_pairs(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normal_ci(values: list[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": n}
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    half = 1.96 * math.sqrt(variance / n)
    return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half, "n": n}


def write_tsv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


def extract_image_payload(image: Any) -> dict[str, Any]:
    if not isinstance(image, dict):
        return {}
    return {
        "url": image.get("url") or image.get("canonical_url"),
        "normalized_url": image.get("normalized_url") or image.get("canonical_url"),
        "local_abs_path": image.get("local_abs_path"),
        "local_file_exists": image.get("local_file_exists"),
        "local_file_bytes": image.get("local_file_bytes") or image.get("bytes"),
        "width": image.get("width"),
        "height": image.get("height"),
        "sha256": image.get("sha256"),
        "stable_image_id": image.get("stable_image_id"),
        "sample_id": image.get("sample_id"),
        "dataset_key": image.get("dataset_key"),
    }


def row_for_sanity(row: dict[str, Any], local_surface: str, reference_surface: str) -> dict[str, Any]:
    return {
        "comparison": row.get("comparison"),
        "pair_id": row.get("pair_id") or row.get("pair_key"),
        "local_surface": local_surface,
        "reference_surface": reference_surface,
        "image": extract_image_payload(row.get("image") or row.get("metadata") or {}),
        "local_caption": row.get("local_caption", ""),
        "reference_caption": row.get("reference_caption", ""),
    }


def summarize_one(payload: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(payload["summary_path"])
    args = payload["args"]
    violation_rates = payload["violation_rates"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pairs_path = Path(summary["outputs"]["pairs_jsonl"])
    comparison = summary["comparison"]
    local_surface = summary["local_surface"]
    reference_surface = summary["reference_surface"]

    local_prefix_counter: Counter[str] = Counter()
    reference_prefix_counter: Counter[str] = Counter()
    local_m3_counter: Counter[str] = Counter()
    reference_m3_counter: Counter[str] = Counter()
    sanity_reservoir: list[dict[str, Any]] = []
    rng = random.Random(args["seed"] + abs(hash(comparison)) % 1_000_000)
    n_pairs = 0

    for row in read_pairs(pairs_path):
        local_caption = row.get("local_caption", "")
        reference_caption = row.get("reference_caption", "")
        local_tokens = truncate_tokens(local_caption, args["token_budget"])
        reference_tokens = truncate_tokens(reference_caption, args["token_budget"])
        local_prefix_counter[content_prefix(local_tokens, args["prefix_tokens"])] += 1
        reference_prefix_counter[content_prefix(reference_tokens, args["prefix_tokens"])] += 1
        local_m3_counter.update(ngrams(local_tokens, 3))
        reference_m3_counter.update(ngrams(reference_tokens, 3))
        n_pairs += 1
        if len(sanity_reservoir) < args["sanity_per_comparison"]:
            sanity_reservoir.append(row_for_sanity(row, local_surface, reference_surface))
        else:
            index = rng.randrange(n_pairs)
            if index < args["sanity_per_comparison"]:
                sanity_reservoir[index] = row_for_sanity(row, local_surface, reference_surface)

    local_top_prefix = {prefix for prefix, _ in local_prefix_counter.most_common(args["top_k"])}
    reference_top_prefix = {prefix for prefix, _ in reference_prefix_counter.most_common(args["top_k"])}
    local_top_m3 = {gram for gram, _ in local_m3_counter.most_common(args["top_k"])}
    reference_top_m3 = {gram for gram, _ in reference_m3_counter.most_common(args["top_k"])}

    metric_deltas: dict[str, list[float]] = {
        "avg_lexical_tokens": [],
        "elig64": [],
        "elig128": [],
        "rep4_at64": [],
        "fast_full_j_meta_statement": [],
        "fast_full_i_overlay_meta_leak": [],
        "content_prefix_top100_hit": [],
        "m3_top100_fraction_at64": [],
    }

    for row in read_pairs(pairs_path):
        local_caption = row.get("local_caption", "")
        reference_caption = row.get("reference_caption", "")
        local_tokens_full = tokenize(local_caption)
        reference_tokens_full = tokenize(reference_caption)
        local_tokens = truncate_tokens(local_caption, args["token_budget"])
        reference_tokens = truncate_tokens(reference_caption, args["token_budget"])
        local_prefix = content_prefix(local_tokens, args["prefix_tokens"])
        reference_prefix = content_prefix(reference_tokens, args["prefix_tokens"])
        local_m3 = ngrams(local_tokens, 3)
        reference_m3 = ngrams(reference_tokens, 3)
        local_residue = fast_residue_codes(local_caption)
        reference_residue = fast_residue_codes(reference_caption)

        metric_deltas["avg_lexical_tokens"].append(len(local_tokens_full) - len(reference_tokens_full))
        metric_deltas["elig64"].append(
            (1.0 if len(local_tokens_full) >= 64 else 0.0)
            - (1.0 if len(reference_tokens_full) >= 64 else 0.0)
        )
        metric_deltas["elig128"].append(
            (1.0 if len(local_tokens_full) >= 128 else 0.0)
            - (1.0 if len(reference_tokens_full) >= 128 else 0.0)
        )
        metric_deltas["rep4_at64"].append(
            has_repeated_ngram(local_tokens, 4) - has_repeated_ngram(reference_tokens, 4)
        )
        metric_deltas["fast_full_j_meta_statement"].append(
            (1.0 if "J_meta_statement" in local_residue else 0.0)
            - (1.0 if "J_meta_statement" in reference_residue else 0.0)
        )
        metric_deltas["fast_full_i_overlay_meta_leak"].append(
            (1.0 if "I_overlay_meta_leak" in local_residue else 0.0)
            - (1.0 if "I_overlay_meta_leak" in reference_residue else 0.0)
        )
        metric_deltas["content_prefix_top100_hit"].append(
            (1.0 if local_prefix in local_top_prefix else 0.0)
            - (1.0 if reference_prefix in reference_top_prefix else 0.0)
        )
        local_m3_fraction = sum(1 for gram in local_m3 if gram in local_top_m3) / len(local_m3) if local_m3 else 0.0
        reference_m3_fraction = (
            sum(1 for gram in reference_m3 if gram in reference_top_m3) / len(reference_m3)
            if reference_m3
            else 0.0
        )
        metric_deltas["m3_top100_fraction_at64"].append(local_m3_fraction - reference_m3_fraction)

    violation_rows: list[dict[str, Any]] = []
    local_code_rates = violation_rates.get((comparison, "local"), {})
    reference_code_rates = violation_rates.get((comparison, "reference"), {})
    for code in VIOLATION_CODES:
        local_rate = local_code_rates.get(code, 0.0)
        reference_rate = reference_code_rates.get(code, 0.0)
        if local_rate or reference_rate:
            violation_rows.append(
                {
                    "comparison": comparison,
                    "local_surface": local_surface,
                    "reference_surface": reference_surface,
                    "records": n_pairs,
                    "code": code,
                    "local_rate": round(local_rate, 6),
                    "reference_rate": round(reference_rate, 6),
                    "delta_local_minus_reference": round(local_rate - reference_rate, 6),
                    "polish_priority": "yes"
                    if code in {"J_meta_statement", "D_markdown_leak", "E_thinking_leak", "G_meta_instruction", "M_numbered_list_leak"}
                    else "review",
                }
            )

    ci_rows: list[dict[str, Any]] = []
    for metric, deltas in metric_deltas.items():
        ci = normal_ci(deltas)
        ci_rows.append(
            {
                "comparison": comparison,
                "local_surface": local_surface,
                "reference_surface": reference_surface,
                "records": n_pairs,
                "metric": metric,
                "delta_mean_local_minus_reference": round(ci["mean"], 6),
                "ci95_low": round(ci["ci_low"], 6),
                "ci95_high": round(ci["ci_high"], 6),
            }
        )

    return {
        "comparison_summary": {
            "comparison": comparison,
            "records": n_pairs,
            "local_surface": local_surface,
            "reference_surface": reference_surface,
            "local_top_prefix_examples": local_prefix_counter.most_common(10),
            "reference_top_prefix_examples": reference_prefix_counter.most_common(10),
        },
        "violation_rows": violation_rows,
        "ci_rows": ci_rows,
        "sanity_rows": sanity_reservoir,
    }


def main() -> int:
    args = parse_args()
    fair_root = Path(args.fair_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_comparisons = set(args.comparison)
    summary_paths = []
    for summary_path in find_pair_summaries(fair_root, args.max_pairs, args.seed):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if selected_comparisons and summary["comparison"] not in selected_comparisons:
            continue
        summary_paths.append(summary_path)

    worker_args = {
        "seed": args.seed,
        "sanity_per_comparison": args.sanity_per_comparison,
        "token_budget": args.token_budget,
        "prefix_tokens": args.prefix_tokens,
        "top_k": args.top_k,
    }
    violation_rates = load_violation_rates(args.survey_json)
    jobs = [
        {"summary_path": str(path), "args": worker_args, "violation_rates": violation_rates}
        for path in summary_paths
    ]

    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(summarize_one, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["comparison_summary"]["comparison"])
    violation_rows = [row for item in results for row in item["violation_rows"]]
    ci_rows = [row for item in results for row in item["ci_rows"]]
    sanity_rows = [row for item in results for row in item["sanity_rows"]]

    summary_payload = {
        "fair_root": str(fair_root),
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "token_budget": args.token_budget,
        "prefix_tokens": args.prefix_tokens,
        "top_k": args.top_k,
        "workers": args.workers,
        "comparisons": [item["comparison_summary"] for item in results],
    }

    write_tsv(
        output_dir / "violation_code_breakdown.tsv",
        violation_rows,
        [
            "comparison",
            "local_surface",
            "reference_surface",
            "records",
            "code",
            "local_rate",
            "reference_rate",
            "delta_local_minus_reference",
            "polish_priority",
        ],
    )
    write_tsv(
        output_dir / "paired_delta_ci.tsv",
        ci_rows,
        [
            "comparison",
            "local_surface",
            "reference_surface",
            "records",
            "metric",
            "delta_mean_local_minus_reference",
            "ci95_low",
            "ci95_high",
        ],
    )
    with (output_dir / "gpu_sanity_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in sanity_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "comparisons": len(results),
                "violation_rows": len(violation_rows),
                "ci_rows": len(ci_rows),
                "sanity_rows": len(sanity_rows),
                "workers": args.workers,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
