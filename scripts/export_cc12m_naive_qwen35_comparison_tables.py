#!/usr/bin/env python3
"""Export comparison tables for the CC12M naive Qwen35 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import caption_corpus_survey as corpus_survey

TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)

DEFAULT_REF_VQA = Path(
    "artifacts/vqa-cbu/cc12m-four-caption-llava-url-bridge-5k-local-crossjudge-gemma4/"
    "cbu_vqa_cc12m_four_caption_b64_4494.responses.gemma4_31b_it_c512_mt2048.summary.json"
)
DEFAULT_NAIVE_VQA = Path(
    "artifacts/vqa-cbu/cc12m-naive-qwen35-baseline-2026-05-01/"
    "cbu_vqa_naive_qwen35_cc12m_b64_4494.responses.gemma4_31b_it_c64_file_mt2048.summary.json"
)
DEFAULT_REF_LONGCLIP = Path(
    "artifacts/longclip/cc12m-four-caption-llava-url-bridge-5k-local/"
    "longclip_retrieval_summary.tsv"
)
DEFAULT_NAIVE_LONGCLIP = Path(
    "artifacts/longclip/cc12m-naive-qwen35-baseline-2026-05-01/"
    "longclip_retrieval_summary.tsv"
)
DEFAULT_REF_CAPTIONS = Path(
    "artifacts/cbu/cc12m-four-caption-llava-url-bridge-5k-local/"
    "claimed_cbu_v2_cc12m_four_caption_llava_url_bridge_b64_4494.requests.jsonl"
)
DEFAULT_NAIVE_CAPTIONS = Path(
    "artifacts/recap-ed/cc12m-naive-qwen35-baseline-2026-05-01/"
    "naive_qwen35_cc12m.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/recap-ed/cc12m-naive-qwen35-baseline-2026-05-01/comparison_tables"
)

SURFACE_ORDER = [
    "ours_cc12m",
    "ref_cc12m_qwen3vl8b",
    "ref_cc12m_llavanext",
    "ref_pixelprose_cc12m",
    "naive_qwen35_cc12m",
]
LONGCLIP_NAME_MAP = {
    "ours": "ours_cc12m",
    "qwen3vl8b": "ref_cc12m_qwen3vl8b",
    "llavanext": "ref_cc12m_llavanext",
    "pixelprose": "ref_pixelprose_cc12m",
    "naive_qwen35_cc12m": "naive_qwen35_cc12m",
}
STOPWORDS = {
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
    "is",
    "are",
    "of",
}
SUSPICIOUS_PHRASES = [
    "this is",
    "overall",
    "appears to",
    "likely",
    "the image",
    "the scene",
    "capturing",
    "detailed",
    "vibrant",
    "cinematic",
    "inviting",
    "evoking",
]
SURVEY_BUDGETS = [16, 32, 64, 128, 248, 256, 320]
SURVEY_NGRAM_ORDERS = [1, 2, 3]
SURVEY_REPEAT_NGRAM_ORDERS = [3, 4, 5, 6]
SURVEY_TOP_KS = [10, 100, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-vqa", type=Path, default=DEFAULT_REF_VQA)
    parser.add_argument("--naive-vqa", type=Path, default=DEFAULT_NAIVE_VQA)
    parser.add_argument("--ref-longclip", type=Path, default=DEFAULT_REF_LONGCLIP)
    parser.add_argument("--naive-longclip", type=Path, default=DEFAULT_NAIVE_LONGCLIP)
    parser.add_argument("--ref-captions", type=Path, default=DEFAULT_REF_CAPTIONS)
    parser.add_argument("--naive-captions", type=Path, default=DEFAULT_NAIVE_CAPTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def surface_rank(surface: str) -> int:
    try:
        return SURFACE_ORDER.index(surface)
    except ValueError:
        return len(SURFACE_ORDER)


def load_vqa_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for surface, stats in data.get("surfaces", {}).items():
            rows.append(
                {
                    "surface": surface,
                    "responses": int(stats.get("responses", 0)),
                    "ok": int(stats.get("ok", 0)),
                    "questions": int(stats.get("questions", 0)),
                    "support": float(stats.get("support_rate", 0.0)),
                    "risk": float(stats.get("risk_rate", 0.0)),
                    "uncertain": float(stats.get("uncertainty_rate", 0.0)),
                    "source": str(path),
                }
            )
    return sorted(rows, key=lambda row: (surface_rank(row["surface"]), row["surface"]))


def load_longclip_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                surface = LONGCLIP_NAME_MAP.get(row["surface"], row["surface"])
                rows.append(
                    {
                        "surface": surface,
                        "rows": int(row["rows"]),
                        "trunc_gt_248": float(row["trunc_gt_248"]),
                        "tok_mean": float(row["tok_mean"]),
                        "pos_mean": float(row["pos_mean"]),
                        "i2t_margin_mean": float(row["i2t_margin_mean"]),
                        "i2t_r1": float(row["i2t_r1"]),
                        "t2i_margin_mean": float(row["t2i_margin_mean"]),
                        "t2i_r1": float(row["t2i_r1"]),
                        "source": str(path),
                    }
                )
    return sorted(rows, key=lambda row: (surface_rank(row["surface"]), row["surface"]))


def write_tsv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_vqa_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| Surface | Resp | OK | Questions | Support ↑ | Risk ↓ | Uncertain ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {surface} | {responses:,} | {ok:,} | {questions:,} | "
            "{support:.4f} | {risk:.4f} | {uncertain:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_longclip_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| Surface | Rows | Trunc >248 | Tok Mean | Pos Cos ↑ | I2T Margin ↑ | I2T R@1 ↑ | T2I Margin ↑ | T2I R@1 ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {surface} | {rows:,} | {trunc_gt_248:.4f} | {tok_mean:.2f} | {pos_mean:.6f} | "
            "{i2t_margin_mean:.6f} | {i2t_r1:.4f} | {t2i_margin_mean:.6f} | {t2i_r1:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * q / 100
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (k - lower)


def ngrams(tokens: list[str], order: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + order]) for index in range(max(0, len(tokens) - order + 1))]


def content_prefix(tokens: list[str], n: int = 5) -> str:
    start = 0
    while start < len(tokens) and tokens[start] in STOPWORDS:
        start += 1
    return " ".join(tokens[start : start + n])


def repeated_ngram_stats(captions: list[str], order: int) -> dict[str, float]:
    repeated = 0
    distinct_ratios: list[float] = []
    max_repeats: list[int] = []
    for caption in captions:
        grams = ngrams(tokenize(caption), order)
        if not grams:
            continue
        counts = Counter(grams)
        repeated += int(any(count >= 2 for count in counts.values()))
        distinct_ratios.append(len(counts) / len(grams))
        max_repeats.append(max(counts.values()))
    eligible = len(distinct_ratios)
    return {
        f"rep{order}_rate": repeated / eligible if eligible else 0.0,
        f"within_d{order}_mean": mean(distinct_ratios) if distinct_ratios else 0.0,
        f"max{order}_mean": mean(max_repeats) if max_repeats else 0.0,
    }


def load_reference_captions(path: Path) -> dict[str, list[str]]:
    latest_by_key: dict[tuple[str, int], str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            surface = row["surface"]
            source_row = int(row["source_row"])
            caption = row.get("source_caption") or row.get("caption") or ""
            latest_by_key[(surface, source_row)] = caption
    rows: dict[str, list[str]] = defaultdict(list)
    for (surface, _source_row), caption in sorted(latest_by_key.items()):
        rows[surface].append(caption)
    return dict(rows)


def load_naive_captions(path: Path) -> list[str]:
    captions: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            captions.append(row["caption"])
    return captions


def summarize_cpu_surface(surface: str, captions: list[str], source: Path) -> dict[str, Any]:
    tokenized = [tokenize(caption) for caption in captions]
    lengths = [len(tokens) for tokens in tokenized]
    raw_prefixes = Counter(" ".join(tokens[:5]) for tokens in tokenized if tokens)
    content_prefixes = Counter(content_prefix(tokens) for tokens in tokenized if tokens)
    phrase_rates = {
        phrase.replace(" ", "_"): sum(1 for caption in captions if phrase in caption.lower()) / len(captions)
        if captions
        else 0.0
        for phrase in SUSPICIOUS_PHRASES
    }
    rep4 = repeated_ngram_stats(captions, 4)
    rep6 = repeated_ngram_stats(captions, 6)
    return {
        "surface": surface,
        "captions": len(captions),
        "mean_lex_tokens": mean(lengths) if lengths else 0.0,
        "p50_lex_tokens": percentile(lengths, 50),
        "p95_lex_tokens": percentile(lengths, 95),
        "max_lex_tokens": max(lengths) if lengths else 0,
        "gt64_rate": sum(length > 64 for length in lengths) / len(lengths) if lengths else 0.0,
        "gt128_rate": sum(length > 128 for length in lengths) / len(lengths) if lengths else 0.0,
        "gt248_rate": sum(length > 248 for length in lengths) / len(lengths) if lengths else 0.0,
        "gt320_rate": sum(length > 320 for length in lengths) / len(lengths) if lengths else 0.0,
        **rep4,
        **rep6,
        "raw_prefix_top10_mass": sum(value for _, value in raw_prefixes.most_common(10)) / len(captions)
        if captions
        else 0.0,
        "content_prefix_top10_mass": sum(value for _, value in content_prefixes.most_common(10)) / len(captions)
        if captions
        else 0.0,
        "newline_rate": sum("\n" in caption for caption in captions) / len(captions) if captions else 0.0,
        **phrase_rates,
        "source": str(source),
        "top_raw_prefixes": [{"prefix": key, "count": value} for key, value in raw_prefixes.most_common(10)],
        "top_content_prefixes": [{"prefix": key, "count": value} for key, value in content_prefixes.most_common(10)],
    }


def load_cpu_rows(ref_captions: Path, naive_captions: Path) -> list[dict[str, Any]]:
    rows_by_surface = load_reference_captions(ref_captions)
    rows_by_surface["naive_qwen35_cc12m"] = load_naive_captions(naive_captions)
    rows = [
        summarize_cpu_surface(
            surface,
            rows_by_surface[surface],
            naive_captions if surface == "naive_qwen35_cc12m" else ref_captions,
        )
        for surface in rows_by_surface
    ]
    return sorted(rows, key=lambda row: (surface_rank(row["surface"]), row["surface"]))


def write_cpu_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| Surface | Captions | Mean Lex | P95 Lex | >248 | Rep4 | Raw Prefix Top10 | "
        "Newline | this_is | overall | likely | appears_to |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {surface} | {captions:,} | {mean_lex_tokens:.2f} | {p95_lex_tokens:.1f} | "
            "{gt248_rate:.4f} | {rep4_rate:.4f} | {raw_prefix_top10_mass:.4f} | "
            "{newline_rate:.4f} | {this_is:.4f} | {overall:.4f} | {likely:.4f} | "
            "{appears_to:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_cpu_full_survey(cpu_rows: list[dict[str, Any]], ref_captions: Path, naive_captions: Path) -> dict[str, Any]:
    reference_captions = load_reference_captions(ref_captions)
    reference_captions["naive_qwen35_cc12m"] = load_naive_captions(naive_captions)
    captions_by_surface = {
        row["surface"]: reference_captions[row["surface"]]
        for row in sorted(cpu_rows, key=lambda item: (surface_rank(item["surface"]), item["surface"]))
    }
    summaries: dict[str, Any] = {}
    budget_tokenizer = corpus_survey.LexicalBudgetTokenizer()
    for surface, captions in captions_by_surface.items():
        source = naive_captions if surface == "naive_qwen35_cc12m" else ref_captions
        summaries[surface] = corpus_survey.build_summary(
            captions=captions,
            inputs=[str(source)],
            hf_dataset=None,
            hf_config=None,
            hf_split="train",
            records_seen=len(captions),
            caption_field="caption",
            observed_caption_fields={"caption": len(captions)},
            budget_tokenizer_backend="lexical",
            budget_tokenizer_name=None,
            prefix_tokens=5,
            opening_tokens=3,
            top_k=100,
            top_ks=SURVEY_TOP_KS,
            ngram_orders=SURVEY_NGRAM_ORDERS,
            repeat_ngram_orders=SURVEY_REPEAT_NGRAM_ORDERS,
            budgets=SURVEY_BUDGETS,
            workers=8,
            budget_tokenizer=budget_tokenizer,
            budget_prefix_metrics=True,
        )
    return {
        "provenance": {
            "script": "scripts/caption_corpus_survey.py",
            "tokenizer": "repo-standard regex lexical units",
            "prefix_tokens": 5,
            "opening_tokens": 3,
            "top_k": 100,
            "top_ks": SURVEY_TOP_KS,
            "ngram_orders": SURVEY_NGRAM_ORDERS,
            "repeat_ngram_orders": SURVEY_REPEAT_NGRAM_ORDERS,
            "token_budgets": SURVEY_BUDGETS,
            "workers": 8,
        },
        "surfaces": summaries,
    }


def flatten_cpu_survey_rows(full_survey: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for surface, data in full_survey["surfaces"].items():
        full = data["full_length_reference"]
        length = data["length_controlled"]
        normalized = full["full_caption_normalized"]
        rows.append(
            {
                "surface": surface,
                "records": data["captions_loaded"],
                "avg_tokens": full["avg_lexical_tokens"],
                "cov64": length["64"]["coverage_rate"],
                "cov128": length["128"]["coverage_rate"],
                "cov248": length["248"]["coverage_rate"],
                "cov256": length["256"]["coverage_rate"],
                "cov320": length["320"]["coverage_rate"],
                "distinct1_full": full["distinct_n"]["1"],
                "distinct2_full": full["distinct_n"]["2"],
                "distinct3_full": full["distinct_n"]["3"],
                "m3_top100_full": full["ngram_top_k_mass"]["3"],
                "prefix_raw_top100_full": full["prefix_raw"]["top_k_mass"],
                "prefix_content_top100_full": full["prefix_content"]["top_k_mass"],
                "prefix_content_entropy_full": full["prefix_content"]["entropy"],
                "rep3_full": full["repeated_ngram_rate_by_n"]["3"],
                "rep4_full": full["repeated_ngram_rate_by_n"]["4"],
                "rep5_full": full["repeated_ngram_rate_by_n"]["5"],
                "rep6_full": full["repeated_ngram_rate_by_n"]["6"],
                "within_d3_full": full["within_caption_distinct_n_mean"]["3"],
                "within_d4_full": full["within_caption_distinct_n_mean"]["4"],
                "violation_rate_full": full["violation_rate"],
                "viol_ind_per64_full": normalized["violation_indicator_per_64_lexical_tokens_mean"],
                "viol_codes_per64_full": normalized["violation_codes_per_64_lexical_tokens_mean"],
                "control_hits_per64_full": normalized["control_hits_per_64_lexical_tokens"],
                "format_newline_rate": full["operational_formatting_rate"].get("newline", 0.0),
                "format_bullet_rate": full["operational_formatting_rate"].get("bullet", 0.0),
                "format_numbered_list_rate": full["operational_formatting_rate"].get("numbered_list", 0.0),
                "top_opening_1": full["top_openings"][0]["opening"] if full["top_openings"] else "",
                "top_opening_1_count": full["top_openings"][0]["count"] if full["top_openings"] else 0,
            }
        )
    return sorted(rows, key=lambda row: (surface_rank(row["surface"]), row["surface"]))


def write_cpu_full_metrics_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| Surface | Avg Tok | Cov64 | Cov128 | Cov248 | D2 Full | D3 Full | "
        "M3 Top100 | Prefix Top100 | Rep4 | Viol/64 | Control/64 | Top Opening |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {surface} | {avg_tokens:.2f} | {cov64:.4f} | {cov128:.4f} | {cov248:.4f} | "
            "{distinct2_full:.6f} | {distinct3_full:.6f} | {m3_top100_full:.4f} | "
            "{prefix_content_top100_full:.4f} | {rep4_full:.4f} | {viol_codes_per64_full:.6f} | "
            "{control_hits_per64_full:.4f} | {top_opening_1} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vqa_rows = load_vqa_rows([args.ref_vqa, args.naive_vqa])
    longclip_rows = load_longclip_rows([args.ref_longclip, args.naive_longclip])
    cpu_rows = load_cpu_rows(args.ref_captions, args.naive_captions)
    cpu_full_survey = build_cpu_full_survey(cpu_rows, args.ref_captions, args.naive_captions)
    cpu_full_rows = flatten_cpu_survey_rows(cpu_full_survey)

    write_tsv(
        vqa_rows,
        args.output_dir / "gemma4_vqa_cbu_comparison.tsv",
        ["surface", "responses", "ok", "questions", "support", "risk", "uncertain", "source"],
    )
    write_vqa_markdown(vqa_rows, args.output_dir / "gemma4_vqa_cbu_comparison.md")
    write_tsv(
        longclip_rows,
        args.output_dir / "longclip_comparison.tsv",
        [
            "surface",
            "rows",
            "trunc_gt_248",
            "tok_mean",
            "pos_mean",
            "i2t_margin_mean",
            "i2t_r1",
            "t2i_margin_mean",
            "t2i_r1",
            "source",
        ],
    )
    write_longclip_markdown(longclip_rows, args.output_dir / "longclip_comparison.md")
    cpu_columns = [
        "surface",
        "captions",
        "mean_lex_tokens",
        "p50_lex_tokens",
        "p95_lex_tokens",
        "max_lex_tokens",
        "gt64_rate",
        "gt128_rate",
        "gt248_rate",
        "gt320_rate",
        "rep4_rate",
        "within_d4_mean",
        "max4_mean",
        "rep6_rate",
        "within_d6_mean",
        "max6_mean",
        "raw_prefix_top10_mass",
        "content_prefix_top10_mass",
        "newline_rate",
        "this_is",
        "overall",
        "appears_to",
        "likely",
        "the_image",
        "the_scene",
        "capturing",
        "detailed",
        "vibrant",
        "cinematic",
        "inviting",
        "evoking",
        "source",
    ]
    write_tsv(cpu_rows, args.output_dir / "cpu_text_comparison.tsv", cpu_columns)
    write_cpu_markdown(cpu_rows, args.output_dir / "cpu_text_comparison.md")
    (args.output_dir / "cpu_text_summary.json").write_text(
        json.dumps(
            {
                "tokenizer": "regex lexical units: [^\\W_]+(?:'[^\\W_]+)*",
                "surfaces": cpu_rows,
                "interpretation": (
                    "CPU text metrics are whole-caption verbosity, repetition, and register diagnostics; "
                    "they are not visual faithfulness scores."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "cpu_text_full_survey.json").write_text(
        json.dumps(cpu_full_survey, indent=2) + "\n",
        encoding="utf-8",
    )
    cpu_full_columns = [
        "surface",
        "records",
        "avg_tokens",
        "cov64",
        "cov128",
        "cov248",
        "cov256",
        "cov320",
        "distinct1_full",
        "distinct2_full",
        "distinct3_full",
        "m3_top100_full",
        "prefix_raw_top100_full",
        "prefix_content_top100_full",
        "prefix_content_entropy_full",
        "rep3_full",
        "rep4_full",
        "rep5_full",
        "rep6_full",
        "within_d3_full",
        "within_d4_full",
        "violation_rate_full",
        "viol_ind_per64_full",
        "viol_codes_per64_full",
        "control_hits_per64_full",
        "format_newline_rate",
        "format_bullet_rate",
        "format_numbered_list_rate",
        "top_opening_1",
        "top_opening_1_count",
    ]
    write_tsv(cpu_full_rows, args.output_dir / "cpu_text_full_metrics.tsv", cpu_full_columns)
    write_cpu_full_metrics_markdown(cpu_full_rows, args.output_dir / "cpu_text_full_metrics.md")

    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "vqa_rows": len(vqa_rows),
        "longclip_rows": len(longclip_rows),
        "cpu_rows": len(cpu_rows),
        "qwen397b_status": "pending_gpu_approval",
        "outputs": sorted(
            [
                "cpu_text_comparison.md",
                "cpu_text_comparison.tsv",
                "cpu_text_full_metrics.md",
                "cpu_text_full_metrics.tsv",
                "cpu_text_full_survey.json",
                "cpu_text_summary.json",
                "gemma4_vqa_cbu_comparison.md",
                "gemma4_vqa_cbu_comparison.tsv",
                "longclip_comparison.md",
                "longclip_comparison.tsv",
                manifest_path.name,
            ]
        ),
        "note": (
            "Gemma4 VQA-CBU and LongCLIP rows are judge/model matched across surfaces. "
            "Claimed CBU counts are not merged here because the existing reference comparison "
            "uses Qwen397 while the naive Qwen35 baseline was judged with Gemma4 first."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
