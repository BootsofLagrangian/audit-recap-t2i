#!/usr/bin/env python3
"""Fast CPU-only survey for recap/caption corpora stored as JSONL or Parquet."""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq
from datasets import load_dataset

TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
DEFAULT_FIELDS = [
    "recap_caption",
    "re_caption",
    "caption",
    "caption_long_llama32",
    "caption_llava",
    "caption_llava_short",
    "llava_next_caption",
    "synthetic_caption",
    "phi3_caption",
    "parsed",
    "text",
    "txt",
]
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

NUMBER_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
}

CONTROL_LEXICONS = {
    "color": {
        "black",
        "white",
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
        "grey",
        "cyan",
        "magenta",
        "gold",
        "silver",
        "monochrome",
    },
    "material": {
        "wood",
        "wooden",
        "metal",
        "metallic",
        "glass",
        "ceramic",
        "plastic",
        "leather",
        "cotton",
        "silk",
        "stone",
        "marble",
        "concrete",
        "paper",
        "fur",
        "wool",
    },
    "lighting": {
        "lighting",
        "light",
        "shadow",
        "sunlight",
        "backlit",
        "neon",
        "glowing",
        "dim",
        "bright",
        "soft",
        "harsh",
        "ambient",
        "dramatic",
        "rim",
        "volumetric",
    },
    "camera": {
        "camera",
        "lens",
        "shot",
        "view",
        "angle",
        "portrait",
        "macro",
        "closeup",
        "close",
        "wide",
        "telephoto",
        "fisheye",
        "aerial",
        "overhead",
        "bokeh",
        "depth",
    },
    "spatial_relation": {
        "above",
        "below",
        "beneath",
        "under",
        "over",
        "behind",
        "front",
        "beside",
        "between",
        "inside",
        "outside",
        "around",
        "across",
        "near",
        "left",
        "right",
        "center",
        "foreground",
        "background",
    },
    "style": {
        "style",
        "anime",
        "photorealistic",
        "realistic",
        "cinematic",
        "illustration",
        "painting",
        "watercolor",
        "sketch",
        "render",
        "cartoon",
        "pixel",
        "vintage",
        "minimalist",
        "surreal",
        "abstract",
    },
}

CJK_RE = re.compile(r"[\u3400-\u9fff]")
EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "]+"
)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")
BULLET_RE = re.compile(r"(?m)^\s*[-*]\s+")
NUMBERED_LIST_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
MARKDOWN_FENCE_RE = re.compile(r"```")


def load_polishing_checker() -> Any:
    module_path = Path(__file__).resolve().parent / "vllm" / "polishing_check.py"
    spec = importlib.util.spec_from_file_location("polishing_check_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load polishing checker from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.classify_caption


CLASSIFY_CAPTION = load_polishing_checker()


class BudgetTokenizer:
    """Tokenizer used only for length-control truncation."""

    def count_units(self, text: str) -> int:
        raise NotImplementedError

    def truncate_with_count(self, text: str, token_budget: int) -> tuple[str, int]:
        raise NotImplementedError


class LexicalBudgetTokenizer(BudgetTokenizer):
    def count_units(self, text: str) -> int:
        return len(TOKEN_RE.findall(text))

    def truncate_with_count(self, text: str, token_budget: int) -> tuple[str, int]:
        if token_budget <= 0:
            return "", 0
        matches = list(TOKEN_RE.finditer(text))
        if len(matches) <= token_budget:
            return text, len(matches)
        end = matches[token_budget - 1].end()
        return text[:end].strip(), token_budget


class HuggingFaceBudgetTokenizer(BudgetTokenizer):
    def __init__(self, name: str):
        from transformers import AutoTokenizer

        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=False)

    def count_units(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_with_count(self, text: str, token_budget: int) -> tuple[str, int]:
        if token_budget <= 0:
            return "", 0
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= token_budget:
            return text, len(token_ids)
        return self.tokenizer.decode(token_ids[:token_budget], skip_special_tokens=True).strip(), token_budget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Survey a caption corpus for fast recap metrics")
    parser.add_argument("--input", action="append", default=[], help="Input file or glob (repeatable)")
    parser.add_argument("--hf-dataset", default=None, help="Optional Hugging Face dataset id to stream")
    parser.add_argument("--hf-config", default=None, help="Optional Hugging Face dataset config")
    parser.add_argument("--hf-split", default="train", help="Hugging Face split name")
    parser.add_argument("--caption-field", default=None, help="Caption field name. Auto-detect by default.")
    parser.add_argument("--max-records", type=int, default=100000, help="Maximum captions to read")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic file shuffle seed")
    parser.add_argument(
        "--budget-tokenizer-backend",
        choices=["lexical", "hf"],
        default="lexical",
        help="Tokenizer used for length-control budgets",
    )
    parser.add_argument(
        "--budget-tokenizer-name",
        default=None,
        help="Hugging Face tokenizer name/path when --budget-tokenizer-backend=hf",
    )
    parser.add_argument("--prefix-tokens", type=int, default=5, help="Prefix length in tokens")
    parser.add_argument("--opening-tokens", type=int, default=3, help="Opening phrase length in tokens")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k mass cutoff for prefix stats")
    parser.add_argument(
        "--top-ks",
        default="10,100,1000",
        help="Comma-separated top-k cutoffs for appendix concentration curves",
    )
    parser.add_argument(
        "--ngram-orders",
        default="1,2,3",
        help="Comma-separated global n-gram orders for exact corpus counters",
    )
    parser.add_argument(
        "--repeat-ngram-orders",
        default="3,4,5,6",
        help="Comma-separated within-caption n-gram orders for repetition diagnostics",
    )
    parser.add_argument(
        "--token-budgets",
        default="16,32,64,128",
        help="Comma-separated token budgets for length-controlled evaluation",
    )
    parser.add_argument(
        "--no-budget-prefix-metrics",
        action="store_false",
        dest="budget_prefix_metrics",
        help="Disable prefix/opening metrics inside each length-controlled budget",
    )
    parser.set_defaults(budget_prefix_metrics=True)
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1), help="CPU worker count")
    return parser.parse_args()


def expand_inputs(patterns: list[str], seed: int) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if any(ch in pattern for ch in "*?[]"):
            files.extend(Path(match) for match in glob.glob(pattern))
        elif path.exists():
            files.append(path)
    unique = sorted({path.resolve() for path in files})
    rng = random.Random(seed)
    rng.shuffle(unique)
    return unique


def detect_format(path: Path) -> str:
    if path.suffix == ".jsonl":
        return "jsonl"
    if path.suffix == ".parquet":
        return "parquet"
    raise ValueError(f"Unsupported input format: {path}")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            yield row


def iter_records(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        fmt = detect_format(path)
        yield from (iter_jsonl(path) if fmt == "jsonl" else iter_parquet(path))


def iter_hf_records(
    dataset_name: str,
    config_name: str | None,
    split: str,
    seed: int,
    selected_columns: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
    if selected_columns:
        dataset = dataset.select_columns(selected_columns)
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    for row in dataset:
        yield dict(row)


def extract_caption(record: dict[str, Any], field: str | None) -> tuple[str | None, str | None]:
    def normalize(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return None

    if field:
        return normalize(record.get(field)), field
    for key in DEFAULT_FIELDS:
        normalized = normalize(record.get(key))
        if normalized is not None:
            return normalized, key
    return None, None


def parse_budgets(raw: str) -> list[int]:
    budgets = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    return [budget for budget in budgets if budget > 0]


def parse_positive_ints(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    return [value for value in values if value > 0]


def build_budget_tokenizer(backend: str, name: str | None) -> BudgetTokenizer:
    if backend == "lexical":
        return LexicalBudgetTokenizer()
    if backend == "hf":
        if not name:
            raise ValueError("--budget-tokenizer-name is required when --budget-tokenizer-backend=hf")
        return HuggingFaceBudgetTokenizer(name)
    raise ValueError(f"Unsupported budget tokenizer backend: {backend}")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def nested_get(mapping: dict[str, Any], path: str) -> float | None:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return float(current) if isinstance(current, (int, float)) else None


def interpolate_budget_at_threshold(
    budgets: list[int],
    values: list[float],
    threshold: float,
) -> float | None:
    if not budgets or not values or len(budgets) != len(values):
        return None
    if values[0] >= threshold:
        return float(budgets[0])
    for index in range(1, len(budgets)):
        previous_value = values[index - 1]
        current_value = values[index]
        if previous_value < threshold <= current_value:
            previous_budget = budgets[index - 1]
            current_budget = budgets[index]
            if current_value == previous_value:
                return float(current_budget)
            ratio = (threshold - previous_value) / (current_value - previous_value)
            return round(previous_budget + ratio * (current_budget - previous_budget), 2)
    return None


def trapezoid_area(xs: list[int], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for index in range(1, len(xs)):
        width = xs[index] - xs[index - 1]
        height = (ys[index] + ys[index - 1]) / 2.0
        area += width * height
    return area


def curve_features_for_metric(
    metric_key: str,
    direction: str,
    budgets: list[int],
    series: list[float],
) -> dict[str, Any]:
    if not series:
        return {
            "metric": metric_key,
            "direction": direction,
            "series": {},
            "goodness_curve": {},
            "headroom_curve": {},
        }

    if direction == "up":
        worst = min(series)
        best = max(series)
        transform = lambda value: 0.0 if best == worst else (value - worst) / (best - worst)
    elif direction == "down":
        worst = max(series)
        best = min(series)
        transform = lambda value: 0.0 if best == worst else (worst - value) / (worst - best)
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    progress_values = [max(0.0, min(1.0, transform(value))) for value in series]
    gap_values = [round(1.0 - value, 4) for value in progress_values]
    span = max(1, budgets[-1] - budgets[0]) if budgets else 1
    auc = trapezoid_area(budgets, progress_values) / span if len(budgets) >= 2 else progress_values[0]

    return {
        "metric": metric_key,
        "direction": direction,
        "normalization_basis": "within_surface_budget_range",
        "series": {str(budget): round(value, 6) for budget, value in zip(budgets, series, strict=False)},
        "relative_progress_curve": {
            str(budget): round(value, 4) for budget, value in zip(budgets, progress_values, strict=False)
        },
        "relative_gap_curve": {
            str(budget): round(value, 4) for budget, value in zip(budgets, gap_values, strict=False)
        },
        "relative_progress_auc": round(auc, 4),
        "relative_gap_auc": round(1.0 - auc, 4),
        "budget_at_50pct_relative_progress": interpolate_budget_at_threshold(budgets, progress_values, 0.5),
        "budget_at_75pct_relative_progress": interpolate_budget_at_threshold(budgets, progress_values, 0.75),
        "budget_at_90pct_relative_progress": interpolate_budget_at_threshold(budgets, progress_values, 0.9),
        "observed_best": round(best, 6),
        "observed_worst": round(worst, 6),
    }


def derive_budget_curve_features(length_controlled: dict[str, Any], budgets: list[int]) -> dict[str, Any]:
    metric_specs = [
        ("distinct_n.2", "up"),
        ("distinct_n.3", "up"),
        ("ngram_top_k_mass.2", "down"),
        ("ngram_top_k_mass.3", "down"),
        ("repeated_4gram_rate", "down"),
        ("violation_rate", "down"),
    ]
    features: dict[str, Any] = {}
    for metric_key, direction in metric_specs:
        series = []
        for budget in budgets:
            summary = length_controlled.get(str(budget), {})
            value = nested_get(summary, metric_key)
            if value is None:
                series = []
                break
            series.append(value)
        if series:
            features[metric_key] = curve_features_for_metric(metric_key, direction, budgets, series)
    return features


def metric_provenance(ngram_orders: list[int], repeat_ngram_orders: list[int]) -> dict[str, Any]:
    return {
        "main_or_appendix_metrics": {
            "budget_eligibility": {
                "fields": ["coverage_rate", "eligibility_rate", "captions_meeting_budget"],
                "claim_scope": "comparability_gate",
                "citation_role": "length-controlled comparison protocol; exact estimator defined in this work",
            },
            "distinct_n": {
                "fields": [f"distinct_n.{order}" for order in ngram_orders],
                "claim_scope": "lexical_diversity_proxy",
                "citation_keys": ["li2016diversity"],
            },
            "ngram_entropy_and_top_mass": {
                "fields": [
                    "ngram_distribution",
                    "ngram_top_k_mass",
                    "ngram_top_k_mass_by_k",
                ],
                "claim_scope": "surface_concentration_proxy",
                "citation_keys": ["shannon1948communication", "miller1955note"],
            },
            "prefix_entropy_and_top_mass": {
                "fields": ["prefix_raw", "prefix_content", "top_openings"],
                "claim_scope": "opening_template_concentration_proxy",
                "citation_role": "entropy/top-mass concentration estimator; caption-register application defined in this work",
                "citation_keys": ["shannon1948communication", "miller1955note"],
            },
            "within_caption_repetition": {
                "fields": [
                    "repeated_ngram_rate_by_n",
                    "within_caption_distinct_n_mean",
                    "within_caption_max_ngram_repeat_mean",
                    "full_caption_normalized",
                ],
                "orders": repeat_ngram_orders,
                "claim_scope": "within_caption_redundancy_proxy",
                "citation_role": "n-gram repetition diagnostic; related to distinct-n diversity practice",
                "citation_keys": ["li2016diversity"],
            },
            "caption_balanced_top_mass": {
                "fields": ["full_caption_normalized.caption_balanced_ngram_top_k_mass"],
                "claim_scope": "length_normalized_surface_concentration_proxy",
                "citation_role": "top-k mass concentration estimator applied with caption-balanced averaging",
                "citation_keys": ["shannon1948communication", "miller1955note"],
            },
            "dedup": {
                "fields": ["dedup"],
                "claim_scope": "exact_normalized_duplicate_collapse",
                "citation_role": "dataset deduplication/collapse audit; exact hash estimator defined in this work",
            },
            "violation_code_rate": {
                "fields": ["violation_rate", "violation_code_rate", "violation_severity_rate"],
                "claim_scope": "operational_caption_register_and_instruction_artifact_audit",
                "citation_role": "regex taxonomy is project-defined and must be reported as an operational audit, not a standard metric",
            },
        },
        "debug_or_stratification_only": {
            "operational_formatting_rate": {
                "claim_scope": "debug_artifact_counter",
                "citation_role": "not a headline metric",
            },
            "debug_control_lexicon": {
                "claim_scope": "weak_control_vocabulary_diagnostic",
                "citation_role": "appendix/debug only unless replaced by validated CBU/SCU extractor",
            },
            "stratification_script_rate": {
                "claim_scope": "language_script_stratification",
                "citation_role": "stratification only; not a quality score",
            },
        },
        "model_based_metrics_not_computed_here": {
            "vendi": {
                "claim_scope": "embedding_semantic_diversity",
                "citation_keys": ["friedman2023vendi"],
            },
            "mauve_or_distribution_frontier": {
                "claim_scope": "prompt_caption_distribution_gap",
                "citation_keys": ["pillutla2021mauve"],
            },
            "clipscore_or_long_context_retrieval_margin": {
                "claim_scope": "image_conditioned_compatibility_proxy",
                "citation_keys": ["hessel2021clipscore"],
            },
        },
    }


def content_tokens(tokens: list[str]) -> list[str]:
    trimmed = list(tokens)
    while trimmed and trimmed[0] in LEADING_STOPWORDS:
        trimmed.pop(0)
    return trimmed


def entropy(counter: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        value -= p * math.log(p)
    return value


def entropy_miller_madow(counter: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    correction = (len(counter) - 1) / (2 * total)
    return entropy(counter, total) + correction


def hhi(counter: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    return sum((count / total) ** 2 for count in counter.values())


def top_mass(counter: Counter[str], total: int, k: int) -> float:
    if total <= 0:
        return 0.0
    return sum(count for _, count in counter.most_common(k)) / total


def counter_distribution_summary(counter: Counter[str], total: int, top_ks: list[int]) -> dict[str, Any]:
    entropy_mle = entropy(counter, total)
    entropy_mm = entropy_miller_madow(counter, total)
    concentration = hhi(counter, total)
    return {
        "entropy": round(entropy_mle, 4),
        "entropy_miller_madow": round(entropy_mm, 4),
        "effective_vocab": round(math.exp(entropy_mle), 2) if total else 0.0,
        "effective_vocab_miller_madow": round(math.exp(entropy_mm), 2) if total else 0.0,
        "hhi": round(concentration, 8),
        "effective_simpson": round(1.0 / concentration, 2) if concentration > 0 else 0.0,
        "top_k_mass_by_k": {
            str(k): round(top_mass(counter, total, k), 4) for k in top_ks
        },
    }


def has_repeated_ngram(tokens: list[str], n: int = 4, threshold: int = 2) -> bool:
    grams = Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))
    return any(count >= threshold for count in grams.values())


def ngrams(tokens: list[str], n: int) -> Iterator[str]:
    for index in range(max(0, len(tokens) - n + 1)):
        yield " ".join(tokens[index : index + n])


def normalized_caption_hash(caption: str) -> str:
    normalized = " ".join(tokenize(caption))
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).hexdigest()


def formatting_hits(caption: str) -> set[str]:
    hits: set[str] = set()
    if "\n" in caption:
        hits.add("newline")
    if BULLET_RE.search(caption):
        hits.add("bullet")
    if NUMBERED_LIST_RE.search(caption):
        hits.add("numbered_list")
    if URL_RE.search(caption):
        hits.add("url")
    if HTML_ENTITY_RE.search(caption):
        hits.add("html_entity")
    if MARKDOWN_FENCE_RE.search(caption):
        hits.add("markdown_fence")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in caption):
        hits.add("control_char")
    return hits


def script_hits(caption: str) -> set[str]:
    hits: set[str] = set()
    if any(ord(char) > 127 for char in caption):
        hits.add("non_ascii")
    if CJK_RE.search(caption):
        hits.add("cjk")
    if EMOJI_RE.search(caption):
        hits.add("emoji")
    return hits


def control_lexicon_hits(tokens: list[str]) -> tuple[Counter[str], Counter[str]]:
    token_counts: Counter[str] = Counter()
    caption_counts: Counter[str] = Counter()
    token_set = set(tokens)
    for category, lexicon in CONTROL_LEXICONS.items():
        hits = sum(1 for token in tokens if token in lexicon)
        if hits:
            token_counts[category] += hits
            caption_counts[category] += 1
    count_hits = sum(1 for token in tokens if token.isdigit() or token in NUMBER_WORDS)
    if count_hits:
        token_counts["count"] += count_hits
        caption_counts["count"] += 1
    if token_set.intersection({"text", "word", "letter", "sign", "logo", "label", "caption", "title"}):
        text_hits = sum(
            1 for token in tokens if token in {"text", "word", "letter", "sign", "logo", "label", "caption", "title"}
        )
        token_counts["text_rendering"] += text_hits
        caption_counts["text_rendering"] += 1
    return caption_counts, token_counts


def within_caption_ngram_stats(tokens: list[str], n: int) -> tuple[float | None, int | None, bool]:
    grams = list(ngrams(tokens, n))
    if not grams:
        return None, None, False
    counts = Counter(grams)
    return len(counts) / len(grams), max(counts.values()), any(count >= 2 for count in counts.values())


def chunked(items: list[str], parts: int) -> list[list[str]]:
    if not items:
        return []
    parts = max(1, min(parts, len(items)))
    chunk_size = math.ceil(len(items) / parts)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def analyze_caption_chunk(
    captions: list[str],
    prefix_tokens: int,
    opening_tokens: int,
    ngram_orders: list[int],
    repeat_ngram_orders: list[int],
) -> dict[str, Any]:
    raw_prefixes: Counter[str] = Counter()
    content_prefixes: Counter[str] = Counter()
    openings: Counter[str] = Counter()
    violation_codes: Counter[str] = Counter()
    violation_severities: Counter[str] = Counter()
    ngram_counters: dict[int, Counter[str]] = {order: Counter() for order in ngram_orders}
    repeated_by_order: Counter[str] = Counter()
    within_distinct_sum: Counter[str] = Counter()
    within_max_repeat_sum: Counter[str] = Counter()
    within_eligible: Counter[str] = Counter()
    formatting: Counter[str] = Counter()
    scripts: Counter[str] = Counter()
    control_caption_counts: Counter[str] = Counter()
    control_token_counts: Counter[str] = Counter()
    caption_hashes: Counter[str] = Counter()
    violation_indicator_per_64_sum = 0.0
    violation_code_count_per_64_sum = 0.0
    total = 0
    token_sum = 0
    char_sum = 0
    violated = 0

    for caption in captions:
        tokens = tokenize(caption)
        if not tokens:
            continue
        total += 1
        token_sum += len(tokens)
        char_sum += len(caption)
        caption_hashes[normalized_caption_hash(caption)] += 1
        for order in ngram_orders:
            ngram_counters[order].update(ngrams(tokens, order))
        for order in repeat_ngram_orders:
            distinct_ratio, max_repeat, has_repeat = within_caption_ngram_stats(tokens, order)
            if distinct_ratio is None or max_repeat is None:
                continue
            order_key = str(order)
            within_eligible[order_key] += 1
            within_distinct_sum[order_key] += distinct_ratio
            within_max_repeat_sum[order_key] += max_repeat
            if has_repeat:
                repeated_by_order[order_key] += 1
        raw_prefixes[" ".join(tokens[:prefix_tokens])] += 1
        content = content_tokens(tokens)
        if content:
            content_prefixes[" ".join(content[:prefix_tokens])] += 1
            openings[" ".join(content[:opening_tokens])] += 1
        else:
            openings[" ".join(tokens[:opening_tokens])] += 1
        caption_violations = CLASSIFY_CAPTION(caption)
        if caption_violations:
            violated += 1
            violation_indicator_per_64_sum += 64.0 / max(1, len(tokens))
        violation_code_count_per_64_sum += len(caption_violations) * 64.0 / max(1, len(tokens))
        for violation in caption_violations:
            violation_codes[violation["code"]] += 1
            violation_severities[violation["severity"]] += 1
        formatting.update(formatting_hits(caption))
        scripts.update(script_hits(caption))
        control_caption_hits, control_token_hits = control_lexicon_hits(tokens)
        control_caption_counts.update(control_caption_hits)
        control_token_counts.update(control_token_hits)

    return {
        "total": total,
        "token_sum": token_sum,
        "char_sum": char_sum,
        "violated": violated,
        "repeated_by_order": repeated_by_order,
        "within_distinct_sum": within_distinct_sum,
        "within_max_repeat_sum": within_max_repeat_sum,
        "within_eligible": within_eligible,
        "raw_prefixes": raw_prefixes,
        "content_prefixes": content_prefixes,
        "openings": openings,
        "violation_codes": violation_codes,
        "violation_severities": violation_severities,
        "ngram_counters": ngram_counters,
        "formatting": formatting,
        "scripts": scripts,
        "control_caption_counts": control_caption_counts,
        "control_token_counts": control_token_counts,
        "caption_hashes": caption_hashes,
        "violation_indicator_per_64_sum": violation_indicator_per_64_sum,
        "violation_code_count_per_64_sum": violation_code_count_per_64_sum,
    }


def caption_balanced_top_mass_chunk(
    captions: list[str],
    ngram_orders: list[int],
    top_ks: list[int],
    top_terms_by_order: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    sums: dict[str, Counter[str]] = {str(order): Counter() for order in ngram_orders}
    eligible: Counter[str] = Counter()
    for caption in captions:
        tokens = tokenize(caption)
        if not tokens:
            continue
        for order in ngram_orders:
            order_key = str(order)
            grams = list(ngrams(tokens, order))
            if not grams:
                continue
            eligible[order_key] += 1
            denominator = len(grams)
            for k in top_ks:
                k_key = str(k)
                top_terms = top_terms_by_order.get(order_key, {}).get(k_key, set())
                if not top_terms:
                    continue
                sums[order_key][k_key] += sum(1 for gram in grams if gram in top_terms) / denominator
    return {
        "sums": sums,
        "eligible": eligible,
    }


def caption_balanced_ngram_top_mass(
    captions: list[str],
    *,
    ngram_counters: dict[int, Counter[str]],
    ngram_orders: list[int],
    top_ks: list[int],
    workers: int,
) -> dict[str, dict[str, float]]:
    top_terms_by_order: dict[str, dict[str, set[str]]] = {}
    for order in ngram_orders:
        order_key = str(order)
        top_terms_by_order[order_key] = {}
        for k in top_ks:
            top_terms_by_order[order_key][str(k)] = {
                gram for gram, _ in ngram_counters[order].most_common(k)
            }

    parts = chunked(captions, workers)
    if len(parts) <= 1:
        partials = [
            caption_balanced_top_mass_chunk(
                parts[0] if parts else [],
                ngram_orders,
                top_ks,
                top_terms_by_order,
            )
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            partials = list(
                pool.map(
                    caption_balanced_top_mass_chunk,
                    parts,
                    [ngram_orders] * len(parts),
                    [top_ks] * len(parts),
                    [top_terms_by_order] * len(parts),
                )
            )

    sums: dict[str, Counter[str]] = {str(order): Counter() for order in ngram_orders}
    eligible: Counter[str] = Counter()
    for partial in partials:
        eligible.update(partial["eligible"])
        for order_key, counter in partial["sums"].items():
            sums[order_key].update(counter)

    return {
        str(order): {
            str(k): round(sums[str(order)][str(k)] / eligible[str(order)], 6)
            if eligible[str(order)]
            else 0.0
            for k in top_ks
        }
        for order in ngram_orders
    }


def summarize_texts(
    captions: list[str],
    *,
    prefix_tokens: int,
    opening_tokens: int,
    top_k: int,
    top_ks: list[int],
    ngram_orders: list[int],
    repeat_ngram_orders: list[int],
    workers: int,
    include_prefix_metrics: bool = True,
    include_caption_balanced_top_mass: bool = False,
) -> dict[str, Any]:
    parts = chunked(captions, workers)
    if len(parts) <= 1:
        partials = [
            analyze_caption_chunk(
                parts[0] if parts else [],
                prefix_tokens,
                opening_tokens,
                ngram_orders,
                repeat_ngram_orders,
            )
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            partials = list(
                pool.map(
                    analyze_caption_chunk,
                    parts,
                    [prefix_tokens] * len(parts),
                    [opening_tokens] * len(parts),
                    [ngram_orders] * len(parts),
                    [repeat_ngram_orders] * len(parts),
                )
            )

    raw_prefixes: Counter[str] = Counter()
    content_prefixes: Counter[str] = Counter()
    openings: Counter[str] = Counter()
    violation_codes: Counter[str] = Counter()
    violation_severities: Counter[str] = Counter()
    ngram_counters: dict[int, Counter[str]] = {order: Counter() for order in ngram_orders}
    repeated_by_order: Counter[str] = Counter()
    within_distinct_sum: Counter[str] = Counter()
    within_max_repeat_sum: Counter[str] = Counter()
    within_eligible: Counter[str] = Counter()
    formatting: Counter[str] = Counter()
    scripts: Counter[str] = Counter()
    control_caption_counts: Counter[str] = Counter()
    control_token_counts: Counter[str] = Counter()
    caption_hashes: Counter[str] = Counter()
    violation_indicator_per_64_sum = 0.0
    violation_code_count_per_64_sum = 0.0
    total = token_sum = char_sum = violated = 0
    for partial in partials:
        total += partial["total"]
        token_sum += partial["token_sum"]
        char_sum += partial["char_sum"]
        violated += partial["violated"]
        repeated_by_order.update(partial["repeated_by_order"])
        within_distinct_sum.update(partial["within_distinct_sum"])
        within_max_repeat_sum.update(partial["within_max_repeat_sum"])
        within_eligible.update(partial["within_eligible"])
        raw_prefixes.update(partial["raw_prefixes"])
        content_prefixes.update(partial["content_prefixes"])
        openings.update(partial["openings"])
        violation_codes.update(partial["violation_codes"])
        violation_severities.update(partial["violation_severities"])
        formatting.update(partial["formatting"])
        scripts.update(partial["scripts"])
        control_caption_counts.update(partial["control_caption_counts"])
        control_token_counts.update(partial["control_token_counts"])
        caption_hashes.update(partial["caption_hashes"])
        violation_indicator_per_64_sum += partial["violation_indicator_per_64_sum"]
        violation_code_count_per_64_sum += partial["violation_code_count_per_64_sum"]
        for order, counter in partial["ngram_counters"].items():
            ngram_counters[int(order)].update(counter)

    raw_total = sum(raw_prefixes.values())
    content_total = sum(content_prefixes.values())
    ngram_totals = {order: sum(counter.values()) for order, counter in ngram_counters.items()}
    ngram_distribution = {
        str(order): counter_distribution_summary(ngram_counters[order], ngram_totals[order], top_ks)
        for order in ngram_orders
    }
    caption_balanced_top_k_mass = (
        caption_balanced_ngram_top_mass(
            captions,
            ngram_counters=ngram_counters,
            ngram_orders=ngram_orders,
            top_ks=top_ks,
            workers=workers,
        )
        if include_caption_balanced_top_mass
        else {}
    )
    repeated_rates = {
        str(order): round(repeated_by_order[str(order)] / total, 4) if total else 0.0
        for order in repeat_ngram_orders
    }
    within_distinct_means = {
        str(order): round(within_distinct_sum[str(order)] / within_eligible[str(order)], 6)
        if within_eligible[str(order)]
        else 0.0
        for order in repeat_ngram_orders
    }
    within_max_repeat_means = {
        str(order): round(within_max_repeat_sum[str(order)] / within_eligible[str(order)], 4)
        if within_eligible[str(order)]
        else 0.0
        for order in repeat_ngram_orders
    }
    duplicate_total = sum(caption_hashes.values())
    duplicate_unique = len(caption_hashes)
    duplicate_mass = sum(count for count in caption_hashes.values() if count > 1)
    control_hits_total = sum(control_token_counts.values())
    summary = {
        "captions_analyzed": total,
        "avg_lexical_tokens": round(token_sum / total, 2) if total else 0.0,
        "avg_chars": round(char_sum / total, 2) if total else 0.0,
        "violation_rate": round(violated / total, 4) if total else 0.0,
        "repeated_4gram_rate": repeated_rates.get("4", 0.0),
        "repeated_ngram_rate_by_n": repeated_rates,
        "within_caption_distinct_n_mean": within_distinct_means,
        "within_caption_max_ngram_repeat_mean": within_max_repeat_means,
        "distinct_n": {
            str(order): round(len(ngram_counters[order]) / max(1, ngram_totals[order]), 6)
            for order in ngram_orders
        },
        "ngram_top_k_mass": {
            str(order): round(top_mass(ngram_counters[order], ngram_totals[order], top_k), 4)
            for order in ngram_orders
        },
        "ngram_distribution": ngram_distribution,
        "ngram_top_k_mass_by_k": {
            str(order): ngram_distribution[str(order)]["top_k_mass_by_k"] for order in ngram_orders
        },
        "violations": dict(violation_codes.most_common()),
        "violation_code_rate": {
            key: round(value / total, 4) if total else 0.0 for key, value in violation_codes.most_common()
        },
        "violation_severity_rate": {
            key: round(value / total, 4) if total else 0.0 for key, value in violation_severities.most_common()
        },
        "dedup": {
            "normalized_unique_rate": round(duplicate_unique / duplicate_total, 6) if duplicate_total else 0.0,
            "normalized_duplicate_rate": round(1.0 - duplicate_unique / duplicate_total, 6) if duplicate_total else 0.0,
            "duplicate_mass_rate": round(duplicate_mass / duplicate_total, 6) if duplicate_total else 0.0,
            "top_duplicate_mass": round(top_mass(caption_hashes, duplicate_total, top_k), 4),
            "max_duplicate_count": max(caption_hashes.values()) if caption_hashes else 0,
            "top_hashes": [{"hash": key, "count": value} for key, value in caption_hashes.most_common(10)],
        },
        "operational_formatting_rate": {
            key: round(value / total, 4) if total else 0.0 for key, value in formatting.most_common()
        },
        "stratification_script_rate": {
            key: round(value / total, 4) if total else 0.0 for key, value in scripts.most_common()
        },
        "debug_control_lexicon": {
            "caption_rate": {
                key: round(value / total, 4) if total else 0.0 for key, value in control_caption_counts.most_common()
            },
            "token_hits": dict(control_token_counts.most_common()),
            "hits_per_caption": round(control_hits_total / total, 4) if total else 0.0,
            "hits_per_64_lexical_tokens": round(control_hits_total / max(1, token_sum) * 64, 4),
        },
        "full_caption_normalized": {
            "normalization_scope": "caption-balanced and/or per-token full-caption metrics; not a faithfulness score",
            "caption_balanced_ngram_top_k_mass": caption_balanced_top_k_mass,
            "within_caption_distinct_n_mean": within_distinct_means,
            "repeated_ngram_rate_by_n": repeated_rates,
            "violation_indicator_per_64_lexical_tokens_mean": round(
                violation_indicator_per_64_sum / total,
                6,
            )
            if total
            else 0.0,
            "violation_codes_per_64_lexical_tokens_mean": round(
                violation_code_count_per_64_sum / total,
                6,
            )
            if total
            else 0.0,
            "control_hits_per_64_lexical_tokens": round(control_hits_total / max(1, token_sum) * 64, 4),
        },
    }
    if include_prefix_metrics:
        raw_prefix_distribution = counter_distribution_summary(raw_prefixes, raw_total, top_ks)
        content_prefix_distribution = counter_distribution_summary(content_prefixes, content_total, top_ks)
        summary["prefix_raw"] = {
            **raw_prefix_distribution,
            "top_k_mass": round(top_mass(raw_prefixes, raw_total, top_k), 4),
            "top_10": [{"prefix": key, "count": value} for key, value in raw_prefixes.most_common(10)],
        }
        summary["prefix_content"] = {
            **content_prefix_distribution,
            "top_k_mass": round(top_mass(content_prefixes, content_total, top_k), 4),
            "top_10": [{"prefix": key, "count": value} for key, value in content_prefixes.most_common(10)],
        }
        summary["top_openings"] = [{"opening": key, "count": value} for key, value in openings.most_common(10)]
    return summary


def summarize_by_budget(
    captions: list[str],
    *,
    budget_tokenizer: BudgetTokenizer,
    budgets: list[int],
    prefix_tokens: int,
    opening_tokens: int,
    top_k: int,
    top_ks: list[int],
    ngram_orders: list[int],
    repeat_ngram_orders: list[int],
    workers: int,
    budget_prefix_metrics: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    previous_budget: int | None = None
    previous_avg_budget_tokens: float | None = None
    for budget in budgets:
        original_counts = [budget_tokenizer.count_units(caption) for caption in captions]
        truncated_with_counts = [budget_tokenizer.truncate_with_count(caption, budget) for caption in captions]
        truncated = [text for text, _ in truncated_with_counts]
        captions_meeting_budget = sum(1 for count in original_counts if count >= budget)
        avg_budget_tokens = round(sum(count for _, count in truncated_with_counts) / len(truncated_with_counts), 2) if truncated_with_counts else 0.0
        summary = summarize_texts(
            truncated,
            prefix_tokens=min(prefix_tokens, budget),
            opening_tokens=min(opening_tokens, budget),
            top_k=top_k,
            top_ks=top_ks,
            ngram_orders=ngram_orders,
            repeat_ngram_orders=repeat_ngram_orders,
            workers=workers,
            include_prefix_metrics=budget_prefix_metrics,
        )
        summary["token_budget"] = budget
        summary["avg_budget_tokens"] = avg_budget_tokens
        summary["captions_meeting_budget"] = captions_meeting_budget
        summary["coverage_rate"] = round(captions_meeting_budget / len(truncated_with_counts), 4) if truncated_with_counts else 0.0
        summary["eligibility_rate"] = summary["coverage_rate"]
        summary["ineligible_count"] = max(0, len(truncated_with_counts) - captions_meeting_budget)
        summary["truncation_rate"] = round(
            sum(1 for count in original_counts if count > budget) / len(original_counts),
            4,
        ) if original_counts else 0.0
        summary["mean_retained_token_fraction"] = round(
            sum(min(count, budget) / max(1, count) for count in original_counts) / len(original_counts),
            4,
        ) if original_counts else 0.0
        if previous_budget is not None and previous_avg_budget_tokens is not None:
            delta = avg_budget_tokens - previous_avg_budget_tokens
            summary["marginal_avg_token_gain"] = round(delta / max(1, budget - previous_budget), 4)
        results[str(budget)] = summary
        previous_budget = budget
        previous_avg_budget_tokens = avg_budget_tokens
    return results


def load_captions(
    *,
    input_patterns: list[str],
    hf_dataset: str | None,
    hf_config: str | None,
    hf_split: str,
    seed: int,
    caption_field: str | None,
    max_records: int,
) -> tuple[list[str], int, list[str], dict[str, int]]:
    paths = expand_inputs(input_patterns, seed) if input_patterns else []
    if not hf_dataset and not paths:
        raise ValueError("No input files matched")

    captions: list[str] = []
    records_seen = 0
    observed_caption_fields: dict[str, int] = {}
    source_iter: Iterator[dict[str, Any]]
    if hf_dataset:
        selected_columns = [caption_field] if caption_field else None
        source_iter = iter_hf_records(hf_dataset, hf_config, hf_split, seed, selected_columns)
    else:
        source_iter = iter_records(paths)

    for record in source_iter:
        records_seen += 1
        caption, used_field = extract_caption(record, caption_field)
        if caption:
            captions.append(caption)
            if used_field is not None:
                observed_caption_fields[used_field] = observed_caption_fields.get(used_field, 0) + 1
        if len(captions) >= max_records:
            break

    return captions, records_seen, [str(path) for path in paths], observed_caption_fields


def build_summary(
    *,
    captions: list[str],
    inputs: list[str],
    hf_dataset: str | None,
    hf_config: str | None,
    hf_split: str,
    records_seen: int,
    caption_field: str | None,
    observed_caption_fields: dict[str, int],
    budget_tokenizer_backend: str,
    budget_tokenizer_name: str | None,
    prefix_tokens: int,
    opening_tokens: int,
    top_k: int,
    top_ks: list[int],
    ngram_orders: list[int],
    repeat_ngram_orders: list[int],
    budgets: list[int],
    workers: int,
    budget_tokenizer: BudgetTokenizer,
    budget_prefix_metrics: bool,
) -> dict[str, Any]:
    length_controlled = summarize_by_budget(
        captions,
        budget_tokenizer=budget_tokenizer,
        budgets=budgets,
        prefix_tokens=prefix_tokens,
        opening_tokens=opening_tokens,
        top_k=top_k,
        top_ks=top_ks,
        ngram_orders=ngram_orders,
        repeat_ngram_orders=repeat_ngram_orders,
        workers=workers,
        budget_prefix_metrics=budget_prefix_metrics,
    )
    if caption_field:
        effective_caption_field = caption_field
    elif len(observed_caption_fields) == 1:
        effective_caption_field = next(iter(observed_caption_fields))
    elif observed_caption_fields:
        effective_caption_field = "mixed"
    else:
        effective_caption_field = "auto"
    return {
        "inputs": inputs,
        "hf_dataset": hf_dataset,
        "hf_config": hf_config,
        "hf_split": hf_split,
        "records_seen": records_seen,
        "captions_loaded": len(captions),
        "caption_field": effective_caption_field,
        "caption_field_requested": caption_field or "auto",
        "caption_fields_observed": observed_caption_fields,
        "budget_tokenizer": {
            "backend": budget_tokenizer_backend,
            "name": budget_tokenizer_name,
        },
        "prefix_tokens": prefix_tokens,
        "top_k": top_k,
        "top_ks": top_ks,
        "ngram_orders": ngram_orders,
        "repeat_ngram_orders": repeat_ngram_orders,
        "token_budgets": budgets,
        "metric_provenance": metric_provenance(ngram_orders, repeat_ngram_orders),
        "full_length_reference": summarize_texts(
            captions,
            prefix_tokens=prefix_tokens,
            opening_tokens=opening_tokens,
            top_k=top_k,
            top_ks=top_ks,
            ngram_orders=ngram_orders,
            repeat_ngram_orders=repeat_ngram_orders,
            workers=workers,
            include_caption_balanced_top_mass=True,
        ),
        "length_controlled": length_controlled,
        "budget_curve_features": derive_budget_curve_features(length_controlled, budgets),
    }


def main() -> int:
    args = parse_args()
    budgets = parse_budgets(args.token_budgets)
    top_ks = parse_positive_ints(args.top_ks)
    if args.top_k not in top_ks:
        top_ks = sorted({*top_ks, args.top_k})
    ngram_orders = parse_positive_ints(args.ngram_orders)
    repeat_ngram_orders = parse_positive_ints(args.repeat_ngram_orders)
    budget_tokenizer = build_budget_tokenizer(args.budget_tokenizer_backend, args.budget_tokenizer_name)
    try:
        captions, records_seen, inputs, observed_caption_fields = load_captions(
            input_patterns=args.input,
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            seed=args.seed,
            caption_field=args.caption_field,
            max_records=args.max_records,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    summary = build_summary(
        captions=captions,
        inputs=inputs,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        records_seen=records_seen,
        caption_field=args.caption_field,
        observed_caption_fields=observed_caption_fields,
        budget_tokenizer_backend=args.budget_tokenizer_backend,
        budget_tokenizer_name=args.budget_tokenizer_name,
        prefix_tokens=args.prefix_tokens,
        opening_tokens=args.opening_tokens,
        top_k=args.top_k,
        top_ks=top_ks,
        ngram_orders=ngram_orders,
        repeat_ngram_orders=repeat_ngram_orders,
        budgets=budgets,
        workers=args.workers,
        budget_tokenizer=budget_tokenizer,
        budget_prefix_metrics=args.budget_prefix_metrics,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Avoid intermittent interpreter-finalization crashes in datasets/pyarrow
    # after the JSON summary has already been produced.
    os._exit(exit_code)
