#!/usr/bin/env python3
"""Summarize claimed or grounded CBU response JSONL into table-ready metrics."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

UNIT_CATEGORIES = [
    "object",
    "attribute",
    "relation",
    "style",
    "camera",
    "lighting",
    "count",
    "text_rendering",
]

TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
ARTICLE_UNITS = {"a", "an", "the"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CBU extraction/audit responses")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["claimed", "grounded"], required=True)
    parser.add_argument("--latest-by-request", action="store_true")
    parser.add_argument("--include", action="append", default=[])
    return parser.parse_args()


def normalize_unit(text: str) -> str:
    tokens = TOKEN_RE.findall(text.lower())
    while tokens and tokens[0] in ARTICLE_UNITS:
        tokens.pop(0)
    return " ".join(tokens)


def normalize_key_part(text: str) -> str:
    normalized = normalize_unit(text)
    return normalized or ""


def caption_token_count(request: dict[str, Any]) -> int:
    caption = request.get("caption", "")
    return len(TOKEN_RE.findall(caption)) if isinstance(caption, str) else 0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    index = round((len(values) - 1) * q)
    return sorted(values)[index]


def trimmed_mean(values: list[float], trim: float = 0.1) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = int(len(ordered) * trim)
    trimmed = ordered[k : len(ordered) - k] if len(ordered) - 2 * k > 0 else ordered
    return statistics.fmean(trimmed)


def empty_category_counts() -> dict[str, int]:
    return {category: 0 for category in UNIT_CATEGORIES}


def unit_records(group: Any) -> list[dict[str, str]]:
    """Normalize both legacy category arrays and v2 atomic record arrays."""
    records: list[dict[str, str]] = []
    if isinstance(group, dict):
        for category in UNIT_CATEGORIES:
            items = group.get(category, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, str) and item.strip():
                    records.append({"category": category, "unit": item.strip(), "span": item.strip(), "target": ""})
        return records
    if isinstance(group, list):
        for item in group:
            if not isinstance(item, dict):
                continue
            category = item.get("category")
            unit = item.get("unit")
            if category not in UNIT_CATEGORIES or not isinstance(unit, str) or not unit.strip():
                continue
            span = item.get("span", "")
            target = item.get("target", "")
            records.append(
                {
                    "category": category,
                    "unit": unit.strip(),
                    "span": span.strip() if isinstance(span, str) else "",
                    "target": target.strip() if isinstance(target, str) else "",
                }
            )
    return records


def count_unit_group(group: Any) -> tuple[int, dict[str, int]]:
    counts = {category: 0 for category in UNIT_CATEGORIES}
    for record in unit_records(group):
        counts[record["category"]] += 1
    return sum(counts.values()), counts


def count_deduped_unit_group(group: Any) -> tuple[int, dict[str, int], int, int]:
    counts = empty_category_counts()
    seen: set[str] = set()
    duplicate = 0
    suspicious = 0
    for record in unit_records(group):
        norm = normalize_unit(record["unit"])
        if not norm:
            continue
        key = f"{record['category']}|{norm}|{normalize_key_part(record.get('target', ''))}"
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        category = record["category"]
        if category == "count" and norm in ARTICLE_UNITS:
            suspicious += 1
            continue
        if category == "text_rendering" and any(
            marker in norm for marker in ["no text", "no visible", "not visible", "without text"]
        ):
            suspicious += 1
            continue
        counts[category] += 1
    return sum(counts.values()), counts, duplicate, suspicious


def add_counts(dst: Counter[str], counts: dict[str, int], prefix: str) -> None:
    for category, count in counts.items():
        dst[f"{prefix}_{category}"] += count


def summarize_claimed_row(parsed: dict[str, Any], request: dict[str, Any]) -> list[tuple[str, Counter[str]]]:
    surface = request.get("surface", "unknown")
    total, counts = count_unit_group(parsed.get("claimed_units"))
    dedup_total, dedup_counts, duplicate, suspicious = count_deduped_unit_group(parsed.get("claimed_units"))
    tokens = caption_token_count(request)
    counter: Counter[str] = Counter()
    counter["captions"] += 1
    counter["claimed_total"] += total
    counter["claimed_dedup_total"] += dedup_total
    counter["duplicate_units"] += duplicate
    counter["suspicious_units"] += suspicious
    counter["caption_tokens"] += tokens
    counter["rows_with_duplicate"] += int(duplicate > 0)
    counter["rows_with_suspicious"] += int(suspicious > 0)
    add_counts(counter, counts, "claimed")
    add_counts(counter, dedup_counts, "claimed_dedup")
    return [(surface, counter)]


def summarize_grounded_row(parsed: dict[str, Any], request: dict[str, Any]) -> list[tuple[str, Counter[str]]]:
    rows = []
    for result in parsed.get("results", []) if isinstance(parsed, dict) else []:
        caption_id = result.get("caption_id")
        surface = None
        for caption in request.get("captions", []):
            if caption.get("caption_id") == caption_id:
                surface = caption.get("surface")
                break
        surface = surface or str(caption_id or "unknown")
        grounded_total, grounded_counts = count_unit_group(result.get("grounded_units"))
        unsupported_total, unsupported_counts = count_unit_group(result.get("unsupported_units"))
        uncertain_total, uncertain_counts = count_unit_group(result.get("uncertain_units"))
        claimed_total = grounded_total + unsupported_total + uncertain_total
        counter: Counter[str] = Counter()
        counter["captions"] += 1
        counter["claimed_total"] += claimed_total
        counter["grounded_total"] += grounded_total
        counter["unsupported_total"] += unsupported_total
        counter["uncertain_total"] += uncertain_total
        counter[f"overall_{result.get('overall', 'missing')}"] += 1
        add_counts(counter, grounded_counts, "grounded")
        add_counts(counter, unsupported_counts, "unsupported")
        add_counts(counter, uncertain_counts, "uncertain")
        rows.append((surface, counter))
    return rows


def merge(dst: Counter[str], src: Counter[str]) -> None:
    for key, value in src.items():
        dst[key] += value


def finalize(counter: Counter[str]) -> dict[str, Any]:
    captions = max(counter["captions"], 1)
    claimed = counter["claimed_total"]
    output: dict[str, Any] = dict(counter)
    output["claimed_per_caption"] = claimed / captions
    output["claimed_dedup_per_caption"] = counter["claimed_dedup_total"] / captions
    output["claimed_dedup_per_100_tokens"] = (
        100 * counter["claimed_dedup_total"] / counter["caption_tokens"] if counter["caption_tokens"] else None
    )
    output["duplicate_units_per_caption"] = counter["duplicate_units"] / captions
    output["suspicious_units_per_caption"] = counter["suspicious_units"] / captions
    output["duplicate_row_rate"] = counter["rows_with_duplicate"] / captions
    output["suspicious_row_rate"] = counter["rows_with_suspicious"] / captions
    output["grounded_precision"] = counter["grounded_total"] / claimed if claimed else None
    output["unsupported_rate"] = counter["unsupported_total"] / claimed if claimed else None
    output["uncertain_rate"] = counter["uncertain_total"] / claimed if claimed else None
    for category in UNIT_CATEGORIES:
        output[f"claimed_{category}_per_caption"] = counter[f"claimed_{category}"] / captions
        output[f"claimed_dedup_{category}_per_caption"] = counter[f"claimed_dedup_{category}"] / captions
        denom = counter[f"grounded_{category}"] + counter[f"unsupported_{category}"] + counter[f"uncertain_{category}"]
        if denom:
            output[f"grounded_{category}_precision"] = counter[f"grounded_{category}"] / denom
            output[f"unsupported_{category}_rate"] = counter[f"unsupported_{category}"] / denom
    return output


def main() -> int:
    args = parse_args()
    by_surface: dict[str, Counter[str]] = defaultdict(Counter)
    per_surface_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    status = Counter()
    input_paths = [Path(args.input), *[Path(item) for item in args.include]]
    if args.latest_by_request:
        latest: dict[str, dict[str, Any]] = {}
        for input_path in input_paths:
            with input_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    request_id = row.get("request_id")
                    if isinstance(request_id, str):
                        latest[request_id] = row
        rows = list(latest.values())
    else:
        rows = []
        for input_path in input_paths:
            with input_path.open("r", encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    for row in rows:
            status["responses"] += 1
            if not row.get("ok"):
                status["bad"] += 1
                continue
            parsed = row.get("parsed")
            request = row.get("request", {})
            items = (
                summarize_claimed_row(parsed, request)
                if args.mode == "claimed"
                else summarize_grounded_row(parsed, request)
            )
            for surface, counter in items:
                merge(by_surface[surface], counter)
                merge(by_surface["__all__"], counter)
                status["captions"] += counter["captions"]
                if args.mode == "claimed":
                    tokens = max(counter["caption_tokens"], 1)
                    for key_surface in [surface, "__all__"]:
                        per_surface_values[key_surface]["claimed"].append(float(counter["claimed_total"]))
                        per_surface_values[key_surface]["claimed_dedup"].append(float(counter["claimed_dedup_total"]))
                        per_surface_values[key_surface]["claimed_dedup_per_100_tokens"].append(
                            100.0 * counter["claimed_dedup_total"] / tokens
                        )
                        per_surface_values[key_surface]["caption_tokens"].append(float(counter["caption_tokens"]))
    surfaces = {surface: finalize(counter) for surface, counter in sorted(by_surface.items())}
    for surface, metrics in per_surface_values.items():
        if surface not in surfaces:
            continue
        for name, values in metrics.items():
            surfaces[surface][f"{name}_median"] = statistics.median(values) if values else None
            surfaces[surface][f"{name}_p25"] = percentile(values, 0.25)
            surfaces[surface][f"{name}_p75"] = percentile(values, 0.75)
            surfaces[surface][f"{name}_trimmed_mean"] = trimmed_mean(values)
    payload = {
        "input": args.input,
        "mode": args.mode,
        "status": dict(status),
        "surfaces": surfaces,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
