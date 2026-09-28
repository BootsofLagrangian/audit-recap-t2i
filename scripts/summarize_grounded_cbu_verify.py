#!/usr/bin/env python3
"""Summarize exact-unit grounded-CBU verification responses."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATUSES = [
    "grounded",
    "unsupported",
    "uncertain",
    "invalid_text_unit",
    "not_a_visual_claim",
    "image_unavailable",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize grounded-CBU verification responses")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Additional response JSONL to merge before latest-by-request summarization.",
    )
    parser.add_argument(
        "--latest-by-request",
        action="store_true",
        help="Use only the last response per request_id. Useful for append/resume retry logs.",
    )
    return parser.parse_args()


def unit_lookup(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {unit["unit_id"]: unit for unit in row.get("claimed_units", []) if isinstance(unit, dict) and "unit_id" in unit}


def add_rates(stats: dict[str, Any]) -> dict[str, Any]:
    valid = stats.get("valid_units", 0)
    visual = stats.get("visual_units", 0)
    for status in STATUSES:
        stats[f"{status}_rate_all"] = stats.get(status, 0) / valid if valid else 0.0
        stats[f"{status}_rate_visual"] = stats.get(status, 0) / visual if visual else 0.0
    stats["grounded_precision"] = stats.get("grounded", 0) / visual if visual else 0.0
    stats["unsupported_rate"] = stats.get("unsupported", 0) / visual if visual else 0.0
    stats["uncertain_rate"] = stats.get("uncertain", 0) / visual if visual else 0.0
    return stats


def main() -> int:
    args = parse_args()
    surface_stats: dict[str, Counter[str]] = defaultdict(Counter)
    category_stats: dict[str, Counter[str]] = defaultdict(Counter)
    status_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0
    ok = 0
    rows: list[dict[str, Any]] = []
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
        total += 1
        surface = row.get("request", {}).get("surface", "__unknown__")
        surface_stats[surface]["responses"] += 1
        if not row.get("ok"):
            surface_stats[surface]["bad"] += 1
            continue
        ok += 1
        surface_stats[surface]["ok"] += 1
        lookup = unit_lookup(row.get("request", {}))
        for result in row.get("parsed", {}).get("unit_results", []):
            unit_id = result.get("unit_id")
            unit = lookup.get(unit_id, {})
            category = unit.get("category", "__unknown__")
            status = result.get("status", "__bad_status__")
            surface_stats[surface]["valid_units"] += 1
            surface_stats[surface][status] += 1
            category_stats[category]["valid_units"] += 1
            category_stats[category][status] += 1
            if status in {"grounded", "unsupported", "uncertain"}:
                surface_stats[surface]["visual_units"] += 1
                category_stats[category]["visual_units"] += 1
            if status in {"unsupported", "uncertain", "invalid_text_unit", "not_a_visual_claim"} and len(status_examples[status]) < 20:
                status_examples[status].append(
                    {
                        "surface": surface,
                        "caption_id": row.get("request", {}).get("caption_id"),
                        "category": category,
                        "unit": unit.get("unit"),
                        "target": unit.get("target"),
                        "status": status,
                        "evidence": result.get("evidence"),
                    }
                )
    surfaces = {surface: add_rates(dict(counter)) for surface, counter in surface_stats.items()}
    categories = {category: add_rates(dict(counter)) for category, counter in category_stats.items()}
    out = {
        "input": args.input,
        "responses": total,
        "ok": ok,
        "bad": total - ok,
        "surfaces": surfaces,
        "categories": categories,
        "examples": status_examples,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": args.output, "responses": total, "ok": ok, "bad": total - ok}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
