#!/usr/bin/env python3
"""Summarize CBU VQA response JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ANSWERS = ["yes", "no", "uncertain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CBU VQA responses")
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
        help="Use only the last response per request_id.",
    )
    return parser.parse_args()


def load_rows(paths: list[Path], latest_by_request: bool) -> list[dict[str, Any]]:
    if not latest_by_request:
        rows: list[dict[str, Any]] = []
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        return rows

    latest: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                request_id = row.get("request_id")
                if isinstance(request_id, str):
                    latest[request_id] = row
    return list(latest.values())


def question_lookup(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    request = row.get("request", {})
    return {
        question["question_id"]: question
        for question in request.get("questions", [])
        if isinstance(question, dict) and isinstance(question.get("question_id"), str)
    }


def add_rates(stats: dict[str, Any]) -> dict[str, Any]:
    total = stats.get("questions", 0)
    for answer in ANSWERS:
        stats[f"{answer}_rate"] = stats.get(answer, 0) / total if total else 0.0
    stats["support_rate"] = stats.get("yes", 0) / total if total else 0.0
    stats["risk_rate"] = stats.get("no", 0) / total if total else 0.0
    stats["uncertainty_rate"] = stats.get("uncertain", 0) / total if total else 0.0
    return stats


def main() -> int:
    args = parse_args()
    paths = [Path(args.input), *[Path(item) for item in args.include]]
    rows = load_rows(paths, args.latest_by_request)

    surface_stats: dict[str, Counter[str]] = defaultdict(Counter)
    category_stats: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    responses = 0
    ok = 0
    for row in rows:
        responses += 1
        request = row.get("request", {})
        surface = request.get("surface", "__unknown__")
        surface_stats[surface]["responses"] += 1
        if not row.get("ok"):
            surface_stats[surface]["bad"] += 1
            if len(examples["bad_response"]) < 20:
                examples["bad_response"].append(
                    {
                        "surface": surface,
                        "caption_id": request.get("caption_id"),
                        "error": row.get("parse_error") or row.get("schema_error") or row.get("error"),
                    }
                )
            continue
        ok += 1
        surface_stats[surface]["ok"] += 1
        lookup = question_lookup(row)
        for result in row.get("parsed", {}).get("question_results", []):
            if not isinstance(result, dict):
                continue
            question_id = result.get("question_id")
            answer = result.get("answer")
            if answer not in ANSWERS:
                continue
            question = lookup.get(question_id, {})
            category = question.get("category", "__unknown__")
            surface_stats[surface]["questions"] += 1
            surface_stats[surface][answer] += 1
            category_stats[category]["questions"] += 1
            category_stats[category][answer] += 1
            if answer in {"no", "uncertain"} and len(examples[answer]) < 20:
                examples[answer].append(
                    {
                        "surface": surface,
                        "caption_id": request.get("caption_id"),
                        "category": category,
                        "question": question.get("question"),
                        "answer": answer,
                        "confidence": result.get("confidence"),
                        "evidence": result.get("evidence"),
                    }
                )

    out = {
        "input": args.input,
        "include": args.include,
        "latest_by_request": args.latest_by_request,
        "responses": responses,
        "ok": ok,
        "bad": responses - ok,
        "surfaces": {surface: add_rates(dict(counter)) for surface, counter in surface_stats.items()},
        "categories": {category: add_rates(dict(counter)) for category, counter in category_stats.items()},
        "examples": examples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "responses": responses, "ok": ok, "bad": responses - ok}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
