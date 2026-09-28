#!/usr/bin/env python3
"""Build VQA-style yes/no question requests from grounded-CBU request JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You are a strict visual question answering judge.
Return only valid compact JSON. Answer each question using only visible image evidence."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VQA-style requests from CBU verification requests")
    parser.add_argument("--input", required=True, help="grounded-CBU request JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--sample-records", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--max-questions-per-request", type=int, default=None)
    return parser.parse_args()


def stable_float(*parts: object) -> float:
    raw = ":".join(str(part) for part in parts)
    digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def question_for(unit: dict[str, Any]) -> str:
    category = str(unit.get("category", ""))
    phrase = str(unit.get("unit", "")).strip()
    target = str(unit.get("target", "")).strip()
    if category == "text_rendering":
        return f"Is the rendered text claim '{phrase}' visibly supported by the image?"
    if target:
        return f"Is the visual claim '{target}: {phrase}' supported by the image?"
    return f"Is the visual claim '{phrase}' supported by the image?"


def user_prompt(questions: list[dict[str, str]]) -> str:
    question_json = json.dumps(questions, ensure_ascii=False, separators=(",", ":"))
    return (
        "Answer each visual question using only the image.\n"
        "Rules:\n"
        "- Do not use any caption text or outside knowledge.\n"
        "- Use yes when the image visibly supports the question.\n"
        "- Use no when the image contradicts the question or lacks visible support.\n"
        "- Use uncertain when the question is too fine-grained, occluded, unreadable, or visually ambiguous.\n"
        "- Keep evidence short and grounded in visible image content.\n"
        "- Return exactly one answer for each input question_id.\n\n"
        f"questions={question_json}"
    )


def iter_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            if args.max_requests is not None and args.sample_records is None and len(rows) >= args.max_requests:
                break
            if line.strip():
                rows.append(json.loads(line))
    if args.sample_records is not None:
        rows.sort(key=lambda row: stable_float(args.sample_seed, row.get("request_id", "")))
        rows = rows[: args.sample_records]
        rows.sort(key=lambda row: row.get("source_row", 0))
    return rows


def main() -> int:
    args = parse_args()
    rows = iter_rows(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            units = row.get("claimed_units", [])
            if args.max_questions_per_request is not None:
                units = units[: args.max_questions_per_request]
            questions = [
                {
                    "question_id": str(unit["unit_id"]),
                    "category": str(unit.get("category", "")),
                    "question": question_for(unit),
                }
                for unit in units
                if isinstance(unit, dict) and isinstance(unit.get("unit_id"), str)
            ]
            if not questions:
                skipped += 1
                continue
            request_id = hashlib.blake2b(
                f"cbu_vqa_v1:{row.get('request_id')}:{row.get('caption_id')}".encode("utf-8"),
                digest_size=16,
            ).hexdigest()
            out = {
                "request_id": request_id,
                "task": "cbu_vqa_v1",
                "surface": row.get("surface"),
                "caption_id": row.get("caption_id"),
                "source_row": row.get("source_row"),
                "token_budget": row.get("token_budget"),
                "questions": questions,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt(questions),
                "image_url": row.get("image_url"),
                "image_path": row.get("image_path"),
                "image_sha256": row.get("image_sha256"),
                "pair_id": row.get("pair_id"),
                "pair_key": row.get("pair_key"),
                "public_lookup_key": row.get("public_lookup_key"),
                "family": row.get("family"),
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
    manifest = {
        "task": "cbu_vqa_v1",
        "input": args.input,
        "output": str(output),
        "requests": written,
        "skipped": skipped,
        "sample_records": args.sample_records,
        "sample_seed": args.sample_seed,
        "max_questions_per_request": args.max_questions_per_request,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
