#!/usr/bin/env python3
"""Build text-only claimed-CBU extraction requests from caption JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
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


SYSTEM_PROMPT = (
    "You extract atomic controllable visual content units from captions for text-to-image training-data evaluation.\n"
    "Return only valid compact JSON. Extract only facts explicitly claimed by the caption. "
    "Do not infer image content beyond the caption."
)


CBU_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "caption_id": {"type": "string"},
        "claimed_units": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": UNIT_CATEGORIES},
                    "unit": {"type": "string", "maxLength": 80},
                    "span": {"type": "string", "maxLength": 120},
                    "target": {"type": "string", "maxLength": 80},
                },
                "required": ["category", "unit", "span", "target"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["caption_id", "claimed_units"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build claimed-CBU extraction request JSONL")
    parser.add_argument("--input", required=True, help="Caption JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field", default="caption")
    parser.add_argument("--id-field", default=None)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--sample-records", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--max-caption-chars", type=int, default=1800)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Optional whitespace token prefix budget for length-controlled CBU@B requests",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="Optional maximum atomic units in the JSON schema; use only for stress/debug caps",
    )
    return parser.parse_args()


def stable_float(*parts: object) -> float:
    raw = ":".join(str(part) for part in parts)
    digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def iter_rows(args: argparse.Namespace) -> list[tuple[int, str | None, str]]:
    rows: list[tuple[int, str | None, str]] = []
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if args.max_records is not None and args.sample_records is None and len(rows) >= args.max_records:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get(args.text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            row_id = row.get(args.id_field) if args.id_field else None
            rows.append((row_index, str(row_id) if row_id is not None else None, text))
    if args.sample_records is not None:
        rows.sort(key=lambda item: stable_float(args.sample_seed, args.surface, item[0], item[1] or ""))
        rows = rows[: args.sample_records]
        rows.sort(key=lambda item: item[0])
    return rows


def schema_with_max_units(max_units: int | None) -> dict[str, Any]:
    schema = json.loads(json.dumps(CBU_JSON_SCHEMA))
    if max_units is not None:
        schema["properties"]["claimed_units"]["maxItems"] = max_units
    return schema


def build_user_prompt(caption_id: str, caption: str, max_caption_chars: int, max_units: int | None) -> str:
    clipped = caption[:max_caption_chars].replace("\n", " ")
    schema = json.dumps(schema_with_max_units(max_units), ensure_ascii=False, separators=(",", ":"))
    categories = ", ".join(UNIT_CATEGORIES)
    return (
        "Extract caption-claimed controllable visual units as atomic records.\n"
        f"Unit categories: {categories}.\n"
        "Rules:\n"
        "- Each record must contain exactly one visual control fact.\n"
        "- Use each semantic fact once; choose the single best category.\n"
        "- unit is a short canonical phrase, not a full clause.\n"
        "- span is the shortest caption span supporting the unit.\n"
        "- target is the object or scene element modified by the unit; use \"scene\" when global.\n"
        "- relation units must include both the relation and participating objects; "
        "do not output lone verbs or prepositions.\n"
        "- count units must attach a number to a target object; never output articles such as a, an, or the.\n"
        "- text_rendering units are only visible rendered text explicitly claimed by the caption; "
        "absent text claims are not units.\n"
        "- Do not output negative or absent facts, metadata, captioner phrases, or duplicate paraphrases.\n"
        "- Keep text_rendering units short; do not copy long copyright, table, or legal text blocks.\n"
        "- Use [] when the caption contains no controllable visual units.\n"
        "Return only JSON matching this schema:\n"
        f"{schema}\n\n"
        f"caption_id={caption_id}\ncaption={clipped}"
    )


def apply_token_budget(caption: str, token_budget: int | None) -> str:
    if token_budget is None:
        return caption
    return " ".join(caption.split()[:token_budget])


def main() -> int:
    args = parse_args()
    if args.max_records is not None and args.sample_records is not None:
        raise SystemExit("--max-records and --sample-records are mutually exclusive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = iter_rows(args)
    with output.open("w", encoding="utf-8") as handle:
        for emitted_index, (source_row, row_id, caption) in enumerate(rows):
            caption_id = row_id or f"{args.surface}:{source_row}"
            request_caption = apply_token_budget(caption, args.token_budget)
            budget_tag = f"b{args.token_budget}" if args.token_budget is not None else "full"
            request_id = hashlib.blake2b(
                f"claimed_cbu_v2:{budget_tag}:{args.surface}:{source_row}:{caption_id}".encode("utf-8"),
                digest_size=16,
            ).hexdigest()
            row = {
                "request_id": request_id,
                "task": "claimed_cbu_v2",
                "token_budget": args.token_budget,
                "surface": args.surface,
                "caption_id": caption_id,
                "source_row": source_row,
                "emitted_index": emitted_index,
                "caption": request_caption,
                "source_caption": caption,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": build_user_prompt(caption_id, request_caption, args.max_caption_chars, args.max_units),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "task": "claimed_cbu_v2",
        "input": args.input,
        "output": str(output),
        "surface": args.surface,
        "text_field": args.text_field,
        "id_field": args.id_field,
        "max_records": args.max_records,
        "sample_records": args.sample_records,
        "sample_seed": args.sample_seed,
        "token_budget": args.token_budget,
        "max_units": args.max_units,
        "rows": len(rows),
        "schema": schema_with_max_units(args.max_units),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), "requests": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
