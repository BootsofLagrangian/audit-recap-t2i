#!/usr/bin/env python3
"""Build exact-unit image audit requests from claimed-CBU responses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are a strict visual grounding judge for text-to-image training captions.\n"
    "Return only valid compact JSON. Judge only whether each provided caption-derived unit "
    "is visibly supported by the image."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exact-unit grounded-CBU verification requests")
    parser.add_argument("--claimed-responses", required=True)
    parser.add_argument("--source-jsonl", required=True, help="Fair-slice JSONL used to build the claimed requests")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-units-per-request", type=int, default=None, help="Debug cap only; omit for main audit")
    parser.add_argument("--image-path-field", default=None)
    parser.add_argument(
        "--require-local-image",
        action="store_true",
        help="Skip rows without a local image path. Use for reproducible image-grounded audits.",
    )
    parser.add_argument(
        "--surface-filter",
        default=None,
        help="If set, keep only claimed responses whose request.surface exactly matches this value.",
    )
    return parser.parse_args()


def iter_ok_claims(path: Path, surface_filter: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            parsed = row.get("parsed")
            request = row.get("request", {})
            if surface_filter is not None and request.get("surface") != surface_filter:
                continue
            units = parsed.get("claimed_units") if isinstance(parsed, dict) else None
            if not row.get("ok") or not isinstance(units, list) or not units:
                continue
            rows.append({"request": request, "parsed": parsed})
    return rows


def load_source_rows(source_jsonl: Path, needed: set[int]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with source_jsonl.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in needed and line.strip():
                out[index] = json.loads(line)
                if len(out) == len(needed):
                    break
    return out


def image_fields(source_row: dict[str, Any], image_path_field: str | None) -> dict[str, Any]:
    image = source_row.get("image") if isinstance(source_row.get("image"), dict) else {}
    metadata = source_row.get("metadata") if isinstance(source_row.get("metadata"), dict) else {}
    local_record = source_row.get("local_record") if isinstance(source_row.get("local_record"), dict) else {}
    public_record = source_row.get("public_record") if isinstance(source_row.get("public_record"), dict) else {}
    if image_path_field:
        image_path = source_row.get(image_path_field)
    else:
        image_path = (
            source_row.get("image_path")
            or source_row.get("image_abs_path")
            or local_record.get("image_abs_path")
            or image.get("local_abs_path")
        )
    image_url = (
        image.get("url")
        or source_row.get("url")
        or source_row.get("image_url")
        or metadata.get("canonical_url")
        or public_record.get("url")
        or source_row.get("pair_key")
    )
    return {
        "image_url": image_url,
        "image_path": image_path,
        "image_sha256": image.get("sha256") or source_row.get("sha256"),
        "pair_id": source_row.get("pair_id"),
        "pair_key": source_row.get("pair_key"),
        "public_lookup_key": source_row.get("public_lookup_key"),
        "family": source_row.get("family"),
    }


def normalize_unit(raw: dict[str, Any], caption_id: str, index: int) -> dict[str, str]:
    return {
        "unit_id": f"{caption_id}:u{index:04d}",
        "category": str(raw.get("category", "")),
        "unit": str(raw.get("unit", "")),
        "span": str(raw.get("span", "")),
        "target": str(raw.get("target", "")),
    }


def user_prompt(caption: str, units: list[dict[str, str]]) -> str:
    unit_json = json.dumps(units, ensure_ascii=False, separators=(",", ":"))
    return (
        "Verify the visual grounding of each provided caption-derived unit.\n"
        "Rules:\n"
        "- Do not add, remove, split, merge, rename, or reinterpret unit_id values.\n"
        "- Use grounded when the image visibly supports the unit.\n"
        "- Use unsupported when the image contradicts the unit or lacks visible support.\n"
        "- Use uncertain when the unit is too fine-grained, occluded, unreadable, or visually ambiguous.\n"
        "- Use invalid_text_unit only when the unit is not a meaningful visual claim from the caption.\n"
        "- Use not_a_visual_claim only for non-visual metadata or captioner-language units.\n"
        "- Keep evidence short; cite only visible image evidence.\n"
        "Return JSON with caption_id and unit_results, exactly one result for each input unit_id.\n\n"
        f"caption={caption}\n"
        f"claimed_units={unit_json}"
    )


def main() -> int:
    args = parse_args()
    claims = iter_ok_claims(Path(args.claimed_responses), args.surface_filter)
    if args.max_requests is not None:
        claims = claims[: args.max_requests]
    needed = {int(item["request"]["source_row"]) for item in claims if item["request"].get("source_row") is not None}
    sources = load_source_rows(Path(args.source_jsonl), needed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with output.open("w", encoding="utf-8") as handle:
        for item in claims:
            req = item["request"]
            source_row = sources.get(int(req["source_row"]))
            if source_row is None:
                skipped += 1
                continue
            image_info = image_fields(source_row, args.image_path_field)
            if args.require_local_image and (
                not image_info.get("image_path") or not Path(str(image_info["image_path"])).is_file()
            ):
                skipped += 1
                continue
            caption_id = str(item["parsed"].get("caption_id") or req.get("caption_id"))
            units = [
                normalize_unit(raw, caption_id, index)
                for index, raw in enumerate(item["parsed"].get("claimed_units", []))
                if isinstance(raw, dict)
            ]
            if args.max_units_per_request is not None:
                units = units[: args.max_units_per_request]
            if not units:
                skipped += 1
                continue
            row = {
                "request_id": hashlib.blake2b(
                    f"grounded_cbu_verify_v2:{req.get('request_id')}:{caption_id}".encode("utf-8"),
                    digest_size=16,
                ).hexdigest(),
                "task": "grounded_cbu_verify_v2",
                "surface": req.get("surface"),
                "caption_id": caption_id,
                "source_row": req.get("source_row"),
                "token_budget": req.get("token_budget"),
                "caption": req.get("caption"),
                "source_caption": req.get("source_caption"),
                "claimed_units": units,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt(str(req.get("caption", "")), units),
                **image_info,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    manifest = {
        "task": "grounded_cbu_verify_v2",
        "claimed_responses": args.claimed_responses,
        "source_jsonl": args.source_jsonl,
        "output": str(output),
        "requests": written,
        "skipped": skipped,
        "max_requests": args.max_requests,
        "max_units_per_request": args.max_units_per_request,
        "surface_filter": args.surface_filter,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
