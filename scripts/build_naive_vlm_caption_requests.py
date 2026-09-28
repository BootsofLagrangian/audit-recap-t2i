#!/usr/bin/env python3
"""Build one-image-per-row naive VLM caption requests from an existing image request JSONL.

The corrected CC12M bridge artifacts already contain vetted image identity
metadata in the image-conditioned CBU-VQA request file. This helper deduplicates
that file down to one request per image/source row and records a no-system-prompt
captioning prompt for a fair captioner-policy baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = "Please generate a detailed caption of this image. Please be as descriptive as possible."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL containing image_path/image_url fields")
    parser.add_argument("--output", required=True)
    parser.add_argument("--surface", default="naive_qwen35_cc12m")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--dedupe-key",
        choices=["source_row", "image_path", "image_url"],
        default="source_row",
        help="Image identity field used to collapse repeated metric requests.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def image_path_for(row: dict[str, Any]) -> str | None:
    image = row.get("image") if isinstance(row.get("image"), dict) else {}
    value = row.get("image_path") or image.get("local_abs_path")
    return value if isinstance(value, str) and value else None


def image_url_for(row: dict[str, Any]) -> str | None:
    image = row.get("image") if isinstance(row.get("image"), dict) else {}
    value = row.get("image_url") or image.get("url")
    return value if isinstance(value, str) and value else None


def key_for(row: dict[str, Any], mode: str) -> str | None:
    if mode == "source_row":
        value = row.get("source_row")
        return str(value) if value is not None else None
    if mode == "image_path":
        return image_path_for(row)
    return image_url_for(row)


def request_id(surface: str, key: str, prompt: str) -> str:
    raw = f"naive_vlm_caption:{surface}:{key}:{prompt}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def build_row(row: dict[str, Any], key: str, args: argparse.Namespace, emitted_index: int) -> dict[str, Any]:
    path = image_path_for(row)
    url = image_url_for(row)
    image = row.get("image") if isinstance(row.get("image"), dict) else {}
    out_image = {
        "local_abs_path": path,
        "url": url,
        "source_image_path": path,
        "source_image_url": url,
        "materialization_source": image.get("materialization_source"),
        "dataset_key": image.get("dataset_key"),
        "sample_id": image.get("sample_id"),
        "stable_image_id": image.get("stable_image_id"),
        "width": image.get("width") or row.get("width"),
        "height": image.get("height") or row.get("height"),
        "bytes": image.get("bytes"),
        "storage_uri": image.get("storage_uri"),
    }
    return {
        "request_id": request_id(args.surface, key, args.prompt),
        "task": "naive_vlm_caption",
        "surface": args.surface,
        "caption_id": f"{args.surface}:{key}",
        "source_row": row.get("source_row"),
        "emitted_index": emitted_index,
        "prompt": args.prompt,
        "system_prompt": None,
        "messages_policy": "single_user_message_with_image_no_system_prompt",
        "image_path": path,
        "image_url": url,
        "image": out_image,
        "pair_key": row.get("pair_key"),
        "public_lookup_key": row.get("public_lookup_key"),
        "family": row.get("family"),
        "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
    }


def main() -> int:
    args = parse_args()
    rows = iter_jsonl(Path(args.input))
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = key_for(row, args.dedupe_key)
        if key is None or key in selected:
            continue
        if not image_path_for(row) and not image_url_for(row):
            continue
        selected[key] = row
        if args.max_images is not None and len(selected) >= args.max_images:
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for emitted_index, (key, row) in enumerate(selected.items()):
            handle.write(json.dumps(build_row(row, key, args, emitted_index), ensure_ascii=False) + "\n")

    manifest = {
        "task": "naive_vlm_caption",
        "input": args.input,
        "output": str(output),
        "surface": args.surface,
        "prompt": args.prompt,
        "system_prompt": None,
        "messages_policy": "single_user_message_with_image_no_system_prompt",
        "dedupe_key": args.dedupe_key,
        "input_rows": len(rows),
        "requests": len(selected),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), "requests": len(selected)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
