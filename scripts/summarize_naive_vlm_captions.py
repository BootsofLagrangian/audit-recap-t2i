#!/usr/bin/env python3
"""Export naive VLM caption responses as a caption-surface JSONL and summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--surface", default="naive_qwen35_cc12m")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokens(text: str) -> int:
    return len(text.split())


def main() -> int:
    args = parse_args()
    responses = read_jsonl(Path(args.responses))
    latest: dict[str, dict[str, Any]] = {}
    for row in responses:
        request_id = row.get("request_id")
        if isinstance(request_id, str):
            latest[request_id] = row

    out_rows: list[dict[str, Any]] = []
    bad = 0
    for row in latest.values():
        request = row.get("request") if isinstance(row.get("request"), dict) else {}
        caption = row.get("caption")
        if not row.get("ok") or not isinstance(caption, str) or not caption.strip():
            bad += 1
            continue
        out_rows.append(
            {
                "surface": args.surface,
                "caption_id": request.get("caption_id") or f"{args.surface}:{request.get('source_row')}",
                "source_row": request.get("source_row"),
                "caption": caption.strip(),
                "prompt": request.get("prompt"),
                "system_prompt": None,
                "messages_policy": request.get("messages_policy"),
                "image": request.get("image") if isinstance(request.get("image"), dict) else {},
                "image_path": request.get("image_path"),
                "image_url": request.get("image_url"),
                "pair_key": request.get("pair_key"),
                "public_lookup_key": request.get("public_lookup_key"),
                "family": request.get("family"),
                "metadata": request.get("metadata") if isinstance(request.get("metadata"), dict) else {},
            }
        )
    out_rows.sort(
        key=lambda row: (
            row["source_row"] if isinstance(row.get("source_row"), int) else 10**18,
            str(row.get("caption_id") or ""),
        )
    )

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = [tokens(row["caption"]) for row in out_rows]
    summary = {
        "responses": len(responses),
        "unique_requests": len(latest),
        "captions": len(out_rows),
        "bad": bad,
        "surface": args.surface,
        "output_jsonl": str(output_jsonl),
        "prompt": out_rows[0]["prompt"] if out_rows else None,
        "system_prompt": None,
        "messages_policy": "single_user_message_with_image_no_system_prompt",
        "token_mean": sum(lengths) / len(lengths) if lengths else 0,
        "token_median": median(lengths) if lengths else 0,
        "token_min": min(lengths) if lengths else 0,
        "token_max": max(lengths) if lengths else 0,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
