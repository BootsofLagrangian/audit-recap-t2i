#!/usr/bin/env python3
"""Rehydrate CC12M local images referenced by request JSONL rows.

Rows produced from the corrected CC12M bridge carry historical local image
filenames like ``00153__000356174.jpg``. The prefix is the WDS tar shard and the
stem after ``__`` is the tar member key. This script extracts those images from
the canonical CC12M WDS tar store and rewrites request rows to the new local
image path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--wds-root", default="data/cc12m-wds")
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def shard_key_from_path(path_text: str) -> tuple[str, str] | None:
    name = Path(path_text).name
    if "__" not in name:
        return None
    shard, rest = name.split("__", 1)
    suffix = Path(rest).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        return None
    key = Path(rest).stem
    if not shard or not key:
        return None
    return shard, key


def scan_members(tar_path: Path, keys: set[str]) -> dict[str, str]:
    matches: dict[str, str] = {}
    proc = subprocess.Popen(["tar", "-tf", str(tar_path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        member = line.strip()
        suffix = Path(member).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            continue
        key = Path(member).stem
        if key in keys and key not in matches:
            matches[key] = member
            if len(matches) == len(keys):
                proc.kill()
                break
    proc.wait()
    return matches


def extract_shard(wds_root: Path, shard: str, keys: set[str], image_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    tar_path = wds_root / f"{shard}.tar"
    if not tar_path.exists():
        return {}, {key: f"missing_tar:{tar_path}" for key in keys}
    matches = scan_members(tar_path, keys)
    image_dir.mkdir(parents=True, exist_ok=True)
    materialized: dict[str, str] = {}
    failures: dict[str, str] = {}
    for key in sorted(keys):
        member = matches.get(key)
        if member is None:
            failures[key] = f"missing_member:{key}"
            continue
        suffix = Path(member).suffix.lower()
        out_path = image_dir / f"{shard}__{key}{suffix}"
        if not out_path.exists():
            data = subprocess.check_output(["tar", "-xOf", str(tar_path), member])
            out_path.write_bytes(data)
        materialized[key] = str(out_path)
    return materialized, failures


def update_row(row: dict[str, Any], new_path: str) -> dict[str, Any]:
    out = dict(row)
    image = dict(out.get("image") if isinstance(out.get("image"), dict) else {})
    image["local_abs_path"] = new_path
    image["materialization_source"] = "cc12m_202603_wds_rehydrated_from_request_path"
    out["image"] = image
    out["image_path"] = new_path
    return out


def main() -> int:
    args = parse_args()
    rows = read_jsonl(Path(args.input))
    image_dir = Path(args.image_dir)
    wanted: dict[str, set[str]] = defaultdict(set)
    row_parts: dict[int, tuple[str, str]] = {}
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        path = image_path_for(row)
        parts = shard_key_from_path(path) if path else None
        if parts is None:
            failures.append(
                {
                    "row": index,
                    "request_id": row.get("request_id"),
                    "reason": "missing_or_unparseable_image_path",
                    "image_path": path,
                }
            )
            continue
        shard, key = parts
        wanted[shard].add(key)
        row_parts[index] = (shard, key)

    by_part: dict[tuple[str, str], str] = {}
    failure_by_part: dict[tuple[str, str], str] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(extract_shard, Path(args.wds_root), shard, keys, image_dir): shard
            for shard, keys in sorted(wanted.items())
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            shard = futures[future]
            materialized, shard_failures = future.result()
            for key, path in materialized.items():
                by_part[(shard, key)] = path
            for key, reason in shard_failures.items():
                failure_by_part[(shard, key)] = reason
            if completed % 25 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "shards_done": completed,
                            "shards_total": len(futures),
                            "materialized": len(by_part),
                            "failures": len(failure_by_part),
                        }
                    ),
                    flush=True,
                )

    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        part = row_parts.get(index)
        if part is None:
            continue
        path = by_part.get(part)
        if path is None:
            failures.append(
                {
                    "row": index,
                    "request_id": row.get("request_id"),
                    "reason": failure_by_part.get(part, "extract_failed"),
                }
            )
            continue
        output_rows.append(update_row(row, path))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    failure_path = output.with_suffix(".failures.jsonl")
    with failure_path.open("w", encoding="utf-8") as handle:
        for item in failures:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "input": args.input,
        "output": str(output),
        "image_dir": str(image_dir),
        "wds_root": args.wds_root,
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "unique_images_requested": sum(len(keys) for keys in wanted.values()),
        "unique_images_materialized": len(by_part),
        "failed_rows": len(rows) - len(output_rows),
        "failures_jsonl": str(failure_path),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
