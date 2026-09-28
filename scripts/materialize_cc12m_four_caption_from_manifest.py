#!/usr/bin/env python3
"""Materialize CC12M four-caption rows from the canonical manifest.

The four-caption CC12M slice is keyed by ``image.dataset_key`` and URL rather
than by an already materialized local image path. This script resolves those
keys through the canonical CC12M manifest, extracts images from the local WDS
tar store, and rewrites all caption-surface JSONLs with the same local image
path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pyarrow.dataset as ds


SURFACES = [
    "ours_cc12m",
    "ref_cc12m_qwen3vl8b",
    "ref_cc12m_llavanext",
    "ref_pixelprose_cc12m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/local-images/cc12m-four-caption-url-key-5k",
    )
    parser.add_argument(
        "--output-dir",
        default="data/local-images/cc12m-four-caption-url-key-5k-local",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/cc12m-202603.manifest.parquet",
    )
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def storage_parts(storage_uri: str) -> tuple[Path, str, str]:
    if "#" not in storage_uri:
        raise ValueError(f"invalid storage_uri: {storage_uri}")
    tar_path, member = storage_uri.split("#", 1)
    shard = Path(tar_path).stem
    key = Path(member).stem
    return Path(tar_path), shard, member


def normalize_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    parts = urlsplit(text)
    if not parts.scheme or not parts.netloc:
        return text
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def row_identity(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    image = row.get("image") if isinstance(row.get("image"), dict) else {}
    return {
        "dataset_key": image.get("dataset_key") or metadata.get("dataset_key") or row.get("pair_key"),
        "canonical_url": metadata.get("canonical_url") or image.get("url"),
        "source_url_sha1": metadata.get("source_url_sha1"),
        "stable_image_id": metadata.get("stable_image_id") or image.get("stable_image_id"),
        "bytes": metadata.get("bytes") or image.get("bytes"),
        "width": metadata.get("width") or image.get("width"),
        "height": metadata.get("height") or image.get("height"),
    }


def load_manifest_rows(manifest: Path, keys: set[str]) -> dict[str, Any]:
    dataset = ds.dataset(str(manifest), format="parquet")
    table = dataset.to_table(
        columns=[
            "dataset_key",
            "storage_uri",
            "canonical_url",
            "source_url_sha1",
            "stable_image_id",
            "sample_id",
            "width",
            "height",
            "bytes",
            "sha256_raw",
        ],
        filter=ds.field("dataset_key").isin(sorted(keys)),
    )
    rows = table.to_pylist()
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row.get("dataset_key")
        uri = row.get("storage_uri")
        if not isinstance(key, str) or not isinstance(uri, str):
            continue
        by_key[key].append(row)
    duplicate_keys = {key for key, values in by_key.items() if len(values) > 1}
    return {
        "rows": dict(by_key),
        "duplicate_keys": {"count": len(duplicate_keys), "keys": sorted(duplicate_keys)[:20]},
    }


def choose_manifest_row(candidates: list[dict[str, Any]], desired: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if not candidates:
        return None, "missing_manifest"
    if len(candidates) == 1:
        candidate = candidates[0]
        stable_ok = not desired.get("stable_image_id") or candidate.get("stable_image_id") == desired.get("stable_image_id")
        url_ok = not desired.get("canonical_url") or normalize_url(candidate.get("canonical_url")) == normalize_url(desired.get("canonical_url"))
        if stable_ok and url_ok:
            return candidate, "unique"
        return None, "unique_identity_mismatch"

    expected_stable = desired.get("stable_image_id")
    expected_sha1 = desired.get("source_url_sha1")
    expected_url = normalize_url(desired.get("canonical_url"))
    expected_bytes = desired.get("bytes")
    expected_width = desired.get("width")
    expected_height = desired.get("height")

    def score(candidate: dict[str, Any]) -> int:
        value = 0
        if expected_stable and candidate.get("stable_image_id") == expected_stable:
            value += 16
        if expected_sha1 and candidate.get("source_url_sha1") == expected_sha1:
            value += 8
        if expected_url and normalize_url(candidate.get("canonical_url")) == expected_url:
            value += 4
        if expected_bytes is not None and candidate.get("bytes") == expected_bytes:
            value += 2
        if expected_width is not None and expected_height is not None:
            if candidate.get("width") == expected_width and candidate.get("height") == expected_height:
                value += 1
        return value

    ranked = sorted(candidates, key=score, reverse=True)
    best = ranked[0]
    best_score = score(best)
    tied = [candidate for candidate in ranked if score(candidate) == best_score]
    if best_score <= 0:
        return None, "duplicate_no_identity_match"
    if len(tied) > 1:
        return None, "duplicate_ambiguous_identity_match"
    return best, "duplicate_identity_match"


def extract_shard(tar_path: Path, members: set[str], image_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not tar_path.exists():
        return {member: f"missing_tar:{tar_path}" for member in members}
    image_dir.mkdir(parents=True, exist_ok=True)
    for member in sorted(members):
        suffix = Path(member).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            out[member] = f"unsupported_suffix:{member}"
            continue
        stem = Path(member).stem
        out_path = image_dir / f"{tar_path.stem}__{stem}{suffix}"
        if not out_path.exists():
            try:
                out_path.write_bytes(subprocess.check_output(["tar", "-xOf", str(tar_path), member]))
            except subprocess.CalledProcessError:
                out[member] = f"missing_member:{member}"
                continue
        out[member] = str(out_path)
    return out


def update_image(row: dict[str, Any], manifest_row: dict[str, Any], image_path: str) -> dict[str, Any]:
    out = dict(row)
    image = dict(out.get("image") if isinstance(out.get("image"), dict) else {})
    image.update(
        {
            "url": manifest_row.get("canonical_url") or image.get("url"),
            "local_abs_path": image_path,
            "materialization_source": "cc12m_202603_manifest_storage_uri",
            "sample_id": manifest_row.get("sample_id"),
            "stable_image_id": manifest_row.get("stable_image_id") or image.get("stable_image_id"),
            "dataset_key": manifest_row.get("dataset_key") or image.get("dataset_key"),
            "width": manifest_row.get("width") or image.get("width"),
            "height": manifest_row.get("height") or image.get("height"),
            "bytes": manifest_row.get("bytes") or image.get("bytes"),
            "sha256_raw": manifest_row.get("sha256_raw"),
            "storage_uri": manifest_row.get("storage_uri"),
        }
    )
    out["image"] = image
    return out


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_surface = {surface: read_jsonl(input_dir / f"{surface}.jsonl") for surface in SURFACES}
    row_counts = {surface: len(rows) for surface, rows in rows_by_surface.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"surface row count mismatch: {row_counts}")

    keys: list[str] = []
    for row in rows_by_surface["ours_cc12m"]:
        image = row.get("image") if isinstance(row.get("image"), dict) else {}
        key = image.get("dataset_key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"missing image.dataset_key at source_row={row.get('source_row')}")
        keys.append(key)

    manifest_payload = load_manifest_rows(Path(args.manifest), set(keys))
    manifest_candidates: dict[str, list[dict[str, Any]]] = manifest_payload["rows"]
    duplicate_info = manifest_payload["duplicate_keys"]
    selected_manifest_rows: dict[int, dict[str, Any]] = {}
    selection_reasons: defaultdict[str, int] = defaultdict(int)
    selection_failures: list[dict[str, Any]] = []
    for index, key in enumerate(keys):
        desired = row_identity(rows_by_surface["ours_cc12m"][index])
        selected, reason = choose_manifest_row(manifest_candidates.get(key, []), desired)
        selection_reasons[reason] += 1
        if selected is None:
            selection_failures.append(
                {
                    "row": index,
                    "dataset_key": key,
                    "reason": reason,
                    "desired": desired,
                    "candidate_count": len(manifest_candidates.get(key, [])),
                }
            )
            continue
        selected_manifest_rows[index] = selected

    members_by_tar: dict[Path, set[str]] = defaultdict(set)
    index_to_member: dict[int, tuple[Path, str]] = {}
    for index, row in selected_manifest_rows.items():
        tar_path, _shard, member = storage_parts(str(row["storage_uri"]))
        members_by_tar[tar_path].add(member)
        index_to_member[index] = (tar_path, member)

    extracted_by_member: dict[tuple[Path, str], str] = {}
    failures: list[dict[str, Any]] = list(selection_failures)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(extract_shard, tar_path, members, image_dir): tar_path
            for tar_path, members in sorted(members_by_tar.items(), key=lambda item: str(item[0]))
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            tar_path = futures[future]
            for member, result in future.result().items():
                if result.startswith(("missing_", "unsupported_")):
                    failures.append({"tar_path": str(tar_path), "member": member, "reason": result})
                else:
                    extracted_by_member[(tar_path, member)] = result
            if completed % 50 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "shards_done": completed,
                            "shards_total": len(futures),
                            "extracted": len(extracted_by_member),
                            "failures": len(failures),
                        }
                    ),
                    flush=True,
                )

    kept_indexes: list[int] = []
    for index, key in enumerate(keys):
        part = index_to_member.get(index)
        if part is None:
            continue
        if part not in extracted_by_member:
            failures.append({"row": index, "dataset_key": key, "reason": "extract_failed"})
            continue
        kept_indexes.append(index)

    for surface, rows in rows_by_surface.items():
        out_rows = []
        for index in kept_indexes:
            tar_path, member = index_to_member[index]
            out_rows.append(update_image(rows[index], selected_manifest_rows[index], extracted_by_member[(tar_path, member)]))
        write_jsonl(output_dir / f"{surface}.jsonl", out_rows)

    if (input_dir / "multi_caption_rows.jsonl").exists():
        multi_rows = read_jsonl(input_dir / "multi_caption_rows.jsonl")
        out_multi = []
        for index in kept_indexes:
            tar_path, member = index_to_member[index]
            out_multi.append(update_image(multi_rows[index], selected_manifest_rows[index], extracted_by_member[(tar_path, member)]))
        write_jsonl(output_dir / "multi_caption_rows.jsonl", out_multi)

    fail_path = output_dir / "failures.jsonl"
    write_jsonl(fail_path, failures)
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "manifest": args.manifest,
        "surfaces": SURFACES,
        "input_rows": row_counts,
        "requested_images": len(keys),
        "manifest_candidate_keys": len(manifest_candidates),
        "manifest_matches": len(selected_manifest_rows),
        "duplicate_manifest_keys_ignored": duplicate_info,
        "selection_reasons": dict(sorted(selection_reasons.items())),
        "materialized_images": len(extracted_by_member),
        "written_rows_per_surface": len(kept_indexes),
        "failed_rows": len(keys) - len(kept_indexes),
        "failures_jsonl": str(fail_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
