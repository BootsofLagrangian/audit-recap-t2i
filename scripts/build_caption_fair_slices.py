#!/usr/bin/env python3
"""Build paired caption slices for fair recap-corpus comparison.

The script treats the local recap surface as the image gate. Reference captions
are retained only when they can be matched to the same image key/URL. If a
canonical manifest is available, width/height and stable provenance fields are
attached to every retained pair.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlsplit, urlunsplit

import pyarrow.parquet as pq

try:
    import orjson
except ImportError:  # pragma: no cover - optional speed path
    orjson = None


DEFAULT_METADATA_FIELDS = [
    "sample_id",
    "dataset_key",
    "canonical_url",
    "source_url_sha1",
    "stable_image_id",
    "width",
    "height",
    "bytes",
    "license",
]


@dataclass
class Candidate:
    local_key: str
    local_caption: str
    local_record: dict[str, Any]
    metadata: dict[str, Any]
    public_lookup_key: str | None = None
    public_caption: str | None = None
    public_record: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired fair-slice JSONL files for caption survey")
    parser.add_argument("--config", default="configs/caption_survey/fair_slices.json")
    parser.add_argument("--comparison", action="append", default=[], help="Comparison name to build")
    parser.add_argument("--all", action="store_true", help="Build all configured comparisons")
    parser.add_argument("--max-pairs", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default="data/caption-fair-slices")
    parser.add_argument("--local-caption-field", default="caption")
    parser.add_argument("--public-caption-field", default="caption")
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--public-max-rows", type=int, default=None, help="Debug only: stop public scan after N rows")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_loads(raw: bytes | str) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(raw)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def json_dumps(row: dict[str, Any]) -> str:
    if orjson is not None:
        return orjson.dumps(row).decode("utf-8")
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def normalize_url(value: str) -> str:
    stripped = value.strip()
    parts = urlsplit(stripped)
    if not parts.scheme or not parts.netloc:
        return stripped
    netloc = parts.netloc.lower()
    scheme = parts.scheme.lower()
    return urlunsplit((scheme, netloc, parts.path, parts.query, ""))


def normalize_key(value: Any, mode: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if mode == "str":
        return text
    if mode == "zfill9":
        return text.zfill(9)
    if mode == "zfill12":
        return text.zfill(12)
    if mode == "url":
        return normalize_url(text)
    raise ValueError(f"Unsupported key normalization mode: {mode}")


def expand_inputs(patterns: Iterable[str], seed: int) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            paths.extend(Path(path) for path in glob.glob(pattern))
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    unique = sorted({path.resolve() for path in paths})
    random.Random(seed).shuffle(unique)
    return unique


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                yield json_loads(line)


def short_edge_ok(metadata: dict[str, Any], min_short_edge: int | None) -> bool:
    if min_short_edge is None:
        return True
    width = metadata.get("width")
    height = metadata.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return False
    return min(width, height) >= min_short_edge


def read_local_candidates(spec: dict[str, Any], args: argparse.Namespace) -> tuple[list[Candidate], dict[str, Any]]:
    key_mode = spec.get("key_normalization", "str")
    key_field = spec["local_key_field"]
    caption_field = spec.get("local_caption_field", args.local_caption_field)
    candidate_target = int(spec.get("local_candidate_limit") or args.max_pairs * spec.get("local_candidate_multiplier", 1))
    files = expand_inputs(spec["local_inputs"], args.seed)
    candidates: dict[str, Candidate] = {}
    rows_seen = 0
    rows_missing_key = 0
    rows_missing_caption = 0
    duplicate_keys = 0
    started = time.time()

    for path in files:
        for row in iter_jsonl(path):
            rows_seen += 1
            key = normalize_key(row.get(key_field), key_mode)
            caption = row.get(caption_field)
            if key is None:
                rows_missing_key += 1
                continue
            if not isinstance(caption, str) or not caption.strip():
                rows_missing_caption += 1
                continue
            if key in candidates:
                duplicate_keys += 1
                continue
            candidates[key] = Candidate(
                local_key=key,
                local_caption=caption.strip(),
                local_record={
                    "image_id": row.get("image_id"),
                    "source": row.get("source"),
                    "image_path": row.get("image_path"),
                    "image_abs_path": absolute_image_path(spec, row.get("image_path")),
                    "url": row.get("url"),
                    "model": row.get("model"),
                    "completion_tokens": row.get("completion_tokens"),
                },
                metadata={},
            )
            if len(candidates) >= candidate_target:
                break
        if len(candidates) >= candidate_target:
            break

    return list(candidates.values()), {
        "files_considered": len(files),
        "rows_seen": rows_seen,
        "candidate_target": candidate_target,
        "candidate_count": len(candidates),
        "rows_missing_key": rows_missing_key,
        "rows_missing_caption": rows_missing_caption,
        "duplicate_keys": duplicate_keys,
        "seconds": round(time.time() - started, 2),
    }


def absolute_image_path(spec: dict[str, Any], rel_path: Any) -> str | None:
    root = spec.get("local_image_root")
    if not root or not isinstance(rel_path, str) or not rel_path:
        return None
    return str(Path(root) / rel_path)


def attach_manifest_metadata(
    spec: dict[str, Any],
    candidates: list[Candidate],
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifest_path = spec.get("manifest_parquet")
    public_lookup_source = spec.get("public_lookup_source", "local_key")
    if not manifest_path:
        for candidate in candidates:
            candidate.public_lookup_key = candidate.local_key
        return {
            "manifest_used": False,
            "candidates_after_manifest": len(candidates),
        }

    manifest_key_field = spec["manifest_key_field"]
    local_key_mode = spec.get("key_normalization", "str")
    public_key_mode = spec.get("public_key_normalization", spec.get("key_normalization", "str"))
    public_manifest_field = spec.get("manifest_public_key_field")
    min_short_edge = spec.get("min_short_edge")
    selected = {candidate.local_key for candidate in candidates}
    by_key = {candidate.local_key: candidate for candidate in candidates}
    metadata_fields = [field for field in DEFAULT_METADATA_FIELDS if field != manifest_key_field]
    if public_manifest_field and public_manifest_field not in metadata_fields:
        metadata_fields.append(public_manifest_field)
    columns = [manifest_key_field, *metadata_fields]
    found = 0
    gate_rejected = 0
    started = time.time()

    parquet = pq.ParquetFile(manifest_path)
    schema_names = set(parquet.schema_arrow.names)
    columns = [column for column in columns if column in schema_names]
    for batch in parquet.iter_batches(columns=columns, batch_size=args.batch_size):
        for row in batch.to_pylist():
            key = normalize_key(row.get(manifest_key_field), local_key_mode)
            if key not in selected:
                continue
            candidate = by_key[key]
            metadata = {field: row.get(field) for field in metadata_fields if field in row}
            if not short_edge_ok(metadata, min_short_edge):
                gate_rejected += 1
                selected.remove(key)
                continue
            candidate.metadata = metadata
            if public_lookup_source == "manifest_field":
                if not public_manifest_field:
                    raise ValueError(f"{spec['name']} requires manifest_public_key_field")
                candidate.public_lookup_key = normalize_key(row.get(public_manifest_field), public_key_mode)
            else:
                candidate.public_lookup_key = candidate.local_key
            found += 1
            selected.remove(key)
        if not selected:
            break

    kept = [candidate for candidate in candidates if candidate.public_lookup_key is not None]
    candidates[:] = kept
    return {
        "manifest_used": True,
        "manifest_path": manifest_path,
        "manifest_rows": parquet.metadata.num_rows,
        "manifest_key_field": manifest_key_field,
        "public_lookup_source": public_lookup_source,
        "manifest_public_key_field": public_manifest_field,
        "min_short_edge": min_short_edge,
        "manifest_matches": found,
        "manifest_missing": len(by_key) - found - gate_rejected,
        "manifest_gate_rejected": gate_rejected,
        "candidates_after_manifest": len(candidates),
        "seconds": round(time.time() - started, 2),
    }


def attach_public_captions(spec: dict[str, Any], candidates: list[Candidate], args: argparse.Namespace) -> dict[str, Any]:
    public_path = Path(spec["public_input"])
    scan_path = public_path
    public_key_mode = spec.get("public_key_normalization", spec.get("key_normalization", "str"))
    public_key_field = spec["public_key_field"]
    public_caption_field = spec.get("public_caption_field", args.public_caption_field)
    remaining = {candidate.public_lookup_key for candidate in candidates if candidate.public_lookup_key is not None}
    by_public_key = {candidate.public_lookup_key: candidate for candidate in candidates if candidate.public_lookup_key is not None}
    rows_seen = 0
    rows_missing_key = 0
    rows_missing_caption = 0
    duplicate_public_matches = 0
    started = time.time()

    for row in iter_jsonl(scan_path):
        rows_seen += 1
        if args.public_max_rows is not None and rows_seen > args.public_max_rows:
            break
        key = normalize_key(row.get(public_key_field), public_key_mode)
        if key is None:
            rows_missing_key += 1
            continue
        if key not in remaining:
            continue
        caption = row.get(public_caption_field)
        if not isinstance(caption, str) or not caption.strip():
            rows_missing_caption += 1
            continue
        candidate = by_public_key[key]
        if candidate.public_caption is not None:
            duplicate_public_matches += 1
            continue
        candidate.public_caption = caption.strip()
        candidate.public_record = {
            public_key_field: row.get(public_key_field),
            "caption_field": row.get("caption_field"),
            "url": row.get("url"),
            "key": row.get("key"),
            "uid": row.get("uid"),
            "sha256": row.get("sha256"),
            "source_id": row.get("source_id"),
            "status": row.get("status"),
            "vlm_model": row.get("vlm_model"),
        }
        remaining.remove(key)
        if not remaining:
            break

    kept = [candidate for candidate in candidates if candidate.public_caption is not None]
    candidates[:] = kept
    return {
        "public_input": str(public_path),
        "public_input_bytes": public_path.stat().st_size if public_path.exists() else None,
        "public_scan_input": str(scan_path),
        "public_scan_input_bytes": scan_path.stat().st_size if scan_path.exists() else None,
        "public_key_field": public_key_field,
        "public_caption_field": public_caption_field,
        "public_max_rows": args.public_max_rows,
        "rows_seen": rows_seen,
        "rows_missing_key": rows_missing_key,
        "rows_missing_caption": rows_missing_caption,
        "duplicate_public_matches": duplicate_public_matches,
        "public_matches": len(candidates),
        "public_missing_after_scan": len(remaining),
        "seconds": round(time.time() - started, 2),
    }


def stable_pair_id(spec: dict[str, Any], candidate: Candidate) -> str:
    raw = "\0".join(
        [
            str(spec.get("name")),
            str(candidate.local_key),
            str(candidate.public_lookup_key),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def image_metadata_for_candidate(candidate: Candidate) -> dict[str, Any]:
    metadata = candidate.metadata
    public_record = candidate.public_record or {}
    url = metadata.get("canonical_url") or candidate.local_record.get("url") or public_record.get("url")
    normalized_url = normalize_url(url) if isinstance(url, str) and url else None
    local_abs_path = candidate.local_record.get("image_abs_path")
    local_file_exists = False
    local_file_bytes = None
    if isinstance(local_abs_path, str) and local_abs_path:
        try:
            stat_result = Path(local_abs_path).stat()
            local_file_exists = True
            local_file_bytes = stat_result.st_size
        except FileNotFoundError:
            pass
    return {
        "url": url,
        "normalized_url": normalized_url,
        "local_image_id": candidate.local_record.get("image_id"),
        "local_rel_path": candidate.local_record.get("image_path"),
        "local_abs_path": local_abs_path,
        "local_file_exists": local_file_exists,
        "local_file_bytes": local_file_bytes,
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "bytes": metadata.get("bytes"),
        "sha256": metadata.get("sha256_raw") or public_record.get("sha256"),
        "stable_image_id": metadata.get("stable_image_id"),
        "sample_id": metadata.get("sample_id"),
        "dataset_key": metadata.get("dataset_key"),
    }


def common_metadata(spec: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
    return {
        "pair_id": stable_pair_id(spec, candidate),
        "comparison": spec["name"],
        "pair_key": candidate.local_key,
        "public_lookup_key": candidate.public_lookup_key,
        "family": spec.get("family"),
        "tier": spec.get("tier"),
        "join": {
            "local_key_field": spec.get("local_key_field"),
            "public_key_field": spec.get("public_key_field"),
            "key_normalization": spec.get("key_normalization", "str"),
            "public_key_normalization": spec.get("public_key_normalization", spec.get("key_normalization", "str")),
            "public_lookup_source": spec.get("public_lookup_source", "local_key"),
        },
        "image": image_metadata_for_candidate(candidate),
        "metadata": candidate.metadata,
    }


def write_outputs(spec: dict[str, Any], candidates: list[Candidate], args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_root) / spec["name"] / f"n{args.max_pairs}_seed{args.seed}"
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {out_dir}. Pass --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / f"{spec['local_surface']}.jsonl"
    ref_path = out_dir / f"{spec['reference_surface']}.jsonl"
    pair_path = out_dir / "pairs.jsonl"

    with local_path.open("w", encoding="utf-8") as local_handle, ref_path.open("w", encoding="utf-8") as ref_handle, pair_path.open("w", encoding="utf-8") as pair_handle:
        for candidate in candidates:
            common = common_metadata(spec, candidate)
            local_row = {
                **common,
                "surface": spec["local_surface"],
                "caption": candidate.local_caption,
                "local_record": candidate.local_record,
            }
            ref_row = {
                **common,
                "surface": spec["reference_surface"],
                "caption": candidate.public_caption,
                "public_record": candidate.public_record,
            }
            pair_row = {
                **common,
                "local_surface": spec["local_surface"],
                "reference_surface": spec["reference_surface"],
                "local_caption": candidate.local_caption,
                "reference_caption": candidate.public_caption,
            }
            local_handle.write(json_dumps(local_row) + "\n")
            ref_handle.write(json_dumps(ref_row) + "\n")
            pair_handle.write(json_dumps(pair_row) + "\n")

    return {
        "output_dir": str(out_dir),
        "local_jsonl": str(local_path),
        "reference_jsonl": str(ref_path),
        "pairs_jsonl": str(pair_path),
        "paired_count": len(candidates),
        "local_jsonl_bytes": local_path.stat().st_size,
        "reference_jsonl_bytes": ref_path.stat().st_size,
        "pairs_jsonl_bytes": pair_path.stat().st_size,
    }


def build_one(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    candidates, local_summary = read_local_candidates(spec, args)
    manifest_summary = attach_manifest_metadata(spec, candidates, args)
    public_summary = attach_public_captions(spec, candidates, args)
    matched_before_cap = len(candidates)
    if len(candidates) > args.max_pairs:
        rng = random.Random(args.seed)
        rng.shuffle(candidates)
        del candidates[args.max_pairs :]
    output_summary = write_outputs(spec, candidates, args)
    return {
        "comparison": spec["name"],
        "family": spec.get("family"),
        "tier": spec.get("tier"),
        "local_surface": spec["local_surface"],
        "reference_surface": spec["reference_surface"],
        "max_pairs_requested": args.max_pairs,
        "seed": args.seed,
        "local": local_summary,
        "manifest": manifest_summary,
        "public": public_summary,
        "matched_before_cap": matched_before_cap,
        "matched_cap": args.max_pairs,
        "outputs": output_summary,
        "seconds_total": round(time.time() - started, 2),
    }


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    comparisons = config.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError(f"Invalid fair-slice config: {args.config}")
    selected_names = set(args.comparison)
    selected = []
    for spec in comparisons:
        if not isinstance(spec, dict):
            continue
        if args.all or spec.get("name") in selected_names:
            selected.append(spec)
    if not selected:
        raise SystemExit("No comparisons selected")

    summaries = []
    for spec in selected:
        summary = build_one(spec, args)
        summaries.append(summary)
        summary_path = Path(summary["outputs"]["output_dir"]) / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
