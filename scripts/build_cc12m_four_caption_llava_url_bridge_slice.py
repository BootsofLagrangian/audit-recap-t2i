#!/usr/bin/env python3
"""Build a CC12M four-caption slice using LLaVA URL as the image bridge.

Old CC12M recap surfaces do not share a portable numeric key space with the
current ``cc12m-202603`` WDS manifest. Ours and Qwen3-VL expose only old
numeric keys, while LLaVA-NeXT exposes old numeric keys plus URLs and
PixelProse exposes URLs. This builder therefore:

1. joins ours, Qwen3-VL, and LLaVA-NeXT by the old numeric key;
2. joins PixelProse by normalized LLaVA URL;
3. resolves the local image asset through the canonical manifest by URL.

The resulting rows are suitable for image-grounded metrics only if the paired
materializer subsequently verifies metadata/image identity.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlsplit, urlunsplit

import pyarrow.dataset as ds
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-jsonl", action="append", default=[])
    parser.add_argument(
        "--ours-parquet",
        default=None,
        help="Parquet file/dir/glob with image_id and caption fields, preferred over raw JSONL.",
    )
    parser.add_argument("--qwen-jsonl", required=True)
    parser.add_argument("--llavanext-jsonl", required=True)
    parser.add_argument("--pixelprose-jsonl", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@dataclass(frozen=True)
class CaptionRow:
    key: str
    caption: str
    url: str | None = None
    sha256: str | None = None
    raw: dict[str, Any] | None = None


def normalize_key(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text.zfill(9) if text else None


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


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    import glob

    out: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            out.extend(Path(item) for item in glob.glob(pattern))
        else:
            path = Path(pattern)
            if path.exists():
                out.append(path)
    return sorted({path.resolve() for path in out})


def load_by_key(paths: Iterable[str], key_field: str, caption_fields: tuple[str, ...]) -> tuple[dict[str, CaptionRow], dict[str, int]]:
    rows: dict[str, CaptionRow] = {}
    stats = {"seen": 0, "loaded": 0, "missing_key": 0, "missing_caption": 0, "duplicate_key": 0}
    for path in expand_paths(paths):
        for row in iter_jsonl(path):
            stats["seen"] += 1
            key = normalize_key(row.get(key_field))
            if key is None:
                stats["missing_key"] += 1
                continue
            caption = next((row.get(field) for field in caption_fields if isinstance(row.get(field), str) and row.get(field).strip()), None)
            if not isinstance(caption, str):
                stats["missing_caption"] += 1
                continue
            if key in rows:
                stats["duplicate_key"] += 1
                continue
            rows[key] = CaptionRow(
                key=key,
                caption=caption.strip(),
                url=normalize_url(row.get("url")),
                sha256=row.get("sha256") if isinstance(row.get("sha256"), str) else None,
                raw=row,
            )
    stats["loaded"] = len(rows)
    return rows, stats


def load_ours_parquet(pattern: str) -> tuple[dict[str, CaptionRow], dict[str, int]]:
    import glob

    paths = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[]") else [pattern]
    expanded: list[Path] = []
    for path in paths:
        path_obj = Path(path)
        if path_obj.is_dir():
            expanded.extend(sorted(path_obj.glob("*.parquet")))
        elif path_obj.exists():
            expanded.append(path_obj)

    rows: dict[str, CaptionRow] = {}
    stats = {"seen": 0, "loaded": 0, "missing_key": 0, "missing_caption": 0, "duplicate_key": 0}
    wanted = [
        "image_id",
        "caption",
        "caption_text",
        "source_url",
        "source_sha256",
        "stable_image_id",
        "width",
        "height",
    ]
    for path in expanded:
        parquet = pq.ParquetFile(path)
        columns = [column for column in wanted if column in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(columns=columns, batch_size=65536):
            for row in batch.to_pylist():
                stats["seen"] += 1
                key = normalize_key(row.get("image_id"))
                if key is None:
                    stats["missing_key"] += 1
                    continue
                caption = row.get("caption") or row.get("caption_text")
                if not isinstance(caption, str) or not caption.strip():
                    stats["missing_caption"] += 1
                    continue
                if key in rows:
                    stats["duplicate_key"] += 1
                    continue
                rows[key] = CaptionRow(
                    key=key,
                    caption=caption.strip(),
                    url=normalize_url(row.get("source_url")),
                    sha256=row.get("source_sha256") if isinstance(row.get("source_sha256"), str) else None,
                    raw=row,
                )
    stats["loaded"] = len(rows)
    return rows, stats


def load_pixel_by_url(path: str) -> tuple[dict[str, CaptionRow], dict[str, int]]:
    rows: dict[str, CaptionRow] = {}
    stats = {"seen": 0, "loaded": 0, "missing_url": 0, "missing_caption": 0, "duplicate_url": 0}
    for row in iter_jsonl(Path(path)):
        stats["seen"] += 1
        url = normalize_url(row.get("url"))
        if url is None:
            stats["missing_url"] += 1
            continue
        caption = row.get("vlm_caption") or row.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            stats["missing_caption"] += 1
            continue
        if url in rows:
            stats["duplicate_url"] += 1
            continue
        rows[url] = CaptionRow(
            key=normalize_key(row.get("key")) or "",
            caption=caption.strip(),
            url=url,
            sha256=row.get("sha256") if isinstance(row.get("sha256"), str) else None,
            raw=row,
        )
    stats["loaded"] = len(rows)
    return rows, stats


def load_manifest_by_url(path: str, urls: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    dataset = ds.dataset(path, format="parquet")
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
        filter=ds.field("canonical_url").isin(sorted(urls)),
    )
    by_url: dict[str, dict[str, Any]] = {}
    duplicate_urls = 0
    for row in table.to_pylist():
        url = normalize_url(row.get("canonical_url"))
        if url is None:
            continue
        if url in by_url:
            duplicate_urls += 1
            old = by_url[url]
            old_id = str(old.get("stable_image_id"))
            new_id = str(row.get("stable_image_id"))
            if old_id != new_id:
                continue
        by_url[url] = row
    return by_url, {"rows": table.num_rows, "loaded_urls": len(by_url), "duplicate_urls": duplicate_urls}


def surface_row(
    *,
    row_idx: int,
    surface: str,
    caption: str,
    metadata: dict[str, Any],
    source_keys: dict[str, Any],
) -> dict[str, Any]:
    image = {
        "url": metadata.get("canonical_url"),
        "stable_image_id": metadata.get("stable_image_id"),
        "dataset_key": metadata.get("dataset_key"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
    }
    return {
        "comparison": "cc12m_four_caption_llava_url_bridge",
        "family": "cc12m",
        "tier": "llava_url_bridge_four_caption_intersection",
        "pair_key": source_keys["old_key"],
        "public_lookup_key": source_keys.get("llavanext_url"),
        "source_row": row_idx,
        "surface": surface,
        "caption": caption,
        "metadata": metadata,
        "image": image,
        "source_keys": source_keys,
    }


def main() -> int:
    args = parse_args()
    if args.ours_parquet:
        ours, ours_stats = load_ours_parquet(args.ours_parquet)
    else:
        if not args.ours_jsonl:
            raise ValueError("provide --ours-parquet or --ours-jsonl")
        ours, ours_stats = load_by_key(args.ours_jsonl, "image_id", ("caption",))
    qwen, qwen_stats = load_by_key([args.qwen_jsonl], "key", ("caption",))
    llava, llava_stats = load_by_key([args.llavanext_jsonl], "key", ("caption_llava", "caption"))
    pixel_by_url, pixel_stats = load_pixel_by_url(args.pixelprose_jsonl)

    common_keys = [key for key in sorted(set(ours) & set(qwen) & set(llava)) if llava[key].url in pixel_by_url]
    rng = random.Random(args.seed)
    rng.shuffle(common_keys)
    if args.max_rows:
        common_keys = common_keys[: args.max_rows]

    urls = {llava[key].url for key in common_keys if llava[key].url is not None}
    manifest_by_url, manifest_stats = load_manifest_by_url(args.manifest, urls)
    selected_keys = [key for key in common_keys if llava[key].url in manifest_by_url]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    surfaces = {
        "ours_cc12m": out_dir / "ours_cc12m.jsonl",
        "ref_cc12m_qwen3vl8b": out_dir / "ref_cc12m_qwen3vl8b.jsonl",
        "ref_cc12m_llavanext": out_dir / "ref_cc12m_llavanext.jsonl",
        "ref_pixelprose_cc12m": out_dir / "ref_pixelprose_cc12m.jsonl",
    }
    handles = {name: path.open("w", encoding="utf-8") for name, path in surfaces.items()}
    bundle_path = out_dir / "multi_caption_rows.jsonl"
    with bundle_path.open("w", encoding="utf-8") as bundle:
        for row_idx, key in enumerate(selected_keys):
            llava_url = llava[key].url
            assert llava_url is not None
            pixel = pixel_by_url[llava_url]
            metadata = manifest_by_url[llava_url]
            source_keys = {
                "old_key": key,
                "ours_image_id": ours[key].key,
                "qwen_key": qwen[key].key,
                "llavanext_key": llava[key].key,
                "llavanext_url": llava_url,
                "pixelprose_key": pixel.key,
                "pixelprose_url": pixel.url,
                "pixelprose_sha256": pixel.sha256,
                "manifest_dataset_key": metadata.get("dataset_key"),
            }
            captions = {
                "ours_cc12m": ours[key].caption,
                "ref_cc12m_qwen3vl8b": qwen[key].caption,
                "ref_cc12m_llavanext": llava[key].caption,
                "ref_pixelprose_cc12m": pixel.caption,
            }
            bundle.write(
                json.dumps(
                    {
                        "row": row_idx,
                        "comparison": "cc12m_four_caption_llava_url_bridge",
                        "metadata": metadata,
                        "image": {
                            "url": metadata.get("canonical_url"),
                            "stable_image_id": metadata.get("stable_image_id"),
                            "dataset_key": metadata.get("dataset_key"),
                            "width": metadata.get("width"),
                            "height": metadata.get("height"),
                        },
                        "source_keys": source_keys,
                        "captions": captions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for surface, caption in captions.items():
                handles[surface].write(
                    json.dumps(
                        surface_row(
                            row_idx=row_idx,
                            surface=surface,
                            caption=caption,
                            metadata=metadata,
                            source_keys=source_keys,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    for handle in handles.values():
        handle.close()

    manifest = {
        "comparison": "cc12m_four_caption_llava_url_bridge",
        "output_dir": str(out_dir),
        "rows": len(selected_keys),
        "max_rows": args.max_rows,
        "seed": args.seed,
        "join_rule": "ours/qwen/llavanext by old numeric key; pixelprose and canonical image by normalized llavanext URL",
        "input_stats": {
            "ours": ours_stats,
            "qwen3vl8b": qwen_stats,
            "llavanext": llava_stats,
            "pixelprose": pixel_stats,
            "manifest": manifest_stats,
        },
        "candidate_counts": {
            "old_key_common_ours_qwen_llava_and_pixel_url": len(common_keys),
            "manifest_url_matched": len(selected_keys),
        },
        "outputs": {
            "bundle_jsonl": str(bundle_path),
            "surfaces": {name: str(path) for name, path in surfaces.items()},
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
