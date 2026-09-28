#!/usr/bin/env python3
"""Extract official DiffusionDB prompt-only pools without image payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
UNSAFE_RE = re.compile(
    r"\b("
    r"nsfw|nude|nudity|naked|porn|pornographic|erotic|sex|sexual|sexy|fetish|"
    r"child\s*porn|loli|shota|underage|minor|rape|incest|"
    r"gore|gory|bloodbath|beheading|decapitated|dismembered"
    r")\b",
    re.IGNORECASE,
)

DIFFUSIONDB_METADATA_URL = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet"
DIFFUSIONDB_COMMIT = "fb620fbe49fa4420e0734bd9c0df11f51176b61f"
DIFFUSIONDB_LINKED_SIZE = 194_548_652
DIFFUSIONDB_LICENSE = "cc0-1.0 per Hugging Face dataset card metadata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract prompt-only JSONL from official DiffusionDB metadata parquet")
    parser.add_argument("--cache-root", default="data/prompt-pools/diffusiondb-official")
    parser.add_argument("--output-root", default="data/prompt-pools/diffusiondb-official-2026-04-24")
    parser.add_argument("--metadata-url", default=DIFFUSIONDB_METADATA_URL)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--min-tokens", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def normalize_prompt(text: str) -> str:
    return SPACE_RE.sub(" ", text.replace("\u0000", " ").strip())


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def stable_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def is_safe_prompt(text: str) -> bool:
    if URL_RE.search(text) or EMAIL_RE.search(text) or PHONE_RE.search(text):
        return False
    return UNSAFE_RE.search(text) is None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_metadata(url: str, path: Path, force: bool) -> None:
    if path.exists() and path.stat().st_size > 0 and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5", "-o", str(tmp), url],
        check=True,
    )
    tmp.replace(path)


def iter_prompts(parquet_path: Path, field: str, batch_size: int, max_records: int | None) -> Iterator[str]:
    parquet_file = pq.ParquetFile(parquet_path)
    seen = 0
    schema_names = set(parquet_file.schema_arrow.names)
    if field not in schema_names:
        raise SystemExit(f"prompt field {field!r} not found; schema fields={sorted(schema_names)}")
    for batch in parquet_file.iter_batches(columns=[field], batch_size=batch_size):
        for row in batch.to_pylist():
            if max_records is not None and seen >= max_records:
                return
            seen += 1
            yield row.get(field, "")


def main() -> int:
    args = parse_args()
    cache_root = Path(args.cache_root)
    output_root = Path(args.output_root)
    metadata_path = cache_root / "metadata.parquet"
    output_root.mkdir(parents=True, exist_ok=True)
    download_metadata(args.metadata_url, metadata_path, args.force_download)

    prompt_out = output_root / "diffusiondb_official.prompt_only.jsonl"
    filtered_out = output_root / "diffusiondb_official.filtered.jsonl"
    seen_hashes: set[str] = set()
    stats: Counter[str] = Counter()
    token_hist: Counter[int] = Counter()

    with prompt_out.open("w", encoding="utf-8") as prompt_handle, filtered_out.open("w", encoding="utf-8") as filtered_handle:
        for raw_index, raw_prompt in enumerate(iter_prompts(metadata_path, args.prompt_field, args.batch_size, args.max_records)):
            stats["read"] += 1
            prompt = normalize_prompt(str(raw_prompt))
            if not prompt:
                stats["empty"] += 1
                continue
            prompt_hash = stable_hash(prompt.lower())
            if prompt_hash in seen_hashes:
                stats["duplicate"] += 1
                continue
            seen_hashes.add(prompt_hash)
            n_tokens = token_count(prompt)
            token_hist[min(n_tokens, 512)] += 1
            if n_tokens < args.min_tokens:
                stats["too_short"] += 1
                continue
            if n_tokens > args.max_tokens:
                stats["too_long"] += 1
                continue
            row = {
                "source": "diffusiondb_official",
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "tokens_lexical": n_tokens,
                "raw_index": raw_index,
            }
            prompt_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["prompt_only_written"] += 1
            if is_safe_prompt(prompt):
                filtered_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["filtered_written"] += 1
            else:
                stats["filtered_safety"] += 1

    manifest = {
        "source": "diffusiondb_official",
        "metadata_url": args.metadata_url,
        "hf_dataset": "poloclub/diffusiondb",
        "hf_commit_observed": DIFFUSIONDB_COMMIT,
        "license_hint": DIFFUSIONDB_LICENSE,
        "linked_size_observed": DIFFUSIONDB_LINKED_SIZE,
        "metadata_path": str(metadata_path),
        "metadata_size": metadata_path.stat().st_size,
        "metadata_sha256": sha256_file(metadata_path),
        "prompt_field": args.prompt_field,
        "prompt_only_jsonl": str(prompt_out),
        "filtered_jsonl": str(filtered_out),
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "stats": dict(stats),
        "token_hist_capped_512": dict(sorted(token_hist.items())),
        "citation_hint": "DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models, ACL 2023.",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
