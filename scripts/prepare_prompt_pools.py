#!/usr/bin/env python3
"""Build prompt-only reference pools for recap prompt-support metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")

# Conservative prompt-pool safety filter. The unfiltered prompt-only split is
# retained separately for reproducibility; paper-facing support metrics should
# default to the filtered split.
UNSAFE_RE = re.compile(
    r"\b("
    r"nsfw|nude|nudity|naked|porn|pornographic|erotic|sex|sexual|sexy|fetish|"
    r"child\s*porn|loli|shota|underage|minor|rape|incest|"
    r"gore|gory|bloodbath|beheading|decapitated|dismembered"
    r")\b",
    re.IGNORECASE,
)


DEFAULT_SOURCES = {
    "sd_prompts_2m_andyyang": {
        "path": "data/prompt-pools/hf-raw/sd_promts_2m.txt",
        "format": "txt",
        "citation_hint": "Stable-Diffusion prompt mirror; cite only as operational mirror, not canonical DiffusionDB.",
        "license_hint": "HF card reports cc0-1.0 for andyyang/stable_diffusion_prompts_2m.",
    },
    "sd_prompts_dedup_xzuyn": {
        "path": "data/prompt-pools/hf-raw/v3_no_bos.txt",
        "format": "txt",
        "citation_hint": "Stable-Diffusion prompt mirror; use as appendix/source-sensitivity pool.",
        "license_hint": "HF card did not expose a license during local inspection; verify before paper release.",
    },
    "pickapic_rankings": {
        "path": "data/prompt-pools/hf-raw/data/train-00000-of-00001-5088566feaadd3d9.parquet",
        "format": "parquet",
        "field": "prompt",
        "citation_hint": "Pick-a-Pic, NeurIPS 2023.",
        "license_hint": "Verify current HF dataset card before release.",
    },
    "civitai_flux_prompts_aconexx": {
        "path": "data/prompt-pools/hf-raw-by-repo/Aconexx_CivitAI_Flux_Prompts/Flux_Prompts_Dataset_v1.1.0.jsonl",
        "format": "jsonl",
        "field": "output",
        "citation_hint": "CivitAI FLUX prompt-gallery trace; use as recent-model source sensitivity, not canonical in-the-wild log.",
        "license_hint": "HF card reports apache-2.0; original CivitAI provenance and sanitization should be cited from dataset card.",
    },
    "flux_prompts_chrisgoringe": {
        "path": "data/prompt-pools/hf-raw-by-repo/ChrisGoringe_flux_prompts/data/train-00000-of-00001.parquet",
        "format": "parquet",
        "field": "prompt",
        "citation_hint": "Small FLUX prompt pool; weak provenance, appendix only.",
        "license_hint": "HF card reports apache-2.0.",
    },
    "flux_prompts_regpeter": {
        "path": "data/prompt-pools/hf-raw-by-repo/regpeter_flux_prompts/data/train-00000-of-00001.parquet",
        "format": "parquet",
        "field": "prompt",
        "citation_hint": "Small caption-like FLUX prompt pool; appendix only.",
        "license_hint": "HF card did not expose a license during local inspection; verify before paper release.",
    },
    "flux_improved_k_mktr": {
        "path": "data/prompt-pools/hf-raw-by-repo/k_mktr_improved_flux_prompts/train-0000.parquet",
        "format": "parquet",
        "field": "prompt",
        "citation_hint": "LLM-enhanced FLUX prompt dataset; curated/synthetic source-sensitivity pool, not in-the-wild.",
        "license_hint": "HF card reports mit.",
    },
    "sdxl_refiner_prompts_falah": {
        "paths": [
            "data/prompt-pools/hf-raw-by-repo/Falah_1M_SDXL_Refiner_Prompts/data/train-00000-of-00002-1a170b93d3874d69.parquet",
            "data/prompt-pools/hf-raw-by-repo/Falah_1M_SDXL_Refiner_Prompts/data/train-00001-of-00002-d358755466309b42.parquet",
        ],
        "format": "parquet",
        "field": "prompts",
        "citation_hint": "Author-curated SDXL refiner prompt set; appendix only.",
        "license_hint": "HF card reports apache-2.0.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare prompt-only reference pools")
    parser.add_argument("--raw-root", default="data/prompt-pools/hf-raw")
    parser.add_argument("--output-root", default="data/prompt-pools/2026-04-24")
    parser.add_argument("--min-tokens", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-per-source", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    return parser.parse_args()


def normalize_prompt(text: str) -> str:
    text = text.replace("\u0000", " ").strip()
    text = SPACE_RE.sub(" ", text)
    return text


def stable_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def is_safe_prompt(text: str) -> bool:
    if URL_RE.search(text) or EMAIL_RE.search(text) or PHONE_RE.search(text):
        return False
    return UNSAFE_RE.search(text) is None


def iter_txt(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line


def iter_csv(path: Path, field: str) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row.get(field, "")


def iter_parquet(path: Path, field: str) -> Iterator[str]:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=[field], batch_size=8192):
        for row in batch.to_pylist():
            yield row.get(field, "")


def iter_jsonl(path: Path, field: str) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield row.get(field, "")


def iter_source_prompts(source: dict[str, str]) -> Iterator[str]:
    raw_paths = source["paths"] if "paths" in source else [source["path"]]
    paths = [Path(path) for path in raw_paths]
    fmt = source["format"]
    for path in paths:
        if fmt == "txt":
            yield from iter_txt(path)
        elif fmt == "csv":
            yield from iter_csv(path, source.get("field", "prompt"))
        elif fmt == "parquet":
            yield from iter_parquet(path, source.get("field", "prompt"))
        elif fmt == "jsonl":
            yield from iter_jsonl(path, source.get("field", "prompt"))
        else:
            raise ValueError(f"unsupported source format: {fmt}")


def accept_by_hash(prompt_hash: str, max_per_source: int | None, seed: int) -> bool:
    if max_per_source is None:
        return True
    # Deterministic downsampling without reading the source twice.
    value = int(hashlib.blake2b(f"{seed}:{prompt_hash}".encode(), digest_size=8).hexdigest(), 16)
    return value < int((2**64 - 1) * 0.999999999)


def write_source(
    name: str,
    source: dict[str, str],
    output_root: Path,
    min_tokens: int,
    max_tokens: int,
    max_per_source: int | None,
    sample_seed: int,
) -> dict[str, object]:
    raw_out = output_root / f"{name}.prompt_only.jsonl"
    filtered_out = output_root / f"{name}.filtered.jsonl"
    seen: set[str] = set()
    stats: Counter[str] = Counter()
    token_hist: Counter[int] = Counter()
    raw_written = 0
    filtered_written = 0
    with raw_out.open("w", encoding="utf-8") as raw_handle, filtered_out.open("w", encoding="utf-8") as filtered_handle:
        for raw_index, raw_prompt in enumerate(iter_source_prompts(source)):
            stats["read"] += 1
            prompt = normalize_prompt(str(raw_prompt))
            if not prompt:
                stats["empty"] += 1
                continue
            prompt_hash = stable_hash(prompt.lower())
            if prompt_hash in seen:
                stats["duplicate"] += 1
                continue
            if not accept_by_hash(prompt_hash, max_per_source, sample_seed):
                stats["sampled_out"] += 1
                continue
            seen.add(prompt_hash)
            n_tokens = token_count(prompt)
            token_hist[min(n_tokens, 512)] += 1
            if n_tokens < min_tokens:
                stats["too_short"] += 1
                continue
            if n_tokens > max_tokens:
                stats["too_long"] += 1
                continue
            row = {
                "source": name,
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "tokens_lexical": n_tokens,
                "raw_index": raw_index,
            }
            raw_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_written += 1
            if is_safe_prompt(prompt):
                filtered_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                filtered_written += 1
            else:
                stats["filtered_safety"] += 1
            if max_per_source is not None and raw_written >= max_per_source:
                break
    stats["raw_written"] = raw_written
    stats["filtered_written"] = filtered_written
    return {
        "name": name,
        "input_path": source.get("path", source.get("paths", [])),
        "format": source["format"],
        "prompt_only_jsonl": str(raw_out),
        "filtered_jsonl": str(filtered_out),
        "citation_hint": source.get("citation_hint", ""),
        "license_hint": source.get("license_hint", ""),
        "stats": dict(stats),
        "token_hist_capped_512": dict(sorted(token_hist.items())),
    }


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "output_root": str(output_root),
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "max_per_source": args.max_per_source,
        "sample_seed": args.sample_seed,
        "sources": [],
        "notes": [
            "Prompt pools are prompt-only artifacts and exclude image payloads.",
            "Filtered pools remove URL/email/phone patterns and conservative unsafe keywords.",
            "Stable-Diffusion prompt mirrors are operational prompt traces, not canonical DiffusionDB releases.",
            "CivitAI/FLUX and SDXL prompt pools are source-sensitivity references unless their provenance supports a main claim.",
            "Prompt-support metrics must report source-level results before any aggregation.",
        ],
    }
    for name, source in DEFAULT_SOURCES.items():
        raw_paths = source["paths"] if "paths" in source else [source["path"]]
        paths = [Path(path) for path in raw_paths]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            manifest["sources"].append({"name": name, "missing": True, "input_path": [str(path) for path in paths], "missing_paths": missing})
            continue
        result = write_source(
            name=name,
            source=source,
            output_root=output_root,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            max_per_source=args.max_per_source,
            sample_seed=args.sample_seed,
        )
        manifest["sources"].append(result)
        print(json.dumps({"name": name, "stats": result["stats"]}, ensure_ascii=False))
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
