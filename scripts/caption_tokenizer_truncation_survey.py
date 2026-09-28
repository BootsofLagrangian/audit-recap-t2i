#!/usr/bin/env python3
"""Measure caption truncation risk under candidate text encoders."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

DEFAULT_ENCODERS = [
    ("raw_clip_l14", "openai/clip-vit-large-patch14", 77),
    ("raw_clip_b32", "openai/clip-vit-base-patch32", 77),
    ("siglip_so400m", "google/siglip-so400m-patch14-384", 64),
    ("siglip2_so400m", "google/siglip2-so400m-patch14-384", 64),
    ("longclip_gmp_l14", "zer0int/LongCLIP-GmP-ViT-L-14", 248),
    ("longclip_registers_l14", "zer0int/LongCLIP-Registers-Gated_MLP-ViT-L-14", 248),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption tokenizer truncation survey")
    parser.add_argument("--fair-root", default="data/caption-fair-slices")
    parser.add_argument("--max-pairs", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--role", choices=["local", "reference", "both"], default="local")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--encoder",
        action="append",
        default=[],
        help="Encoder spec name=model_id=max_len. May be repeated. Defaults cover CLIP/SigLIP/LongCLIP.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_encoder_spec(raw: str) -> tuple[str, str, int]:
    parts = raw.split("=")
    if len(parts) != 3:
        raise ValueError(f"encoder spec must be name=model=max_len: {raw}")
    return parts[0], parts[1], int(parts[2])


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fair_jobs(fair_root: Path, max_pairs: int, seed: int, role: str) -> list[dict[str, Any]]:
    jobs = []
    seen = set()
    for summary_path in sorted(fair_root.glob(f"*/n{max_pairs}_seed{seed}/summary.json")):
        summary = json_load(summary_path)
        outputs = summary["outputs"]
        roles = ["local", "reference"] if role == "both" else [role]
        for item_role in roles:
            path_key = "local_jsonl" if item_role == "local" else "reference_jsonl"
            surface = summary["local_surface"] if item_role == "local" else summary["reference_surface"]
            path = outputs[path_key]
            dedupe_key = (item_role, surface, path)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            jobs.append(
                {
                    "comparison": summary["comparison"],
                    "role": item_role,
                    "surface": surface,
                    "path": path,
                    "paired_count": outputs["paired_count"],
                }
            )
    return jobs


def iter_captions(path: Path, max_records: int | None):
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_records is not None and count >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            caption = row.get("caption")
            if isinstance(caption, str):
                count += 1
                yield caption


def quantile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


def summarize_lengths(lengths: list[int], max_len: int) -> dict[str, Any]:
    lengths.sort()
    n = len(lengths)
    over = sum(1 for length in lengths if length > max_len)
    at_or_over = sum(1 for length in lengths if length >= max_len)
    slack = [max_len - length for length in lengths]
    return {
        "records": n,
        "max_len": max_len,
        "truncated_rate_gt_limit": over / n if n else 0.0,
        "at_or_over_limit_rate": at_or_over / n if n else 0.0,
        "mean_tokens": statistics.fmean(lengths) if n else 0.0,
        "p05_tokens": quantile(lengths, 0.05),
        "p50_tokens": quantile(lengths, 0.50),
        "p90_tokens": quantile(lengths, 0.90),
        "p95_tokens": quantile(lengths, 0.95),
        "p99_tokens": quantile(lengths, 0.99),
        "max_tokens": float(lengths[-1]) if n else 0.0,
        "mean_slack": statistics.fmean(slack) if n else 0.0,
    }


def run_one_surface(job: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("transformers is required; install the eval extra with `uv sync --extra eval`") from exc

    tokenizers = []
    for encoder_name, model_id, max_len in job["encoders"]:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=job["trust_remote_code"])
        # We intentionally count untruncated sequences; suppress model_max_length warnings.
        tokenizer.model_max_length = max(int(max_len) * 1000, 1_000_000)
        tokenizers.append((encoder_name, model_id, int(max_len), tokenizer, []))

    batch: list[str] = []
    for caption in iter_captions(Path(job["path"]), job["max_records"]):
        batch.append(caption)
        if len(batch) >= job["batch_size"]:
            for _encoder_name, _model_id, _max_len, tokenizer, lengths in tokenizers:
                encoded = tokenizer(batch, add_special_tokens=True, padding=False, truncation=False)
                lengths.extend(len(ids) for ids in encoded["input_ids"])
            batch = []
    if batch:
        for _encoder_name, _model_id, _max_len, tokenizer, lengths in tokenizers:
            encoded = tokenizer(batch, add_special_tokens=True, padding=False, truncation=False)
            lengths.extend(len(ids) for ids in encoded["input_ids"])

    rows = []
    for encoder_name, model_id, max_len, _tokenizer, lengths in tokenizers:
        summary = summarize_lengths(lengths, max_len)
        rows.append(
            {
                "comparison": job["comparison"],
                "role": job["role"],
                "surface": job["surface"],
                "path": job["path"],
                "paired_count": job["paired_count"],
                "encoder": encoder_name,
                "model_id": model_id,
                **summary,
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "comparison",
        "role",
        "surface",
        "encoder",
        "model_id",
        "records",
        "max_len",
        "truncated_rate_gt_limit",
        "at_or_over_limit_rate",
        "mean_tokens",
        "p50_tokens",
        "p90_tokens",
        "p95_tokens",
        "p99_tokens",
        "max_tokens",
        "mean_slack",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


def main() -> int:
    args = parse_args()
    encoders = [parse_encoder_spec(raw) for raw in args.encoder] if args.encoder else DEFAULT_ENCODERS
    caption_jobs = fair_jobs(Path(args.fair_root), args.max_pairs, args.seed, args.role)
    jobs = [
        {
            **caption_job,
            "encoders": encoders,
            "max_records": args.max_records,
            "batch_size": args.batch_size,
            "trust_remote_code": args.trust_remote_code,
        }
        for caption_job in caption_jobs
    ]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one_surface, job) for job in jobs]
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda row: (row["surface"], row["role"], row["encoder"], row["comparison"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fair_root": args.fair_root,
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "role": args.role,
        "max_records": args.max_records,
        "batch_size": args.batch_size,
        "encoders": [{"name": name, "model_id": model_id, "max_len": max_len} for name, model_id, max_len in encoders],
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_tsv(output.with_suffix(".tsv"), results)
    print(json.dumps({"output": str(output), "tsv": str(output.with_suffix(".tsv")), "rows": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
