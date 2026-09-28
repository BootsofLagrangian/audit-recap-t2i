#!/usr/bin/env python3
"""Compute LongCLIP-style image-caption retrieval separability.

This metric is a frozen dual-encoder compatibility diagnostic, not a
faithfulness certificate. It reports whether each caption distinguishes its
paired image from same-slice negatives, while also reporting text truncation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", action="append", required=True, metavar="LABEL=JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="zer0int/LongCLIP-GmP-ViT-L-14")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--sample-records", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--retrieval-block-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=248)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--save-embeddings", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def parse_surface(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--surface must be LABEL=JSONL: {spec}")
    label, path = spec.split("=", 1)
    return label, Path(path)


def stable_float(*parts: object) -> float:
    raw = ":".join(str(part) for part in parts)
    digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def image_path(row: dict[str, Any]) -> str | None:
    image = row.get("image") if isinstance(row.get("image"), dict) else {}
    local = image.get("local_abs_path") or row.get("image_abs_path") or row.get("image_path")
    if isinstance(local, str) and local:
        return local
    return None


def load_surface(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            caption = row.get("caption")
            if isinstance(caption, str) and caption.strip():
                rows.append(row)
    return rows


def align_rows(surface_rows: dict[str, list[dict[str, Any]]], sample_records: int | None, max_records: int | None, seed: int) -> dict[str, list[dict[str, Any]]]:
    labels = list(surface_rows)
    n = min(len(surface_rows[label]) for label in labels)
    indices = list(range(n))
    if sample_records is not None:
        indices.sort(key=lambda i: stable_float(seed, i))
        indices = indices[:sample_records]
        indices.sort()
    elif max_records is not None:
        indices = indices[:max_records]
    return {label: [surface_rows[label][i] for i in indices] for label in labels}


def load_model(model_id: str, device: str, dtype_name: str, trust_remote_code: bool):
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    dtype = torch_dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code, torch_dtype=dtype)
    model.eval().to(device)
    return tokenizer, image_processor, model


def normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=-1)


def pooled_tensor(output: Any) -> torch.Tensor:
    """Return a tensor embedding from HF tensor/model-output variants."""
    if isinstance(output, torch.Tensor):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    image_embeds = getattr(output, "image_embeds", None)
    if isinstance(image_embeds, torch.Tensor):
        return image_embeds
    text_embeds = getattr(output, "text_embeds", None)
    if isinstance(text_embeds, torch.Tensor):
        return text_embeds
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state[:, 0]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        first = output[0]
        return first[:, 0] if first.ndim == 3 else first
    raise TypeError(f"Cannot extract pooled tensor from {type(output)!r}")


def encode_texts(tokenizer: Any, model: Any, texts: list[str], device: str, max_length: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    embs: list[np.ndarray] = []
    lengths: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            raw = tokenizer(batch, padding=False, truncation=False, add_special_tokens=True)
            lengths.extend(len(ids) for ids in raw["input_ids"])
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            if hasattr(model, "get_text_features"):
                features = pooled_tensor(model.get_text_features(**encoded))
            else:
                features = pooled_tensor(model(**encoded))
            embs.append(normalize(features).cpu().numpy().astype("float32"))
    return np.concatenate(embs, axis=0), np.asarray(lengths, dtype=np.int32)


def encode_images(image_processor: Any, model: Any, rows: list[dict[str, Any]], device: str, batch_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    embs: list[np.ndarray] = []
    kept_indices: list[int] = []
    failures: list[dict[str, Any]] = []
    batch_images: list[Image.Image] = []
    batch_indices: list[int] = []

    def flush() -> None:
        if not batch_images:
            return
        inputs = image_processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            if hasattr(model, "get_image_features"):
                features = pooled_tensor(model.get_image_features(**inputs))
            else:
                features = pooled_tensor(model(**inputs))
        embs.append(normalize(features).cpu().numpy().astype("float32"))
        kept_indices.extend(batch_indices)
        batch_images.clear()
        batch_indices.clear()

    for index, row in enumerate(rows):
        path = image_path(row)
        if path is None:
            failures.append({"index": index, "reason": "missing_image_path"})
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            failures.append({"index": index, "path": path, "reason": repr(exc)[:500]})
            continue
        batch_images.append(image)
        batch_indices.append(index)
        if len(batch_images) >= batch_size:
            flush()
    flush()
    if embs:
        arr = np.concatenate(embs, axis=0)
    else:
        arr = np.zeros((0, 0), dtype=np.float32)
    return arr, {"kept_indices": kept_indices, "failures": failures}


def mean_ci(values: np.ndarray, reps: int, rng: np.random.Generator) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    if reps <= 0 or values.size == 1:
        mean = float(values.mean())
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    means = np.empty(reps, dtype=np.float64)
    n = values.size
    for i in range(reps):
        means[i] = values[rng.integers(0, n, n)].mean()
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def retrieval_metrics(image_emb: np.ndarray, text_emb: np.ndarray, block_size: int) -> dict[str, np.ndarray]:
    n = min(len(image_emb), len(text_emb))
    pos = np.sum(image_emb[:n] * text_emb[:n], axis=1).astype(np.float32)
    max_i2t = np.full(n, -np.inf, dtype=np.float32)
    max_t2i = np.full(n, -np.inf, dtype=np.float32)
    rank_i2t = np.ones(n, dtype=np.int32)
    rank_t2i = np.ones(n, dtype=np.int32)

    for image_start in range(0, n, block_size):
        image_end = min(image_start + block_size, n)
        image_block = image_emb[image_start:image_end]
        image_idx = np.arange(image_start, image_end)
        for text_start in range(0, n, block_size):
            text_end = min(text_start + block_size, n)
            text_block = text_emb[text_start:text_end]
            text_idx = np.arange(text_start, text_end)
            sims = image_block @ text_block.T
            diag_mask = image_idx[:, None] == text_idx[None, :]

            masked = sims.copy()
            masked[diag_mask] = -np.inf
            max_i2t[image_start:image_end] = np.maximum(max_i2t[image_start:image_end], masked.max(axis=1))
            max_t2i[text_start:text_end] = np.maximum(max_t2i[text_start:text_end], masked.max(axis=0))

            greater_i2t = sims > pos[image_start:image_end, None]
            greater_i2t[diag_mask] = False
            rank_i2t[image_start:image_end] += greater_i2t.sum(axis=1).astype(np.int32)

            greater_t2i = sims > pos[text_start:text_end][None, :]
            greater_t2i[diag_mask] = False
            rank_t2i[text_start:text_end] += greater_t2i.sum(axis=0).astype(np.int32)

    return {
        "pos": pos,
        "i2t_margin": (pos - max_i2t).astype(np.float32),
        "t2i_margin": (pos - max_t2i).astype(np.float32),
        "i2t_r1": (rank_i2t <= 1).astype(np.float32),
        "i2t_r5": (rank_i2t <= 5).astype(np.float32),
        "t2i_r1": (rank_t2i <= 1).astype(np.float32),
        "t2i_r5": (rank_t2i <= 5).astype(np.float32),
    }


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_specs = dict(parse_surface(spec) for spec in args.surface)
    raw_rows = {label: load_surface(path) for label, path in surface_specs.items()}
    rows = align_rows(raw_rows, args.sample_records, args.max_records, args.sample_seed)
    labels = list(rows)
    if not labels:
        raise SystemExit("No surfaces provided")

    tokenizer, image_processor, model = load_model(args.model, args.device, args.dtype, args.trust_remote_code)
    image_emb, image_info = encode_images(image_processor, model, rows[labels[0]], args.device, args.batch_size)
    kept_indices = image_info["kept_indices"]
    rng = np.random.default_rng(args.sample_seed)

    summaries: dict[str, Any] = {}
    text_cache: dict[str, np.ndarray] = {}
    token_cache: dict[str, np.ndarray] = {}
    for label in labels:
        kept_rows = [rows[label][index] for index in kept_indices]
        texts = [str(row["caption"]) for row in kept_rows]
        text_emb, token_lengths = encode_texts(tokenizer, model, texts, args.device, args.max_length, args.batch_size)
        text_cache[label] = text_emb
        token_cache[label] = token_lengths
        metrics = retrieval_metrics(image_emb, text_emb, args.retrieval_block_size)
        summaries[label] = {
            "rows": int(len(texts)),
            "token_mean": float(token_lengths.mean()) if len(token_lengths) else 0.0,
            "token_p50": float(np.percentile(token_lengths, 50)) if len(token_lengths) else 0.0,
            "token_p95": float(np.percentile(token_lengths, 95)) if len(token_lengths) else 0.0,
            "truncated_rate_gt_limit": float((token_lengths > args.max_length).mean()) if len(token_lengths) else 0.0,
            "pos_score": mean_ci(metrics["pos"], args.bootstrap_reps, rng),
            "i2t_margin": mean_ci(metrics["i2t_margin"], args.bootstrap_reps, rng),
            "t2i_margin": mean_ci(metrics["t2i_margin"], args.bootstrap_reps, rng),
            "i2t_r_at_1": mean_ci(metrics["i2t_r1"], args.bootstrap_reps, rng),
            "i2t_r_at_5": mean_ci(metrics["i2t_r5"], args.bootstrap_reps, rng),
            "t2i_r_at_1": mean_ci(metrics["t2i_r1"], args.bootstrap_reps, rng),
            "t2i_r_at_5": mean_ci(metrics["t2i_r5"], args.bootstrap_reps, rng),
        }

    payload = {
        "model": args.model,
        "max_length": args.max_length,
        "surface_inputs": {label: str(path) for label, path in surface_specs.items()},
        "labels": labels,
        "image_rows": len(rows[labels[0]]),
        "image_kept": len(kept_indices),
        "image_failures": image_info["failures"][:100],
        "retrieval_block_size": args.retrieval_block_size,
        "bootstrap_reps": args.bootstrap_reps,
        "seconds": round(time.time() - started, 2),
        "summaries": summaries,
    }
    summary_path = output_dir / "longclip_retrieval_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows_tsv = [
        [
            "surface",
            "rows",
            "trunc_gt_248",
            "tok_mean",
            "tok_p95",
            "pos_mean",
            "pos_ci95",
            "i2t_margin_mean",
            "i2t_margin_ci95",
            "i2t_r1",
            "i2t_r5",
            "t2i_margin_mean",
            "t2i_margin_ci95",
            "t2i_r1",
            "t2i_r5",
        ]
    ]
    for label in labels:
        s = summaries[label]
        rows_tsv.append(
            [
                label,
                str(s["rows"]),
                f"{s['truncated_rate_gt_limit']:.4f}",
                f"{s['token_mean']:.2f}",
                f"{s['token_p95']:.1f}",
                f"{s['pos_score']['mean']:.6f}",
                f"[{s['pos_score']['ci95_low']:.6f},{s['pos_score']['ci95_high']:.6f}]",
                f"{s['i2t_margin']['mean']:.6f}",
                f"[{s['i2t_margin']['ci95_low']:.6f},{s['i2t_margin']['ci95_high']:.6f}]",
                f"{s['i2t_r_at_1']['mean']:.4f}",
                f"{s['i2t_r_at_5']['mean']:.4f}",
                f"{s['t2i_margin']['mean']:.6f}",
                f"[{s['t2i_margin']['ci95_low']:.6f},{s['t2i_margin']['ci95_high']:.6f}]",
                f"{s['t2i_r_at_1']['mean']:.4f}",
                f"{s['t2i_r_at_5']['mean']:.4f}",
            ]
        )
    (output_dir / "longclip_retrieval_summary.tsv").write_text(
        "\n".join("\t".join(row) for row in rows_tsv) + "\n",
        encoding="utf-8",
    )
    if args.save_embeddings:
        np.save(output_dir / "image_embeddings.npy", image_emb.astype(np.float16))
        for label, emb in text_cache.items():
            np.save(output_dir / f"text_embeddings_{label}.npy", emb.astype(np.float16))
            np.save(output_dir / f"token_lengths_{label}.npy", token_cache[label])
    print(json.dumps({"summary": str(summary_path), "rows": len(kept_indices), "labels": labels}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
