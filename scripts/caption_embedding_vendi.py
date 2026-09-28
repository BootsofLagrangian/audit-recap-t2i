#!/usr/bin/env python3
"""Encode caption text and compute block Vendi scores.

The script is intentionally split into three subcommands:
- `inspect`: report tokenizer/config limits for candidate encoders
- `encode`: cache normalized text embeddings from JSONL captions
- `vendi`: compute sampled block Vendi/effective-rank summaries from caches

The encoder path is GPU-ready but the same code can be sanity-checked on CPU
with a tiny sample before H200 allocation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


@dataclass
class EmbeddingShard:
    path: str
    rows: int
    dim: int
    dtype: str
    start_row: int
    end_row: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption embedding cache and Vendi utilities")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    inspect = subparsers.add_parser("inspect", help="Inspect tokenizer/model text limits")
    inspect.add_argument("--model", action="append", required=True, help="HF model id/path; may be repeated")
    inspect.add_argument("--trust-remote-code", action="store_true")
    inspect.add_argument(
        "--compat-remote-code",
        action="store_true",
        help="Install small compatibility shims for older HF remote-code embedding models.",
    )

    encode = subparsers.add_parser("encode", help="Extract normalized text embeddings")
    encode.add_argument("--input", required=True, help="JSONL input")
    encode.add_argument("--text-field", default="caption")
    encode.add_argument("--id-field", default=None)
    encode.add_argument("--model", required=True)
    encode.add_argument("--output-dir", required=True)
    encode.add_argument("--max-records", type=int, default=None)
    encode.add_argument(
        "--sample-records",
        type=int,
        default=None,
        help="Reservoir-sample this many records before modulo splitting. Mutually exclusive with --max-records.",
    )
    encode.add_argument("--sample-seed", type=int, default=0)
    encode.add_argument("--split-count", type=int, default=1, help="Modulo split count for multi-GPU extraction")
    encode.add_argument("--split-index", type=int, default=0, help="Modulo split index for this worker")
    encode.add_argument("--batch-size", type=int, default=256)
    encode.add_argument("--max-length", type=int, default=None)
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    encode.add_argument("--embedding-dtype", default="float16", choices=["float16", "float32"])
    encode.add_argument("--shard-rows", type=int, default=100_000)
    encode.add_argument("--pooling", default="auto", choices=["auto", "cls", "mean", "pooler", "last"])
    encode.add_argument("--padding-side", default=None, choices=["left", "right"], help="Override tokenizer padding side")
    encode.add_argument("--text-prefix", default="", help="Prefix applied to every text before tokenization")
    encode.add_argument(
        "--text-template",
        default=None,
        help="Python format template applied before tokenization. Must contain '{text}'. Overrides --text-prefix.",
    )
    encode.add_argument("--trust-remote-code", action="store_true")
    encode.add_argument(
        "--compat-remote-code",
        action="store_true",
        help="Install small compatibility shims for older HF remote-code embedding models.",
    )
    encode.add_argument("--compile", action="store_true")

    bge = subparsers.add_parser("encode-bge-m3", help="Extract official BGE-M3 dense embeddings via FlagEmbedding")
    bge.add_argument("--input", required=True, help="JSONL input")
    bge.add_argument("--text-field", default="caption")
    bge.add_argument("--id-field", default=None)
    bge.add_argument("--model", default="BAAI/bge-m3")
    bge.add_argument("--output-dir", required=True)
    bge.add_argument("--max-records", type=int, default=None)
    bge.add_argument("--sample-records", type=int, default=None)
    bge.add_argument("--sample-seed", type=int, default=0)
    bge.add_argument("--split-count", type=int, default=1)
    bge.add_argument("--split-index", type=int, default=0)
    bge.add_argument("--batch-size", type=int, default=256)
    bge.add_argument("--max-length", type=int, default=512)
    bge.add_argument("--device", default="cuda")
    bge.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=True)
    bge.add_argument("--embedding-dtype", default="float16", choices=["float16", "float32"])
    bge.add_argument("--shard-rows", type=int, default=100_000)
    bge.add_argument("--text-prefix", default="", help="Prefix applied to every text before encoding")
    bge.add_argument("--text-template", default=None, help="Python format template containing '{text}'")
    bge.add_argument("--encode-mode", default="corpus", choices=["corpus", "queries", "encode"])
    bge.add_argument("--query-instruction", default=None, help="Optional BGEM3 query_instruction_for_retrieval")
    bge.add_argument("--query-instruction-format", default="{}{}", help="BGEM3 query_instruction_format")

    st = subparsers.add_parser(
        "encode-sentence-transformer",
        help="Extract embeddings with SentenceTransformer's model-specific encode protocol",
    )
    st.add_argument("--input", required=True, help="JSONL input")
    st.add_argument("--text-field", default="caption")
    st.add_argument("--id-field", default=None)
    st.add_argument("--model", required=True)
    st.add_argument("--output-dir", required=True)
    st.add_argument("--max-records", type=int, default=None)
    st.add_argument("--sample-records", type=int, default=None)
    st.add_argument("--sample-seed", type=int, default=0)
    st.add_argument("--split-count", type=int, default=1)
    st.add_argument("--split-index", type=int, default=0)
    st.add_argument("--batch-size", type=int, default=256)
    st.add_argument("--max-length", type=int, default=None)
    st.add_argument("--device", default="cuda")
    st.add_argument("--embedding-dtype", default="float16", choices=["float16", "float32"])
    st.add_argument("--shard-rows", type=int, default=100_000)
    st.add_argument("--text-prefix", default="", help="Prefix applied to every text before encoding")
    st.add_argument("--text-template", default=None, help="Python format template containing '{text}'")
    st.add_argument("--prompt-name", default=None, help="SentenceTransformer prompt_name, e.g. document or query")

    vendi = subparsers.add_parser("vendi", help="Compute sampled block Vendi from embedding cache")
    vendi.add_argument("--manifest", required=True)
    vendi.add_argument("--output", required=True)
    vendi.add_argument("--block-size", type=int, default=4096)
    vendi.add_argument("--blocks", type=int, default=64)
    vendi.add_argument(
        "--sampling",
        choices=["random", "partition"],
        default="random",
        help="random samples blocks; partition shuffles once and uses every row in disjoint blocks.",
    )
    vendi.add_argument("--seed", type=int, default=0)
    vendi.add_argument("--device", default="cuda")
    vendi.add_argument("--matrix-device", default=None, help="Override device for eigvalsh; defaults to --device")
    vendi.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])

    geom = subparsers.add_parser("geometry", help="Compute embedding-distribution geometry summaries")
    geom.add_argument("--manifest", required=True)
    geom.add_argument("--output", required=True)
    geom.add_argument("--max-rows", type=int, default=100_000)
    geom.add_argument("--seed", type=int, default=0)
    geom.add_argument("--device", default="cuda")
    geom.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])

    knn = subparsers.add_parser("knn", help="Compute exact nearest-neighbor support between two embedding caches")
    knn.add_argument("--query-manifest", required=True)
    knn.add_argument("--gallery-manifest", required=True)
    knn.add_argument("--output", required=True)
    knn.add_argument("--query-max-rows", type=int, default=None)
    knn.add_argument("--gallery-max-rows", type=int, default=None)
    knn.add_argument("--seed", type=int, default=0)
    knn.add_argument("--device", default="cuda")
    knn.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    knn.add_argument("--query-batch-size", type=int, default=1024)
    knn.add_argument(
        "--gallery-chunk-size",
        type=int,
        default=0,
        help="0 keeps the full gallery resident on device; positive values stream gallery chunks.",
    )
    knn.add_argument("--thresholds", default="0.60,0.70,0.75,0.80,0.85,0.90")
    knn.add_argument("--save-scores", default=None, help="Optional .npy path for per-query nearest-neighbor cosine scores")

    support = subparsers.add_parser("support", help="Compute PRDC-style query-in-gallery manifold support")
    support.add_argument("--query-manifest", required=True, help="Prompt/query embedding manifest P")
    support.add_argument("--gallery-manifest", required=True, help="Caption/support embedding manifest C")
    support.add_argument("--output", required=True)
    support.add_argument("--query-max-rows", type=int, default=None)
    support.add_argument("--gallery-max-rows", type=int, default=None)
    support.add_argument("--seed", type=int, default=0)
    support.add_argument("--k", type=int, default=10)
    support.add_argument("--device", default="cuda")
    support.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    support.add_argument("--query-batch-size", type=int, default=512)
    support.add_argument("--gallery-batch-size", type=int, default=512)
    support.add_argument("--save-scores", default=None, help="Optional .npz path for per-query support scores")

    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def numpy_dtype(name: str) -> np.dtype:
    return {"float16": np.float16, "float32": np.float32}[name]


def load_transformers():
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on uv environment
        raise SystemExit("transformers is required. Install the eval extra: `uv sync --extra eval`.") from exc
    return AutoConfig, AutoModel, AutoTokenizer


def install_remote_code_compat() -> None:
    """Compatibility shims for embedding-model remote code.

    Jina v2 imports `transformers.onnx.OnnxConfig`, which is absent in the
    current Transformers build used by this project. Jina v3 also expects the
    legacy `all_tied_weights_keys` property on PreTrainedModel. The shims are
    intentionally minimal and only installed when requested.
    """
    try:
        import transformers
        from transformers import PreTrainedModel
    except ImportError:
        return

    if "transformers.onnx" not in sys.modules:
        onnx_module = types.ModuleType("transformers.onnx")

        class OnnxConfig:  # pragma: no cover - exercised by remote code import
            pass

        onnx_module.OnnxConfig = OnnxConfig
        sys.modules["transformers.onnx"] = onnx_module
        setattr(transformers, "onnx", onnx_module)

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):

        def all_tied_weights_keys(self: Any) -> dict[str, None]:
            stored = getattr(self, "_compat_all_tied_weights_keys", None)
            if stored is not None:
                return stored
            keys = getattr(self, "_tied_weights_keys", None) or []
            return {key: None for key in keys}

        def set_all_tied_weights_keys(self: Any, value: Any) -> None:
            if isinstance(value, dict):
                self._compat_all_tied_weights_keys = value
            elif value is None:
                self._compat_all_tied_weights_keys = {}
            else:
                self._compat_all_tied_weights_keys = {key: None for key in value}

        PreTrainedModel.all_tied_weights_keys = property(  # type: ignore[attr-defined]
            all_tied_weights_keys,
            set_all_tied_weights_keys,
        )

    try:
        import transformers.pytorch_utils as pytorch_utils

        if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
            def find_pruneable_heads_and_indices(
                heads: list[int] | set[int],
                n_heads: int,
                head_size: int,
                already_pruned_heads: set[int],
            ) -> tuple[set[int], torch.Tensor]:
                heads = set(heads) - already_pruned_heads
                mask = torch.ones(n_heads, head_size)
                for head in heads:
                    pruned_before = sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
                    mask[head - pruned_before] = 0
                mask = mask.view(-1).contiguous().eq(1)
                index = torch.arange(len(mask))[mask].long()
                return heads, index

            pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        if not hasattr(pytorch_utils, "prune_linear_layer"):
            from transformers.modeling_utils import prune_linear_layer

            pytorch_utils.prune_linear_layer = prune_linear_layer
    except Exception:
        pass


def iter_jsonl(
    path: Path,
    text_field: str,
    id_field: str | None,
    max_records: int | None,
    split_count: int,
    split_index: int,
) -> Iterable[tuple[str, str | None, int]]:
    emitted = 0
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_records is not None and emitted >= max_records:
                break
            line = line.strip()
            if not line:
                seen += 1
                continue
            row_index = seen
            seen += 1
            if row_index % split_count != split_index:
                continue
            row = json.loads(line)
            text = row.get(text_field)
            if not isinstance(text, str):
                text = ""
            row_id = str(row.get(id_field)) if id_field and row.get(id_field) is not None else None
            emitted += 1
            yield text, row_id, row_index


def iter_jsonl_sampled(
    path: Path,
    text_field: str,
    id_field: str | None,
    sample_records: int,
    sample_seed: int,
    split_count: int,
    split_index: int,
) -> Iterable[tuple[str, str | None, int]]:
    if sample_records < 1:
        raise SystemExit("--sample-records must be >= 1")
    rng = random.Random(sample_seed)
    reservoir: list[tuple[str, str | None, int]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row_index = seen
            seen += 1
            row = json.loads(line)
            text = row.get(text_field)
            if not isinstance(text, str):
                text = ""
            row_id = str(row.get(id_field)) if id_field and row.get(id_field) is not None else None
            item = (text, row_id, row_index)
            if len(reservoir) < sample_records:
                reservoir.append(item)
            else:
                replace_index = rng.randrange(seen)
                if replace_index < sample_records:
                    reservoir[replace_index] = item
    reservoir.sort(key=lambda item: item[2])
    for emitted, item in enumerate(reservoir):
        if emitted % split_count == split_index:
            yield item


def batched(items: Iterable[tuple[str, str | None, int]], batch_size: int) -> Iterable[list[tuple[str, str | None, int]]]:
    batch: list[tuple[str, str | None, int]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def config_text_limit(config: Any) -> int | None:
    candidates = []
    for obj in [config, getattr(config, "text_config", None)]:
        if obj is None:
            continue
        for name in ["max_position_embeddings", "max_sequence_length", "context_length", "seq_length"]:
            value = getattr(obj, name, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)
    return min(candidates) if candidates else None


def inspect_models(args: argparse.Namespace) -> int:
    if args.compat_remote_code:
        install_remote_code_compat()
    AutoConfig, _AutoModel, AutoTokenizer = load_transformers()
    rows = []
    for model_id in args.model:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
        rows.append(
            {
                "model": model_id,
                "model_type": getattr(config, "model_type", None),
                "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
                "config_text_limit": config_text_limit(config),
                "text_config_max_position_embeddings": getattr(getattr(config, "text_config", None), "max_position_embeddings", None),
                "max_position_embeddings": getattr(config, "max_position_embeddings", None),
                "projection_dim": getattr(config, "projection_dim", None) or getattr(config, "projection_size", None),
                "hidden_size": getattr(config, "hidden_size", None) or getattr(getattr(config, "text_config", None), "hidden_size", None),
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


def load_encoder(
    model_id: str,
    device: str,
    dtype: str,
    trust_remote_code: bool,
    compile_model: bool,
    compat_remote_code: bool,
    padding_side: str | None,
):
    if compat_remote_code:
        install_remote_code_compat()
    AutoConfig, AutoModel, AutoTokenizer = load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    config = None
    if compat_remote_code:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        for name, value in {
            "is_decoder": False,
            "add_cross_attention": False,
            "chunk_size_feed_forward": 0,
            "use_return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
        }.items():
            if not hasattr(config, name):
                setattr(config, name, value)
    model = AutoModel.from_pretrained(
        model_id,
        config=config,
        dtype=torch_dtype(dtype),
        trust_remote_code=trust_remote_code,
    )
    model.eval().to(device)
    if compile_model:
        model = torch.compile(model)
    return tokenizer, model


def pool_outputs(model: Any, outputs: Any, encoded: dict[str, torch.Tensor], pooling: str) -> torch.Tensor:
    if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        return outputs.text_embeds
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None and pooling in {"auto", "pooler"}:
        return outputs.pooler_output
    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    if pooling == "last":
        attention = encoded.get("attention_mask")
        if attention is None:
            return hidden[:, -1]
        left_padding = bool((attention[:, -1].sum() == attention.shape[0]).item())
        if left_padding:
            return hidden[:, -1]
        sequence_lengths = attention.sum(dim=1) - 1
        batch_size = hidden.shape[0]
        return hidden[torch.arange(batch_size, device=hidden.device), sequence_lengths]
    if pooling == "cls":
        return hidden[:, 0]
    attention = encoded.get("attention_mask")
    if pooling in {"auto", "mean"} and attention is not None:
        weights = attention.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return hidden[:, 0]


@torch.inference_mode()
def encode_batch(
    tokenizer: Any,
    model: Any,
    texts: list[str],
    device: str,
    max_length: int | None,
    pooling: str,
) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    if hasattr(model, "get_text_features"):
        features = model.get_text_features(**encoded)
        if not isinstance(features, torch.Tensor):
            features = pool_outputs(model, features, encoded, pooling)
    else:
        outputs = model(**encoded)
        features = pool_outputs(model, outputs, encoded, pooling)
    features = torch.nn.functional.normalize(features.float(), dim=-1)
    return features.cpu()


def flush_shard(
    output_dir: Path,
    shard_index: int,
    start_row: int,
    rows: list[np.ndarray],
    embedding_dtype: str,
) -> EmbeddingShard:
    array = np.asarray(rows, dtype=numpy_dtype(embedding_dtype))
    path = output_dir / f"embeddings-{shard_index:05d}.npy"
    np.save(path, array)
    return EmbeddingShard(
        path=str(path),
        rows=int(array.shape[0]),
        dim=int(array.shape[1]) if array.ndim == 2 else 0,
        dtype=embedding_dtype,
        start_row=start_row,
        end_row=start_row + int(array.shape[0]),
    )


def encode_main(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_encoder(
        args.model,
        args.device,
        args.dtype,
        args.trust_remote_code,
        args.compile,
        args.compat_remote_code,
        args.padding_side,
    )
    config_limit = config_text_limit(getattr(model, "config", None))
    max_length = args.max_length or config_limit or getattr(tokenizer, "model_max_length", None)
    if isinstance(max_length, int) and max_length > 1_000_000:
        max_length = None

    rows: list[np.ndarray] = []
    row_ids: list[str | None] = []
    row_indices: list[int] = []
    shards: list[EmbeddingShard] = []
    total = 0
    shard_start = 0
    started = time.time()
    if args.split_count < 1:
        raise SystemExit("--split-count must be >= 1")
    if not (0 <= args.split_index < args.split_count):
        raise SystemExit("--split-index must satisfy 0 <= split_index < split_count")
    if args.sample_records is not None and args.max_records is not None:
        raise SystemExit("--sample-records and --max-records are mutually exclusive")
    if args.text_template is not None and "{text}" not in args.text_template:
        raise SystemExit("--text-template must contain '{text}'")
    if args.sample_records is not None:
        source = iter_jsonl_sampled(
            Path(args.input),
            args.text_field,
            args.id_field,
            args.sample_records,
            args.sample_seed,
            args.split_count,
            args.split_index,
        )
    else:
        source = iter_jsonl(
            Path(args.input),
            args.text_field,
            args.id_field,
            args.max_records,
            args.split_count,
            args.split_index,
        )
    for batch in batched(source, args.batch_size):
        texts = [text for text, _row_id, _row_index in batch]
        if args.text_template is not None:
            texts = [args.text_template.format(text=text) for text in texts]
        elif args.text_prefix:
            texts = [f"{args.text_prefix}{text}" for text in texts]
        ids = [row_id for _text, row_id, _row_index in batch]
        indices = [row_index for _text, _row_id, row_index in batch]
        features = encode_batch(tokenizer, model, texts, args.device, max_length, args.pooling)
        rows.extend(features.numpy())
        row_ids.extend(ids)
        row_indices.extend(indices)
        total += len(batch)
        if len(rows) >= args.shard_rows:
            shards.append(flush_shard(output_dir, len(shards), shard_start, rows, args.embedding_dtype))
            shard_start += len(rows)
            rows = []
    if rows:
        shards.append(flush_shard(output_dir, len(shards), shard_start, rows, args.embedding_dtype))

    if row_indices:
        with (output_dir / "row_ids.jsonl").open("w", encoding="utf-8") as handle:
            for index, (row_id, row_index) in enumerate(zip(row_ids, row_indices, strict=True)):
                handle.write(
                    json.dumps(
                        {"split_row": index, "source_row": row_index, "id": row_id},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    manifest = {
        "input": args.input,
        "text_field": args.text_field,
        "id_field": args.id_field,
        "model": args.model,
        "max_length": max_length,
        "max_records": args.max_records,
        "sample_records": args.sample_records,
        "sample_seed": args.sample_seed,
        "split_count": args.split_count,
        "split_index": args.split_index,
        "pooling": args.pooling,
        "padding_side": getattr(tokenizer, "padding_side", None),
        "text_prefix": args.text_prefix,
        "text_template": args.text_template,
        "compat_remote_code": args.compat_remote_code,
        "device": args.device,
        "dtype": args.dtype,
        "embedding_dtype": args.embedding_dtype,
        "rows": total,
        "seconds": round(time.time() - started, 3),
        "rows_per_second": round(total / max(time.time() - started, 1e-6), 3),
        "shards": [asdict(shard) for shard in shards],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": total, "shards": len(shards), "max_length": max_length}, indent=2))
    return 0


def encode_bge_m3_main(args: argparse.Namespace) -> int:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise SystemExit("FlagEmbedding is required for encode-bge-m3. Install with `uv sync --extra eval`.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.split_count < 1:
        raise SystemExit("--split-count must be >= 1")
    if not (0 <= args.split_index < args.split_count):
        raise SystemExit("--split-index must satisfy 0 <= split_index < split_count")
    if args.sample_records is not None and args.max_records is not None:
        raise SystemExit("--sample-records and --max-records are mutually exclusive")
    if args.text_template is not None and "{text}" not in args.text_template:
        raise SystemExit("--text-template must contain '{text}'")

    model = BGEM3FlagModel(
        args.model,
        normalize_embeddings=True,
        use_fp16=args.use_fp16,
        devices=args.device,
        pooling_method="cls",
        batch_size=args.batch_size,
        query_max_length=args.max_length,
        passage_max_length=args.max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
        query_instruction_for_retrieval=args.query_instruction,
        query_instruction_format=args.query_instruction_format,
    )
    if args.sample_records is not None:
        source = iter_jsonl_sampled(
            Path(args.input),
            args.text_field,
            args.id_field,
            args.sample_records,
            args.sample_seed,
            args.split_count,
            args.split_index,
        )
    else:
        source = iter_jsonl(
            Path(args.input),
            args.text_field,
            args.id_field,
            args.max_records,
            args.split_count,
            args.split_index,
        )

    rows: list[np.ndarray] = []
    row_ids: list[str | None] = []
    row_indices: list[int] = []
    shards: list[EmbeddingShard] = []
    total = 0
    shard_start = 0
    started = time.time()
    for batch in batched(source, args.batch_size):
        texts = [text for text, _row_id, _row_index in batch]
        if args.text_template is not None:
            texts = [args.text_template.format(text=text) for text in texts]
        elif args.text_prefix:
            texts = [f"{args.text_prefix}{text}" for text in texts]
        ids = [row_id for _text, row_id, _row_index in batch]
        indices = [row_index for _text, _row_id, row_index in batch]
        encode_fn = {
            "corpus": model.encode_corpus,
            "queries": model.encode_queries,
            "encode": model.encode,
        }[args.encode_mode]
        encoded = encode_fn(
            texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        features = np.asarray(encoded["dense_vecs"], dtype=np.float32)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        rows.extend(features)
        row_ids.extend(ids)
        row_indices.extend(indices)
        total += len(batch)
        if len(rows) >= args.shard_rows:
            shards.append(flush_shard(output_dir, len(shards), shard_start, rows, args.embedding_dtype))
            shard_start += len(rows)
            rows = []
    if rows:
        shards.append(flush_shard(output_dir, len(shards), shard_start, rows, args.embedding_dtype))

    if row_indices:
        with (output_dir / "row_ids.jsonl").open("w", encoding="utf-8") as handle:
            for index, (row_id, row_index) in enumerate(zip(row_ids, row_indices, strict=True)):
                handle.write(
                    json.dumps(
                        {"split_row": index, "source_row": row_index, "id": row_id},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    elapsed = time.time() - started
    manifest = {
        "input": args.input,
        "text_field": args.text_field,
        "id_field": args.id_field,
        "model": args.model,
        "backend": "FlagEmbedding.BGEM3FlagModel",
        "max_length": args.max_length,
        "max_records": args.max_records,
        "sample_records": args.sample_records,
        "sample_seed": args.sample_seed,
        "split_count": args.split_count,
        "split_index": args.split_index,
        "pooling": "cls",
        "encode_mode": args.encode_mode,
        "normalize_embeddings": True,
        "text_prefix": args.text_prefix,
        "text_template": args.text_template,
        "query_instruction": args.query_instruction,
        "query_instruction_format": args.query_instruction_format,
        "device": args.device,
        "use_fp16": args.use_fp16,
        "embedding_dtype": args.embedding_dtype,
        "rows": total,
        "seconds": round(elapsed, 3),
        "rows_per_second": round(total / max(elapsed, 1e-6), 3),
        "shards": [asdict(shard) for shard in shards],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": total, "shards": len(shards), "max_length": args.max_length}, indent=2))
    return 0


def encode_sentence_transformer_main(args: argparse.Namespace) -> int:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("sentence-transformers is required. Run `uv sync --extra eval`.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.split_count < 1:
        raise SystemExit("--split-count must be >= 1")
    if not (0 <= args.split_index < args.split_count):
        raise SystemExit("--split-index must satisfy 0 <= split_index < split_count")
    if args.sample_records is not None and args.max_records is not None:
        raise SystemExit("--sample-records and --max-records are mutually exclusive")
    if args.text_template is not None and "{text}" not in args.text_template:
        raise SystemExit("--text-template must contain '{text}'")

    model = SentenceTransformer(args.model, device=args.device)
    if args.max_length is not None:
        model.max_seq_length = args.max_length
    max_length = int(model.max_seq_length) if getattr(model, "max_seq_length", None) is not None else args.max_length
    if args.sample_records is not None:
        source = iter_jsonl_sampled(
            Path(args.input),
            args.text_field,
            args.id_field,
            args.sample_records,
            args.sample_seed,
            args.split_count,
            args.split_index,
        )
    else:
        source = iter_jsonl(
            Path(args.input),
            args.text_field,
            args.id_field,
            args.max_records,
            args.split_count,
            args.split_index,
        )

    rows: list[np.ndarray] = []
    row_ids: list[str | None] = []
    row_indices: list[int] = []
    shards: list[EmbeddingShard] = []
    total = 0
    shard_start = 0
    started = time.time()
    for batch in batched(source, args.batch_size):
        texts = [text for text, _row_id, _row_index in batch]
        if args.text_template is not None:
            texts = [args.text_template.format(text=text) for text in texts]
        elif args.text_prefix:
            texts = [f"{args.text_prefix}{text}" for text in texts]
        ids = [row_id for _text, row_id, _row_index in batch]
        indices = [row_index for _text, _row_id, row_index in batch]
        encode_kwargs = {
            "batch_size": args.batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if args.prompt_name is not None:
            encode_kwargs["prompt_name"] = args.prompt_name
        features = model.encode(texts, **encode_kwargs)
        features = np.asarray(features, dtype=np.float32)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        rows.extend(features)
        row_ids.extend(ids)
        row_indices.extend(indices)
        total += len(batch)
        if len(rows) >= args.shard_rows:
            shards.append(flush_shard(output_dir, len(shards), shard_start, rows, args.embedding_dtype))
            shard_start += len(rows)
            rows = []
    if rows:
        shards.append(flush_shard(output_dir, len(shards), shard_start, rows, args.embedding_dtype))

    if row_indices:
        with (output_dir / "row_ids.jsonl").open("w", encoding="utf-8") as handle:
            for index, (row_id, row_index) in enumerate(zip(row_ids, row_indices, strict=True)):
                handle.write(
                    json.dumps(
                        {"split_row": index, "source_row": row_index, "id": row_id},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    elapsed = time.time() - started
    manifest = {
        "input": args.input,
        "text_field": args.text_field,
        "id_field": args.id_field,
        "model": args.model,
        "backend": "sentence_transformers.SentenceTransformer",
        "max_length": max_length,
        "max_records": args.max_records,
        "sample_records": args.sample_records,
        "sample_seed": args.sample_seed,
        "split_count": args.split_count,
        "split_index": args.split_index,
        "pooling": "model_default",
        "normalize_embeddings": True,
        "text_prefix": args.text_prefix,
        "text_template": args.text_template,
        "prompt_name": args.prompt_name,
        "available_prompts": getattr(model, "prompts", None),
        "device": args.device,
        "embedding_dtype": args.embedding_dtype,
        "rows": total,
        "seconds": round(elapsed, 3),
        "rows_per_second": round(total / max(elapsed, 1e-6), 3),
        "shards": [asdict(shard) for shard in shards],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": total, "shards": len(shards), "max_length": max_length}, indent=2))
    return 0


def load_embedding_manifest(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    arrays = [np.load(shard["path"], mmap_mode="r") for shard in manifest["shards"]]
    if not arrays:
        return manifest, np.zeros((0, 0), dtype=np.float32)
    return manifest, np.concatenate(arrays, axis=0)


def sample_embeddings(embeddings: np.ndarray, max_rows: int | None, seed: int) -> tuple[np.ndarray, list[int]]:
    n = int(embeddings.shape[0])
    if max_rows is None or max_rows >= n:
        indices = list(range(n))
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(n), max_rows))
    return np.asarray(embeddings[indices], dtype=np.float32), indices


def vendi_from_block(block: torch.Tensor) -> dict[str, float]:
    block = torch.nn.functional.normalize(block.float(), dim=-1)
    kernel = block @ block.T
    eigenvalues = torch.linalg.eigvalsh(kernel).clamp_min(0)
    total = eigenvalues.sum().clamp_min(1e-12)
    probs = eigenvalues / total
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    vendi = torch.exp(entropy)
    return {
        "vendi": float(vendi.item()),
        "effective_rank": float(vendi.item()),
        "trace": float(total.item()),
        "max_eigen_prob": float(probs.max().item()),
    }


def mean_ci(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half = 1.96 * math.sqrt(variance / len(values))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def parse_thresholds(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not -1.0 <= value <= 1.0:
            raise SystemExit(f"invalid cosine threshold outside [-1, 1]: {value}")
        values.append(value)
    if not values:
        raise SystemExit("--thresholds must contain at least one value")
    return values


def summarize_scores(scores: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    percentiles = {
        f"p{percentile:02d}": float(np.percentile(scores, percentile))
        for percentile in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }
    support = {
        f"support_at_{threshold:.2f}": float(np.mean(scores >= threshold))
        for threshold in thresholds
    }
    return {
        "mean_nn_cosine": float(np.mean(scores)),
        "std_nn_cosine": float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0,
        **percentiles,
        **support,
    }


def summarize_support(covered: np.ndarray, density: np.ndarray, nn_cosine: np.ndarray) -> dict[str, Any]:
    nn_distance = 1.0 - nn_cosine
    return {
        "coverage": float(np.mean(covered)),
        "density": float(np.mean(density)),
        "density_p50": float(np.percentile(density, 50)),
        "density_p95": float(np.percentile(density, 95)),
        "nn_cosine_mean": float(np.mean(nn_cosine)),
        "nn_cosine_p50": float(np.percentile(nn_cosine, 50)),
        "nn_cosine_p05": float(np.percentile(nn_cosine, 5)),
        "nn_distance_p95": float(np.percentile(nn_distance, 95)),
        "nn_distance_p99": float(np.percentile(nn_distance, 99)),
    }


@torch.inference_mode()
def exact_nn_cosine(
    query: np.ndarray,
    gallery: np.ndarray,
    device: str,
    dtype: torch.dtype,
    query_batch_size: int,
    gallery_chunk_size: int,
) -> np.ndarray:
    if query.ndim != 2 or gallery.ndim != 2:
        raise SystemExit("query and gallery embeddings must be 2D arrays")
    if query.shape[1] != gallery.shape[1]:
        raise SystemExit(f"dimension mismatch: query dim {query.shape[1]} vs gallery dim {gallery.shape[1]}")
    if query.shape[0] == 0 or gallery.shape[0] == 0:
        raise SystemExit("query and gallery embeddings must be non-empty")
    if query_batch_size < 1:
        raise SystemExit("--query-batch-size must be >= 1")
    if gallery_chunk_size < 0:
        raise SystemExit("--gallery-chunk-size must be >= 0")

    scores: list[np.ndarray] = []
    if gallery_chunk_size == 0:
        gallery_tensor = torch.from_numpy(gallery).to(device=device, dtype=dtype)
        gallery_tensor = torch.nn.functional.normalize(gallery_tensor.float(), dim=-1).to(dtype)
        gallery_t = gallery_tensor.T.contiguous()
        for start in range(0, query.shape[0], query_batch_size):
            query_tensor = torch.from_numpy(query[start : start + query_batch_size]).to(device=device, dtype=dtype)
            query_tensor = torch.nn.functional.normalize(query_tensor.float(), dim=-1).to(dtype)
            sims = query_tensor @ gallery_t
            scores.append(sims.float().max(dim=1).values.cpu().numpy())
        return np.concatenate(scores, axis=0)

    for start in range(0, query.shape[0], query_batch_size):
        query_tensor = torch.from_numpy(query[start : start + query_batch_size]).to(device=device, dtype=dtype)
        query_tensor = torch.nn.functional.normalize(query_tensor.float(), dim=-1).to(dtype)
        best = torch.full((query_tensor.shape[0],), -2.0, device=device, dtype=torch.float32)
        for gallery_start in range(0, gallery.shape[0], gallery_chunk_size):
            gallery_tensor = torch.from_numpy(gallery[gallery_start : gallery_start + gallery_chunk_size]).to(device=device, dtype=dtype)
            gallery_tensor = torch.nn.functional.normalize(gallery_tensor.float(), dim=-1).to(dtype)
            sims = query_tensor @ gallery_tensor.T
            best = torch.maximum(best, sims.float().max(dim=1).values)
        scores.append(best.cpu().numpy())
    return np.concatenate(scores, axis=0)


@torch.inference_mode()
def kth_self_neighbor_cosine(
    gallery: np.ndarray,
    k: int,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    if k < 1:
        raise SystemExit("--k must be >= 1")
    if gallery.shape[0] <= k:
        raise SystemExit(f"gallery rows ({gallery.shape[0]}) must be > k ({k})")
    if batch_size < 1:
        raise SystemExit("--gallery-batch-size must be >= 1")
    gallery_tensor = torch.from_numpy(gallery).to(device=device, dtype=dtype)
    gallery_tensor = torch.nn.functional.normalize(gallery_tensor.float(), dim=-1).to(dtype)
    gallery_t = gallery_tensor.T.contiguous()
    thresholds: list[np.ndarray] = []
    for start in range(0, gallery.shape[0], batch_size):
        stop = min(start + batch_size, gallery.shape[0])
        sims = gallery_tensor[start:stop] @ gallery_t
        row_indices = torch.arange(stop - start, device=device)
        sims[row_indices, torch.arange(start, stop, device=device)] = -2.0
        kth = torch.topk(sims.float(), k=k, dim=1).values[:, -1]
        thresholds.append(kth.cpu().numpy())
    return np.concatenate(thresholds, axis=0)


@torch.inference_mode()
def prdc_query_in_gallery_support(
    query: np.ndarray,
    gallery: np.ndarray,
    gallery_thresholds: np.ndarray,
    k: int,
    device: str,
    dtype: torch.dtype,
    query_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if query_batch_size < 1:
        raise SystemExit("--query-batch-size must be >= 1")
    gallery_tensor = torch.from_numpy(gallery).to(device=device, dtype=dtype)
    gallery_tensor = torch.nn.functional.normalize(gallery_tensor.float(), dim=-1).to(dtype)
    gallery_t = gallery_tensor.T.contiguous()
    thresholds = torch.from_numpy(gallery_thresholds.astype(np.float32)).to(device=device)
    covered_rows: list[np.ndarray] = []
    density_rows: list[np.ndarray] = []
    nn_rows: list[np.ndarray] = []
    for start in range(0, query.shape[0], query_batch_size):
        query_tensor = torch.from_numpy(query[start : start + query_batch_size]).to(device=device, dtype=dtype)
        query_tensor = torch.nn.functional.normalize(query_tensor.float(), dim=-1).to(dtype)
        sims = (query_tensor @ gallery_t).float()
        support_hits = sims >= thresholds.unsqueeze(0)
        hit_counts = support_hits.sum(dim=1).float()
        covered_rows.append((hit_counts > 0).cpu().numpy())
        density_rows.append((hit_counts / float(k)).cpu().numpy())
        nn_rows.append(sims.max(dim=1).values.cpu().numpy())
    return (
        np.concatenate(covered_rows, axis=0),
        np.concatenate(density_rows, axis=0),
        np.concatenate(nn_rows, axis=0),
    )


def vendi_main(args: argparse.Namespace) -> int:
    manifest, embeddings = load_embedding_manifest(Path(args.manifest))
    n = int(embeddings.shape[0])
    if n == 0:
        raise SystemExit("empty embedding cache")
    block_size = min(args.block_size, n)
    rng = random.Random(args.seed)
    matrix_device = args.matrix_device or args.device
    dtype = torch_dtype(args.dtype)
    block_rows = []
    if args.sampling == "partition":
        order = list(range(n))
        rng.shuffle(order)
        index_blocks = [order[start : start + block_size] for start in range(0, n, block_size)]
        if index_blocks and len(index_blocks[-1]) < max(2, block_size // 2):
            # Avoid a tiny tail block with a non-comparable Vendi scale.
            index_blocks[-2].extend(index_blocks[-1])
            index_blocks.pop()
    else:
        index_blocks = [
            rng.sample(range(n), block_size) if block_size < n else list(range(n))
            for _ in range(args.blocks)
        ]
    for block_index, indices in enumerate(index_blocks):
        array = np.asarray(embeddings[indices], dtype=np.float32)
        block = torch.from_numpy(array).to(matrix_device, dtype=dtype)
        stats = vendi_from_block(block)
        stats.update({"block_index": block_index, "block_size": len(indices)})
        block_rows.append(stats)
    vendi_values = [row["vendi"] for row in block_rows]
    payload = {
        "embedding_manifest": args.manifest,
        "source_model": manifest.get("model"),
        "source_rows": n,
        "block_size": block_size,
        "blocks": len(block_rows),
        "requested_blocks": args.blocks,
        "sampling": args.sampling,
        "seed": args.seed,
        "device": matrix_device,
        "summary": {
            "vendi": mean_ci(vendi_values),
            "max_eigen_prob": mean_ci([row["max_eigen_prob"] for row in block_rows]),
        },
        "block_rows": block_rows,
        "boundary": "Vendi is an embedding-space semantic diversity metric; it does not measure faithfulness, density, or downstream utility.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "vendi_mean": payload["summary"]["vendi"]["mean"], "blocks": args.blocks}, indent=2))
    return 0


def geometry_main(args: argparse.Namespace) -> int:
    manifest, embeddings = load_embedding_manifest(Path(args.manifest))
    n = int(embeddings.shape[0])
    if n == 0:
        raise SystemExit("empty embedding cache")
    rng = np.random.default_rng(args.seed)
    take = min(args.max_rows, n)
    indices = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
    x = torch.from_numpy(np.asarray(embeddings[indices], dtype=np.float32)).to(args.device, dtype=torch_dtype(args.dtype))
    x = torch.nn.functional.normalize(x.float(), dim=-1)
    centroid = torch.nn.functional.normalize(x.mean(dim=0, keepdim=True), dim=-1)
    cosine_to_centroid = (x @ centroid.T).squeeze(1)
    centered = x - x.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(take - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    eig_sum = eig.sum().clamp_min(1e-12)
    probs = eig / eig_sum
    spectral_entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    erank = torch.exp(spectral_entropy)
    participation = eig_sum.square() / eig.square().sum().clamp_min(1e-12)
    payload = {
        "embedding_manifest": args.manifest,
        "source_model": manifest.get("model"),
        "source_rows": n,
        "sample_rows": take,
        "seed": args.seed,
        "device": args.device,
        "metrics": {
            "mean_cosine_to_centroid": float(cosine_to_centroid.mean().item()),
            "std_cosine_to_centroid": float(cosine_to_centroid.std(unbiased=True).item()) if take > 1 else 0.0,
            "mean_pairwise_cosine_estimate": float((x.mean(dim=0).square().sum().item() * take - 1.0) / max(take - 1, 1)),
            "cov_effective_rank": float(erank.item()),
            "cov_participation_ratio": float(participation.item()),
            "cov_top1_mass": float((eig.max() / eig_sum).item()),
            "cov_top10_mass": float((eig.topk(min(10, eig.numel())).values.sum() / eig_sum).item()),
            "cov_trace": float(eig_sum.item()),
        },
        "boundary": "Geometry metrics describe embedding distribution shape: concentration, anisotropy, and effective dimensionality. They do not measure faithfulness or prompt support.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["metrics"]}, indent=2))
    return 0


def knn_main(args: argparse.Namespace) -> int:
    query_manifest, query_embeddings_all = load_embedding_manifest(Path(args.query_manifest))
    gallery_manifest, gallery_embeddings_all = load_embedding_manifest(Path(args.gallery_manifest))
    query_embeddings, query_indices = sample_embeddings(query_embeddings_all, args.query_max_rows, args.seed)
    gallery_embeddings, gallery_indices = sample_embeddings(gallery_embeddings_all, args.gallery_max_rows, args.seed + 1)
    started = time.time()
    scores = exact_nn_cosine(
        query_embeddings,
        gallery_embeddings,
        args.device,
        torch_dtype(args.dtype),
        args.query_batch_size,
        args.gallery_chunk_size,
    )
    thresholds = parse_thresholds(args.thresholds)
    payload = {
        "query_manifest": args.query_manifest,
        "gallery_manifest": args.gallery_manifest,
        "query_model": query_manifest.get("model"),
        "gallery_model": gallery_manifest.get("model"),
        "query_source_rows": int(query_embeddings_all.shape[0]),
        "gallery_source_rows": int(gallery_embeddings_all.shape[0]),
        "query_rows": int(query_embeddings.shape[0]),
        "gallery_rows": int(gallery_embeddings.shape[0]),
        "query_seed": args.seed,
        "gallery_seed": args.seed + 1,
        "query_indices_preview": query_indices[:10],
        "gallery_indices_preview": gallery_indices[:10],
        "device": args.device,
        "dtype": args.dtype,
        "query_batch_size": args.query_batch_size,
        "gallery_chunk_size": args.gallery_chunk_size,
        "seconds": round(time.time() - started, 3),
        "metrics": summarize_scores(scores, thresholds),
        "boundary": (
            "kNN support measures nearest-neighbor coverage in the chosen embedding space. "
            "It is directional, encoder-dependent, and not a faithfulness or density metric."
        ),
    }
    if args.save_scores is not None:
        score_path = Path(args.save_scores)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(score_path, scores.astype(np.float32))
        payload["score_path"] = str(score_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "query_rows": payload["query_rows"], "gallery_rows": payload["gallery_rows"], **payload["metrics"]}, indent=2))
    return 0


def support_main(args: argparse.Namespace) -> int:
    query_manifest, query_embeddings_all = load_embedding_manifest(Path(args.query_manifest))
    gallery_manifest, gallery_embeddings_all = load_embedding_manifest(Path(args.gallery_manifest))
    query_embeddings, query_indices = sample_embeddings(query_embeddings_all, args.query_max_rows, args.seed)
    gallery_embeddings, gallery_indices = sample_embeddings(gallery_embeddings_all, args.gallery_max_rows, args.seed + 1)
    started = time.time()
    gallery_thresholds = kth_self_neighbor_cosine(
        gallery_embeddings,
        args.k,
        args.device,
        torch_dtype(args.dtype),
        args.gallery_batch_size,
    )
    covered, density, nn_cosine = prdc_query_in_gallery_support(
        query_embeddings,
        gallery_embeddings,
        gallery_thresholds,
        args.k,
        args.device,
        torch_dtype(args.dtype),
        args.query_batch_size,
    )
    payload = {
        "query_manifest": args.query_manifest,
        "gallery_manifest": args.gallery_manifest,
        "query_model": query_manifest.get("model"),
        "gallery_model": gallery_manifest.get("model"),
        "query_source_rows": int(query_embeddings_all.shape[0]),
        "gallery_source_rows": int(gallery_embeddings_all.shape[0]),
        "query_rows": int(query_embeddings.shape[0]),
        "gallery_rows": int(gallery_embeddings.shape[0]),
        "query_seed": args.seed,
        "gallery_seed": args.seed + 1,
        "query_indices_preview": query_indices[:10],
        "gallery_indices_preview": gallery_indices[:10],
        "k": args.k,
        "device": args.device,
        "dtype": args.dtype,
        "query_batch_size": args.query_batch_size,
        "gallery_batch_size": args.gallery_batch_size,
        "seconds": round(time.time() - started, 3),
        "gallery_thresholds": {
            "mean_kth_neighbor_cosine": float(np.mean(gallery_thresholds)),
            "p05_kth_neighbor_cosine": float(np.percentile(gallery_thresholds, 5)),
            "p50_kth_neighbor_cosine": float(np.percentile(gallery_thresholds, 50)),
            "p95_kth_neighbor_cosine": float(np.percentile(gallery_thresholds, 95)),
        },
        "metrics": summarize_support(covered, density, nn_cosine),
        "boundary": (
            "P-in-C support is a PRDC-style embedding-manifold estimate: query points are covered "
            "when they fall inside at least one gallery kNN ball. It measures support in the chosen "
            "embedding space, not image faithfulness or overall caption quality."
        ),
    }
    if args.save_scores is not None:
        score_path = Path(args.save_scores)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            score_path,
            covered=covered.astype(np.bool_),
            density=density.astype(np.float32),
            nn_cosine=nn_cosine.astype(np.float32),
            gallery_thresholds=gallery_thresholds.astype(np.float32),
        )
        payload["score_path"] = str(score_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "query_rows": payload["query_rows"], "gallery_rows": payload["gallery_rows"], **payload["metrics"]}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.cmd == "inspect":
        return inspect_models(args)
    if args.cmd == "encode":
        return encode_main(args)
    if args.cmd == "encode-bge-m3":
        return encode_bge_m3_main(args)
    if args.cmd == "encode-sentence-transformer":
        return encode_sentence_transformer_main(args)
    if args.cmd == "vendi":
        return vendi_main(args)
    if args.cmd == "geometry":
        return geometry_main(args)
    if args.cmd == "knn":
        return knn_main(args)
    if args.cmd == "support":
        return support_main(args)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
