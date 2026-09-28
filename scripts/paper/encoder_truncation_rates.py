#!/usr/bin/env python3
"""Encoder-token truncation rates for caption JSONL surfaces.

Counts every non-empty ``caption`` of each surface with the tokenizers of
CLIP-77, LongCLIP-248, and SigLIP2-64, untruncated and with special tokens
(the counting of caption_tokenizer_truncation_survey.py), and reports the share
of captions over each encoder's token cap, together with the mean length in
lexical units and whitespace words. It is the script behind
results/sensitivity/naive_encoder_truncation.json, run with

    --surface naive_qwen35_cc12m_greedy=results/naive_qwen35_cc12m/captions.jsonl.gz
    --surface naive_qwen35_sampled_cc12m=results/naive_qwen35_sampled_cc12m/captions.jsonl.gz
    --surface naive_qwen35_sampled_datacomp=results/naive_qwen35_sampled_datacomp/captions.jsonl.gz

on the unmasked captions. The released captions mask personal data from image
text ([email], [phone], [name], [id], [address]), so a rerun on them can differ
from the published file in the last digits. Inputs may be plain or gzip JSONL.
Needs the ``eval`` extra (Transformers) and network access to the three
tokenizers on the Hugging Face Hub.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from pathlib import Path

from transformers import AutoTokenizer

ENCODERS = [
    ("clip77", "openai/clip-vit-large-patch14", 77),
    ("longclip248", "zer0int/LongCLIP-GmP-ViT-L-14", 248),
    ("siglip2_64", "google/siglip2-so400m-patch14-384", 64),
]
LEX = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


def open_jsonl(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--surface", action="append", required=True, metavar="LABEL=JSONL",
                        help="surface label and caption JSONL (.jsonl or .jsonl.gz); repeat per surface")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizers = []
    for name, model_id, cap in ENCODERS:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tok.model_max_length = 1_000_000
        tokenizers.append((name, model_id, cap, tok))
    out = {}
    for spec in args.surface:
        label, path = spec.split("=", 1)
        captions = []
        with open_jsonl(path) as handle:
            for line in handle:
                row = json.loads(line)
                caption = row.get("caption")
                if isinstance(caption, str) and caption.strip():
                    captions.append(caption)
        lex = [len(LEX.findall(c.lower())) for c in captions]
        words = [len(c.split()) for c in captions]
        entry = {"records": len(captions), "mean_lex": statistics.fmean(lex), "mean_words": statistics.fmean(words),
                 "lex_overflow_gt_248": sum(v > 248 for v in lex) / len(lex), "encoders": {}}
        for name, model_id, cap, tok in tokenizers:
            lengths = []
            for start in range(0, len(captions), 1024):
                enc = tok(captions[start:start + 1024], add_special_tokens=True, padding=False, truncation=False)
                lengths.extend(len(ids) for ids in enc["input_ids"])
            entry["encoders"][name] = {"model_id": model_id, "max_len": cap, "mean_tokens": statistics.fmean(lengths),
                                       "truncated_rate_gt_limit": sum(v > cap for v in lengths) / len(lengths)}
        out[label] = entry
        print(label, json.dumps(entry)[:600])
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
