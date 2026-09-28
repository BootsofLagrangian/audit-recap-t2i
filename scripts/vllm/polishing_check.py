#!/usr/bin/env python3
"""
Check generated captions against the polishing taxonomy (A-N categories).
Regex-based prefilter for all violation types.

Usage:
    # Check a JSONL file of captions
    uv run python scripts/vllm/polishing_check.py --input captions.jsonl

    # Inline check (pipe from bench or recap)
    echo '{"text":"The image shows a girl..."}' | uv run python scripts/vllm/polishing_check.py --stdin
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ── Polishing Taxonomy Violations (A-N) ──────────────────

VIOLATIONS = {
    "A_prose_coord_echo": {
        "desc": "YOLO bbox coordinates in prose body",
        "severity": "critical",
        "patterns": [
            r"\[\s*\d{2,4}\s*,\s*\d{2,4}\s*,\s*\d{2,4}\s*,\s*\d{2,4}\s*\]",
        ],
    },
    "B_text_coord_echo": {
        "desc": "Coordinates in TEXT: OCR section",
        "severity": "moderate",
        "patterns": [
            r"TEXT:.*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]",
        ],
    },
    "C_bbox_tag": {
        "desc": "XML-style bbox tags",
        "severity": "moderate",
        "patterns": [
            r"<bbox>",
            r"</bbox>",
            r"<ref>",
            r"</ref>",
        ],
    },
    "D_markdown_leak": {
        "desc": "Markdown syntax in prose",
        "severity": "moderate",
        "patterns": [
            r"\*\*[^*]+\*\*",          # **bold**
            r"^[-*]\s",                  # bullet lists
            r"^#+\s",                    # headers
            r"\[([^\]]+)\]\([^\)]+\)",   # [link](url)
        ],
    },
    "E_thinking_leak": {
        "desc": "Model reasoning / thinking exposed",
        "severity": "critical",
        "patterns": [
            r"<think>",
            r"</think>",
            r"(?i)the user wants",
            r"(?i)let me ",
            r"(?i)I need to ",
            r"(?i)I should ",
            r"(?i)I will ",
            r"(?i)I'll describe",
        ],
    },
    "F_repetition": {
        "desc": "Excessive n-gram repetition",
        "severity": "severe",
        "patterns": [],  # handled by custom logic
    },
    "G_meta_instruction": {
        "desc": "Prompt/instruction references",
        "severity": "low",
        "patterns": [
            r"(?i)per (the )?instructions",
            r"(?i)as (instructed|requested|prompted)",
            r"(?i)following the (prompt|guidelines)",
            r"(?i)requirements?:",
        ],
    },
    "H_detection_meta_leak": {
        "desc": "Detection box language in prose",
        "severity": "moderate",
        "patterns": [
            r"(?i)bounding box",
            r"(?i)detection (box|region|area)",
            r"(?i)confidence\s*[:=]\s*\d",
            r"(?i)detected at",
        ],
    },
    "I_overlay_meta_leak": {
        "desc": "YOLO visual overlay descriptions",
        "severity": "moderate",
        "patterns": [
            r"(?i)overlay",
            r"(?i)annotation layer",
            r"(?i)green rectangle",
            r"(?i)labeled box",
        ],
    },
    "J_meta_statement": {
        "desc": "Meta-framing (the image shows...)",
        "severity": "moderate",
        "patterns": [
            r"(?i)^the (image|picture|photo|illustration|artwork|scene) (shows?|depicts?|features?|presents?|displays?|contains?|captures?)",
            r"(?i)^this (image|picture|photo|illustration|artwork|scene) (shows?|depicts?|features?|presents?|displays?|contains?|captures?)",
            r"(?i)^in this (image|picture|photo|illustration)",
            r"(?i)^we (can )?see ",
        ],
    },
    "K_qwen_bbox_echo": {
        "desc": "Qwen VL bbox_2d JSON structures",
        "severity": "moderate",
        "patterns": [
            r"bbox_2d",
            r'"type"\s*:\s*"bbox"',
        ],
    },
    "L_tag_verbatim": {
        "desc": "Raw underscore tags in prose",
        "severity": "low",
        "patterns": [
            r"\b\w+_\w+_\w+\b",  # 3+ part underscore tags like blue_hair_ribbon
        ],
    },
    "M_numbered_list_leak": {
        "desc": "Numbered lists instead of prose",
        "severity": "low",
        "patterns": [
            r"^\d+\.\s+\w",  # "1. Something"
        ],
    },
    "N_stage_header_leak": {
        "desc": "Pipeline stage headers, resolution info",
        "severity": "critical",
        "patterns": [
            r"(?i)stage\s*[1-4]",
            r"(?i)resolution\s*:\s*\d+",
            r"(?i)caption (type|format|style)\s*:",
        ],
    },
}


def check_repetition(text: str, n: int = 5, threshold: int = 10) -> bool:
    """Check for 5-gram repeated 10+ times."""
    words = text.lower().split()
    if len(words) < n:
        return False
    ngrams: dict[tuple, int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
    return any(v >= threshold for v in ngrams.values())


def classify_caption(text: str) -> list[dict]:
    """Return list of violations found in the caption."""
    violations = []
    for code, spec in VIOLATIONS.items():
        if code == "F_repetition":
            if check_repetition(text):
                violations.append({"code": code, "severity": spec["severity"],
                                   "desc": spec["desc"], "match": "(5-gram ≥10)"})
            continue
        for pat in spec["patterns"]:
            m = re.search(pat, text, re.MULTILINE)
            if m:
                violations.append({
                    "code": code, "severity": spec["severity"],
                    "desc": spec["desc"], "match": m.group()[:80],
                })
                break  # one match per category is enough
    return violations


def analyze_captions(captions: list[dict]) -> dict:
    """Analyze a batch of captions. Each dict must have 'text' field."""
    total = len(captions)
    violated = 0
    by_code: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    examples: list[dict] = []

    for cap in captions:
        text = cap.get("text", "")
        vs = classify_caption(text)
        if vs:
            violated += 1
            for v in vs:
                by_code[v["code"]] = by_code.get(v["code"], 0) + 1
                by_severity[v["severity"]] = by_severity.get(v["severity"], 0) + 1
            if len(examples) < 20:
                examples.append({"text": text[:200], "violations": vs})

    return {
        "total": total,
        "violated": violated,
        "violation_rate": f"{violated/total*100:.1f}%" if total else "N/A",
        "by_code": dict(sorted(by_code.items(), key=lambda x: -x[1])),
        "by_severity": by_severity,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Polishing taxonomy violation checker")
    parser.add_argument("--input", type=str, help="JSONL file with 'text' field per line")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--field", default="text", help="JSON field containing caption text")
    args = parser.parse_args()

    captions = []
    if args.stdin:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                captions.append(json.loads(line))
            except json.JSONDecodeError:
                captions.append({"text": line})
    elif args.input:
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        captions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    else:
        print("Provide --input or --stdin")
        return

    # Normalize field name
    for cap in captions:
        if args.field != "text" and args.field in cap:
            cap["text"] = cap[args.field]

    result = analyze_captions(captions)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
