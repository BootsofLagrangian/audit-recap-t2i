#!/usr/bin/env python3
"""Per-category CBU-VQA summaries and a count/relation-excluded headline.

Re-reads the exact response JSONL files behind the paper's VQA summaries and
applies the counting rule of ``scripts/summarize_cbu_vqa_responses.py`` (latest
response per request_id; supported CBU/cap = yes / responses; risk = no /
questions). For every (slice, judge, surface) it reports

* all claim types (must reproduce the paper-facing summary exactly),
* the same cell after dropping count and relation claims, and
* per-claim-type yes / no / uncertain counts.

On CC12M, where both judges answered the same request file, it also reports
per-type exact Qwen--Gemma answer agreement joined on question_id.

The script only reads inputs; run it where the response JSONL files live.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ANSWERS = ("yes", "no", "uncertain")
EXCLUDED = frozenset({"count", "relation"})

QWEN = "Qwen3.5-397B-A17B-FP8"
GEMMA = "Gemma-4-31B-IT"
VQA = "artifacts/vqa-cbu"
GEMMA_XC = f"{VQA}/gemma-cross-corpus-2026-05-02/responses"

# (slice, judge, [response files merged latest-by-request], surface -> side)
SOURCES: list[tuple[str, str, list[str], dict[str, str]]] = [
    ("DataComp", QWEN, [f"{VQA}/pair5k-local/cbu_vqa_datacomp_b64_5k.responses.qwen397_full.jsonl"], {
        "datacomp_recap_llava15_paired_url__ours_datacomp_forward": "Ours",
        "datacomp_recap_llava15_paired_url__ref_datacomp_recap_llava15_llama3_8b": "Ref",
    }),
    ("LAION-pop", QWEN, [f"{VQA}/pair5k-local/cbu_vqa_noncc12m_nondc_b64_5k.responses.qwen397_full.jsonl"], {
        "laion_pop_llama32_paired__ours_laion_pop": "Ours",
        "laion_pop_llama32_paired__ref_laion_pop_llama32_11b": "Ref",
    }),
    ("PD12M", QWEN, [f"{VQA}/pair5k-local/cbu_vqa_noncc12m_nondc_b64_5k.responses.qwen397_full.jsonl"], {
        "pd12m_full_paired__ours_pd12m_img2dataset": "Ours",
        "pd12m_full_paired__ref_pd12m_full": "Ref",
    }),
    ("Danbooru", QWEN, [f"{VQA}/pair5k-local/cbu_vqa_noncc12m_nondc_b64_5k.responses.qwen397_full.jsonl"], {
        "danbooru2023_florence2_paired__ours_danbooru2023": "Ours",
        "danbooru2023_florence2_paired__ref_danbooru_florence2": "Ref",
    }),
    ("DataComp", GEMMA, [
        f"{GEMMA_XC}/cbu_vqa_ours_datacomp_forward_b64_5000.responses.gemma4_31b_file_mt2048.merged.jsonl",
        f"{GEMMA_XC}/cbu_vqa_ref_datacomp_recap_llava15_b64_5000.responses.gemma4_31b_file_mt2048.merged.jsonl",
    ], {"ours_datacomp_forward": "Ours", "ref_datacomp_recap_llava15_llama3_8b": "Ref"}),
    ("LAION-pop", GEMMA, [
        f"{GEMMA_XC}/cbu_vqa_laion_pop_llama32_paired__ours_laion_pop_b64_5k.responses.gemma4_31b_file_mt2048.merged.jsonl",
        f"{GEMMA_XC}/cbu_vqa_laion_pop_llama32_paired__ref_laion_pop_llama32_11b_b64_5k.responses.gemma4_31b_file_mt2048.merged.jsonl",
    ], {"laion_pop_llama32_paired__ours_laion_pop": "Ours", "laion_pop_llama32_paired__ref_laion_pop_llama32_11b": "Ref"}),
    ("PD12M", GEMMA, [
        f"{GEMMA_XC}/cbu_vqa_pd12m_full_paired__ours_pd12m_img2dataset_b64_5k.responses.gemma4_31b_file_mt2048.merged.jsonl",
        f"{GEMMA_XC}/cbu_vqa_pd12m_full_paired__ref_pd12m_full_b64_5k.responses.gemma4_31b_file_mt2048.merged.jsonl",
    ], {"pd12m_full_paired__ours_pd12m_img2dataset": "Ours", "pd12m_full_paired__ref_pd12m_full": "Ref"}),
    ("Danbooru", GEMMA, [
        f"{GEMMA_XC}/cbu_vqa_danbooru2023_florence2_paired__ours_danbooru2023_b64_5k.responses.gemma4_31b_file_mt2048.merged.jsonl",
        f"{GEMMA_XC}/cbu_vqa_danbooru2023_florence2_paired__ref_danbooru_florence2_b64_5k.responses.gemma4_31b_file_mt2048.merged.jsonl",
    ], {"danbooru2023_florence2_paired__ours_danbooru2023": "Ours", "danbooru2023_florence2_paired__ref_danbooru_florence2": "Ref"}),
    ("CC12M", QWEN, [
        f"{VQA}/cc12m-four-caption-llava-url-bridge-5k-local/"
        "cbu_vqa_cc12m_four_caption_llava_url_bridge_b64_4494.responses.qwen397_image_local_c512_mt2048_compact.jsonl",
    ], {
        "ours_cc12m": "Ours", "ref_cc12m_llavanext": "LLaVA-NeXT",
        "ref_pixelprose_cc12m": "PixelProse", "ref_cc12m_qwen3vl8b": "Qwen3-VL-8B",
    }),
    ("CC12M", GEMMA, [
        f"{VQA}/cc12m-four-caption-llava-url-bridge-5k-local-crossjudge-gemma4/"
        "cbu_vqa_cc12m_four_caption_b64_4494.responses.gemma4_31b_it_c512_mt2048.jsonl",
        f"{VQA}/cc12m-four-caption-llava-url-bridge-5k-local-crossjudge-gemma4/"
        "cbu_vqa_cc12m_four_caption_b64_500img_2000req.responses.gemma4_31b_it_c512_mt2048.jsonl",
    ], {
        "ours_cc12m": "Ours", "ref_cc12m_llavanext": "LLaVA-NeXT",
        "ref_pixelprose_cc12m": "PixelProse", "ref_cc12m_qwen3vl8b": "Qwen3-VL-8B",
    }),
]


CC12M_EXTRACTION = (
    "artifacts/cbu/cc12m-four-caption-llava-url-bridge-5k-local/"
    "claimed_cbu_v2_cc12m_four_caption_llava_url_bridge_b64_4494.responses.qwen397_c1024_mt4096.jsonl"
)


def cc12m_extraction(root: Path) -> dict[str, dict[str, int]]:
    """Per-surface extraction accounting on the 4,494-image CC12M slice.

    A response is valid when it parsed against the schema; an invalid one is
    counted as truncated when it used the whole completion budget.
    """
    out: dict[str, Counter] = defaultdict(Counter)
    for row in latest_rows([root / CC12M_EXTRACTION]):
        surface = row["request"]["surface"]
        out[surface]["requests"] += 1
        valid = row.get("ok") and row.get("parsed") is not None and not row.get("parse_error") and not row.get("schema_error")
        if not valid:
            out[surface]["invalid"] += 1
            if (row.get("usage") or {}).get("completion_tokens") == row["request"].get("max_tokens", 4096):
                out[surface]["invalid_at_token_cap"] += 1
            continue
        out[surface]["valid"] += 1
        if not row["parsed"].get("claimed_units"):
            out[surface]["valid_zero_claims"] += 1
    return {s: dict(c) for s, c in sorted(out.items())}


def latest_rows(paths: list[Path]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row.get("request_id"), str):
                        latest[row["request_id"]] = row
    return list(latest.values())


def answered(row: dict[str, Any]):
    """Yield (question_id, category, answer) using the summarizer's rules."""
    lookup = {
        q["question_id"]: q
        for q in row.get("request", {}).get("questions", [])
        if isinstance(q, dict) and isinstance(q.get("question_id"), str)
    }
    for result in (row.get("parsed") or {}).get("question_results", []):
        if isinstance(result, dict) and result.get("answer") in ANSWERS:
            qid = result.get("question_id")
            yield qid, lookup.get(qid, {}).get("category", "__unknown__"), result["answer"]


def cell(counter: Counter, responses: int) -> dict[str, Any]:
    questions = sum(counter[a] for a in ANSWERS)
    return {
        "responses": responses,
        "questions": questions,
        **{a: counter[a] for a in ANSWERS},
        "supported_cap": counter["yes"] / responses if responses else 0.0,
        "risk": counter["no"] / questions if questions else 0.0,
        "uncertain_rate": counter["uncertain"] / questions if questions else 0.0,
    }


def summarize(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"cells": [], "cc12m_judge_agreement": {}}
    cc12m_answers: dict[str, dict[str, tuple[str, str]]] = {}
    for slice_name, judge, files, surfaces in SOURCES:
        rows = latest_rows([root / f for f in files])
        responses: Counter = Counter()
        all_types: dict[str, Counter] = defaultdict(Counter)
        kept: dict[str, Counter] = defaultdict(Counter)
        by_type: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        answers: dict[str, tuple[str, str]] = {}
        for row in rows:
            surface = row.get("request", {}).get("surface")
            if surface not in surfaces:
                continue
            side = surfaces[surface]
            responses[side] += 1
            if not row.get("ok"):
                continue
            for qid, category, answer in answered(row):
                all_types[side][answer] += 1
                by_type[side][category][answer] += 1
                if category not in EXCLUDED:
                    kept[side][answer] += 1
                if slice_name == "CC12M":
                    answers[qid] = (category, answer)
        for side in surfaces.values():
            out["cells"].append({
                "slice": slice_name, "judge": judge, "surface": side,
                "all_types": cell(all_types[side], responses[side]),
                "excluding_count_relation": cell(kept[side], responses[side]),
                "by_type": {c: {a: n[a] for a in ANSWERS} for c, n in sorted(by_type[side].items())},
            })
        if slice_name == "CC12M":
            cc12m_answers[judge] = answers

    if QWEN in cc12m_answers and GEMMA in cc12m_answers:
        agree: dict[str, Counter] = defaultdict(Counter)
        qwen, gemma = cc12m_answers[QWEN], cc12m_answers[GEMMA]
        for qid in qwen.keys() & gemma.keys():
            category = qwen[qid][0]
            for key in (category, "__all__"):
                agree[key]["joined"] += 1
                agree[key]["exact"] += qwen[qid][1] == gemma[qid][1]
        out["cc12m_judge_agreement"] = {
            c: {**dict(n), "exact_rate": n["exact"] / n["joined"]} for c, n in sorted(agree.items())
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="Repository root holding artifacts/vqa-cbu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(Path(args.root))
    result["cc12m_extraction"] = cc12m_extraction(Path(args.root))
    for surface, n in result["cc12m_extraction"].items():
        print("extraction", surface, n)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for c in result["cells"]:
        a, k = c["all_types"], c["excluding_count_relation"]
        print(f'{c["slice"]:9s} {c["judge"][:5]} {c["surface"]:12s} n={a["responses"]:5d} '
              f'sup/cap {a["supported_cap"]:.2f} risk {a["risk"]:.3f} | '
              f'excl sup/cap {k["supported_cap"]:.2f} risk {k["risk"]:.3f}')
    for c, n in result["cc12m_judge_agreement"].items():
        print(f'agree {c:15s} {n["exact"]}/{n["joined"]} = {n["exact_rate"]:.3f}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
