#!/usr/bin/env python3
"""Build CC12M claim-extraction requests whose window is B lexical units.

The audit cuts the claim window at B whitespace-delimited words. This variant
cuts every caption after its B-th lexical unit (the regex token of the text
statistics) and leaves the builder, prompts, and schema unchanged, so the two
window definitions can be compared on the same 4,494 images. It is the run
behind results/sensitivity/cc12m_lexical_window_claimed_cbu_summary.json.

Input (``--requests``, resolved against ``--repo``): the B = 64 claimed-CBU
request file of the CC12M four-surface slice (README step 6), whose rows carry
``surface``, ``source_row``, and the full ``source_caption``. The default is the
same file that prepare_cc12m_bgrid_full.py reads.

Outputs under ``--work``::

    inputs/<surface>.lex<B>.jsonl                          one {"caption": window} row per image
    claimed_cbu_v2_<surface>_lex<B>_4494.requests.jsonl    requests per surface
    claimed_cbu_v2_cc12m_four_caption_lex<B>_4494.requests.jsonl
                                                           the four surfaces merged
    prepare_lex<B>_report.json                             request counts, mean words, mean lexical units

Each window goes through scripts/build_caption_cbu_requests.py without
--token-budget, so the whitespace cut is not applied a second time. Run the
merged file through run_text_json_requests.py and summarize_cbu_responses.py
--mode claimed as in README step 4.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
SURFACES = ("ours_cc12m", "ref_cc12m_qwen3vl8b", "ref_cc12m_llavanext", "ref_pixelprose_cc12m")
FULL_B64 = (
    "artifacts/cbu/cc12m-four-caption-llava-url-bridge-5k-local/"
    "claimed_cbu_v2_cc12m_four_caption_llava_url_bridge_b64_4494.requests.jsonl"
)
BUILDER = Path(__file__).resolve().parents[1] / "build_caption_cbu_requests.py"


def lexical_window(text: str, budget: int) -> str:
    matches = list(TOKEN_RE.finditer(text))
    if len(matches) <= budget:
        return text
    return text[: matches[budget - 1].end()].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".", help="working tree that holds artifacts/ (default: current directory)")
    parser.add_argument("--requests", default=FULL_B64,
                        help="B = 64 CC12M claimed-CBU request JSONL, relative to --repo unless absolute")
    parser.add_argument("--work", required=True, help="output directory")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    repo, work = Path(args.repo), Path(args.work)
    requests = Path(args.requests)
    if not requests.is_absolute():
        requests = repo / requests
    (work / "inputs").mkdir(parents=True, exist_ok=True)
    captions: dict[str, dict[int, str]] = defaultdict(dict)
    with requests.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            captions[row["surface"]][row["source_row"]] = row["source_caption"]
    merged = work / f"claimed_cbu_v2_cc12m_four_caption_lex{args.budget}_4494.requests.jsonl"
    report = {}
    with merged.open("w", encoding="utf-8") as out:
        for surface in SURFACES:
            rows = captions[surface]
            source = work / "inputs" / f"{surface}.lex{args.budget}.jsonl"
            words = lexical = 0
            with source.open("w", encoding="utf-8") as handle:
                for index in range(len(rows)):
                    window = lexical_window(rows[index], args.budget)
                    words += len(window.split())
                    lexical += len(TOKEN_RE.findall(window))
                    handle.write(json.dumps({"caption": window}, ensure_ascii=False) + "\n")
            built = work / f"claimed_cbu_v2_{surface}_lex{args.budget}_4494.requests.jsonl"
            subprocess.run([args.python, str(BUILDER), "--input", str(source),
                            "--output", str(built), "--surface", surface], check=True, stdout=subprocess.DEVNULL)
            n = 0
            for line in built.open(encoding="utf-8"):
                out.write(line)
                n += 1
            report[surface] = {"requests": n, "mean_words": words / len(rows), "mean_lexical_units": lexical / len(rows)}
    (work / f"prepare_lex{args.budget}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
