#!/usr/bin/env python3
"""Rebuild CC12M CBU@B requests for all 4,494 aligned images.

The 1,000-image budget pilot (``cbu/cc12m-four-caption-llava-url-bridge-bgrid-1k``)
was built by ``scripts/build_caption_cbu_requests.py`` from per-surface caption
JSONL files that no longer exist on node-local NVMe. The full-slice B=64
request file keeps every caption verbatim as ``source_caption`` together with
its ``source_row``, so this script

1. writes one caption JSONL per surface whose line index equals ``source_row``;
2. runs the unchanged builder for every surface and budget;
3. checks that the rebuilt first 1,000 rows equal the pilot requests record for
   record (request_id, captions, prompts), which pins the protocol; and
4. writes the rows the pilot did not cover into one run file per budget.

Inputs are only read; everything is written under ``--work``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SURFACES = ("ours_cc12m", "ref_cc12m_qwen3vl8b", "ref_cc12m_llavanext", "ref_pixelprose_cc12m")
BUDGETS = (16, 32, 48)
FULL_B64 = (
    "artifacts/cbu/cc12m-four-caption-llava-url-bridge-5k-local/"
    "claimed_cbu_v2_cc12m_four_caption_llava_url_bridge_b64_4494.requests.jsonl"
)
PILOT = "artifacts/cbu/cc12m-four-caption-llava-url-bridge-bgrid-1k"
PILOT_ROWS = 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="repository checkout holding artifacts/ and scripts/")
    parser.add_argument("--work", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    repo, work = Path(args.repo), Path(args.work)
    (work / "inputs").mkdir(parents=True, exist_ok=True)
    (work / "requests").mkdir(exist_ok=True)

    captions: dict[str, dict[int, str]] = defaultdict(dict)
    with (repo / FULL_B64).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            captions[row["surface"]][row["source_row"]] = row["source_caption"]
    for surface in SURFACES:
        rows = captions[surface]
        assert sorted(rows) == list(range(len(rows))), f"{surface}: source_row is not 0..n-1"
        with (work / "inputs" / f"{surface}.jsonl").open("w", encoding="utf-8") as out:
            for index in range(len(rows)):
                out.write(json.dumps({"caption": rows[index]}, ensure_ascii=False) + "\n")

    report: dict[str, object] = {"surfaces": {}, "pilot_match": {}, "run_files": {}}
    for budget in BUDGETS:
        run_rows = []
        for surface in SURFACES:
            built = work / "requests" / f"claimed_cbu_v2_{surface}_b{budget}_4494.requests.jsonl"
            subprocess.run([
                args.python, str(repo / "scripts/build_caption_cbu_requests.py"),
                "--input", str(work / "inputs" / f"{surface}.jsonl"),
                "--output", str(built), "--surface", surface, "--token-budget", str(budget),
            ], check=True, stdout=subprocess.DEVNULL)
            rebuilt = [json.loads(line) for line in built.open(encoding="utf-8")]
            pilot_path = repo / PILOT / f"claimed_cbu_v2_{surface}_b{budget}_1k.requests.jsonl"
            pilot = [json.loads(line) for line in pilot_path.open(encoding="utf-8")]
            identical = len(pilot) == PILOT_ROWS and rebuilt[:PILOT_ROWS] == pilot
            report["pilot_match"][f"{surface}:b{budget}"] = identical
            if not identical:
                raise SystemExit(f"rebuilt requests differ from the pilot for {surface} b{budget}")
            report["surfaces"][surface] = len(rebuilt)
            run_rows.extend(rebuilt[PILOT_ROWS:])
        run_file = work / f"claimed_cbu_v2_cc12m_four_caption_b{budget}_rows1000-4493.requests.jsonl"
        with run_file.open("w", encoding="utf-8") as out:
            for row in run_rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["run_files"][f"b{budget}"] = {"path": str(run_file), "requests": len(run_rows)}
    (work / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
