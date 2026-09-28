#!/usr/bin/env python3
"""Build cc12m_budget_frontier_plot.csv from claimed-CBU summary JSON files.

Each ``--summary B=PATH`` names the ``scripts/summarize_cbu_responses.py
--mode claimed`` output for budget B. Columns match the CSV behind Figure 2
(right); ``pareto_efficiency_yield`` marks surfaces that no other surface at the
same budget matches or beats on both CBU per caption and CBU per 100 lex.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SURFACES = [
    ("ours_cc12m", "Ours"),
    ("ref_cc12m_llavanext", "LLaVA-NeXT"),
    ("ref_cc12m_qwen3vl8b", "Qwen3-VL-8B"),
    ("ref_pixelprose_cc12m", "PixelProse"),
]
CATEGORIES = ["object", "attribute", "relation", "style", "camera", "lighting", "count", "text_rendering"]
FIELDS = ["budget", "surface", "label", "valid", "bad_json", "cbu_per_cap", "cbu_per_100tok", "dup_row_rate",
          *[f"{c}_per_cap" for c in CATEGORIES], "pareto_efficiency_yield"]


def rows_for(budget: int, summary: dict) -> list[dict]:
    rows = []
    for key, label in SURFACES:
        s = summary["surfaces"][key]
        rows.append({
            "budget": budget, "surface": key, "label": label, "valid": s["captions"],
            "bad_json": summary["status"].get("bad", 0),
            "cbu_per_cap": s["claimed_dedup_per_caption"],
            "cbu_per_100tok": s["claimed_dedup_per_100_tokens"],
            "dup_row_rate": s["duplicate_row_rate"],
            **{f"{c}_per_cap": s[f"claimed_dedup_{c}_per_caption"] for c in CATEGORIES},
        })
    for row in rows:
        dominated = any(
            o is not row
            and o["cbu_per_cap"] >= row["cbu_per_cap"] and o["cbu_per_100tok"] >= row["cbu_per_100tok"]
            and (o["cbu_per_cap"] > row["cbu_per_cap"] or o["cbu_per_100tok"] > row["cbu_per_100tok"])
            for o in rows
        )
        row["pareto_efficiency_yield"] = 0 if dominated else 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", action="append", required=True, help="B=path/to/summary.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = []
    for item in args.summary:
        budget, path = item.split("=", 1)
        out.extend(rows_for(int(budget), json.loads(Path(path).read_text())))
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {len(out)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
