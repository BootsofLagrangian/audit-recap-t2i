#!/usr/bin/env python3
"""Export compact tables from CBU-VQA summary JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True, help="CBU-VQA summary JSON")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-tex", default=None)
    return parser.parse_args()


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for surface, stats in sorted(data.get("surfaces", {}).items()):
            rows.append(
                {
                    "source": path,
                    "surface": surface,
                    "responses": stats.get("responses", 0),
                    "ok": stats.get("ok", 0),
                    "questions": stats.get("questions", 0),
                    "support": stats.get("support_rate", 0.0),
                    "risk": stats.get("risk_rate", 0.0),
                    "uncertain": stats.get("uncertainty_rate", 0.0),
                }
            )
    return rows


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| Surface | Resp | OK | Q | Support ↑ | Risk ↓ | Uncertain ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {surface} | {responses:,} | {ok:,} | {questions:,} | "
            "{support:.4f} | {risk:.4f} | {uncertain:.4f} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Surface & Resp. & OK & Q & Support $\uparrow$ & Risk $\downarrow$ & Uncertain $\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            (
                "{surface} & {responses:,} & {ok:,} & {questions:,} & "
                "{support:.4f} & {risk:.4f} & {uncertain:.4f} \\\\"
            )
            .format(**row)
            .replace("_", r"\_")
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.summary)
    write_markdown(rows, Path(args.output_md))
    if args.output_tex:
        write_latex(rows, Path(args.output_tex))
    print(json.dumps({"rows": len(rows), "output_md": args.output_md, "output_tex": args.output_tex}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
