#!/usr/bin/env python3
"""Plot budget-curve metrics from caption survey JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("coverage_rate", "Budget Eligibility@B", "up"),
    ("distinct_n.2", "Distinct-2@B", "up"),
    ("distinct_n.3", "Distinct-3@B", "up"),
    ("ngram_top_k_mass.2", "Top-100 Bigram Mass@B", "down"),
    ("ngram_top_k_mass.3", "Top-100 Trigram Mass@B", "down"),
    ("violation_rate", "Violation Rate@B", "down"),
    ("repeated_4gram_rate", "Repeated 4-gram Rate@B", "down"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot caption survey budget curves")
    parser.add_argument("--input", action="append", required=True, help="Survey JSON path (repeatable)")
    parser.add_argument("--output-dir", required=True, help="Directory for output PNG plots")
    parser.add_argument(
        "--long-coverage-threshold",
        type=float,
        default=0.5,
        help="budget-eligibility@64 threshold used to split long vs short regimes",
    )
    return parser.parse_args()


def nested_get(mapping: dict[str, Any], path: str) -> float | None:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return float(current) if isinstance(current, (int, float)) else None


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        if "results" in payload:
            for item in payload.get("results", []):
                summary = item.get("summary") or item.get("survey_summary")
                if not isinstance(summary, dict):
                    continue
                entry = item.get("entry") or {}
                length_controlled = summary.get("length_controlled") or {}
                budgets = sorted(int(key) for key in length_controlled.keys())
                if not budgets:
                    continue
                cov64 = nested_get(length_controlled.get("64", {}), "coverage_rate") or 0.0
                full = summary.get("full_length_reference") or {}
                avg_tokens = full.get("avg_tokens", full.get("avg_lexical_tokens", 0.0))
                rows.append(
                    {
                        "name": entry.get("name", Path(raw_path).stem),
                        "family": entry.get("source_family", "unknown"),
                        "group": entry.get("group", "unknown"),
                        "description": entry.get("description", ""),
                        "captioner": entry.get("captioner", ""),
                        "avg_tokens": float(avg_tokens),
                        "coverage64": float(cov64),
                        "budgets": budgets,
                        "length_controlled": length_controlled,
                    }
                )
            continue

        if "length_controlled" in payload:
            length_controlled = payload.get("length_controlled") or {}
            budgets = sorted(int(key) for key in length_controlled.keys())
            if not budgets:
                continue
            cov64 = nested_get(length_controlled.get("64", {}), "coverage_rate") or 0.0
            full = payload.get("full_length_reference") or {}
            avg_tokens = full.get("avg_tokens", full.get("avg_lexical_tokens", 0.0))
            stem = Path(raw_path).stem
            name = stem.removesuffix("_1m").removesuffix("_50k")
            family = "unknown"
            if "datacomp" in name:
                family = "datacomp"
            elif "pd12m" in name:
                family = "pd12m"
            rows.append(
                {
                    "name": name,
                    "family": family,
                    "group": "direct_summary",
                    "description": "",
                    "captioner": "",
                    "avg_tokens": float(avg_tokens),
                    "coverage64": float(cov64),
                    "budgets": budgets,
                    "length_controlled": length_controlled,
                }
            )
    return rows


def label_for_row(row: dict[str, Any]) -> str:
    name = row["name"]
    if name.startswith("ours_"):
        label = f"ours:{name.removeprefix('ours_')}"
    elif name.startswith("ref_"):
        label = f"ref:{name.removeprefix('ref_')}"
    else:
        label = name
    if name == "ref_cc12m_qwen3vl8b":
        label += "†"
    return label


def decorate_metric_label(metric_label: str, direction: str) -> str:
    arrow = "↑" if direction == "up" else "↓"
    return f"{metric_label} {arrow}"


def style_for_row(row: dict[str, Any]) -> dict[str, Any]:
    if row["name"].startswith("ours_"):
        return {"linewidth": 2.8, "alpha": 0.95, "linestyle": "-"}
    return {"linewidth": 1.6, "alpha": 0.85, "linestyle": "--"}


def series_for_metric(row: dict[str, Any], metric_key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for budget in row["budgets"]:
        summary = row["length_controlled"].get(str(budget), {})
        value = nested_get(summary, metric_key)
        if value is None:
            continue
        xs.append(budget)
        ys.append(value)
    return xs, ys


def save_metric_plot(
    rows: list[dict[str, Any]],
    metric_key: str,
    metric_label: str,
    direction: str,
    regime_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for row in sorted(rows, key=lambda item: (item["family"], item["name"])):
        xs, ys = series_for_metric(row, metric_key)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", label=label_for_row(row), **style_for_row(row))

    decorated_label = decorate_metric_label(metric_label, direction)
    ax.set_title(f"{decorated_label} by Budget ({regime_name})")
    ax.set_xlabel("Token Budget")
    ax.set_ylabel(decorated_label)
    ax.set_xticks(sorted({budget for row in rows for budget in row["budgets"]}))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_family_plot(rows: list[dict[str, Any]], family: str, output_path: Path) -> None:
    family_rows = [row for row in rows if row["family"] == family]
    if not family_rows:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes = axes.flatten()
    for axis, (metric_key, metric_label, direction) in zip(axes, METRICS[:6], strict=False):
        for row in sorted(family_rows, key=lambda item: item["name"]):
            xs, ys = series_for_metric(row, metric_key)
            if not xs:
                continue
            axis.plot(xs, ys, marker="o", label=label_for_row(row), **style_for_row(row))
        axis.set_title(decorate_metric_label(metric_label, direction))
        axis.set_xlabel("Budget")
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle(f"{family} Budget Curves", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise SystemExit("No survey rows loaded")

    output_dir = Path(args.output_dir)
    long_rows = [row for row in rows if row["coverage64"] >= args.long_coverage_threshold]
    short_rows = [row for row in rows if row["coverage64"] < args.long_coverage_threshold]

    for metric_key, metric_label, direction in METRICS:
        if long_rows:
            save_metric_plot(
                long_rows,
                metric_key,
                metric_label,
                direction,
                "long-regime",
                output_dir / "overview" / "long" / f"{metric_key.replace('.', '_')}.png",
            )
        if short_rows:
            save_metric_plot(
                short_rows,
                metric_key,
                metric_label,
                direction,
                "short-regime",
                output_dir / "overview" / "short" / f"{metric_key.replace('.', '_')}.png",
            )

    for family in sorted({row["family"] for row in rows}):
        save_family_plot(rows, family, output_dir / "families" / f"{family}.png")

    manifest = {
        "inputs": args.input,
        "output_dir": str(output_dir),
        "long_coverage_threshold": args.long_coverage_threshold,
        "rows_loaded": len(rows),
        "long_rows": [row["name"] for row in long_rows],
        "short_rows": [row["name"] for row in short_rows],
        "metrics": [metric_key for metric_key, _, _ in METRICS],
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
