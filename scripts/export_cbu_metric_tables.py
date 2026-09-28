#!/usr/bin/env python3
"""Export paper-facing CBU tables with caption-level bootstrap CIs.

The script consumes existing CBU response JSONL artifacts. It does not call a
model and does not modify source captions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

UNIT_CATEGORIES = [
    "object",
    "attribute",
    "relation",
    "style",
    "camera",
    "lighting",
    "count",
    "text_rendering",
]

VISUAL_STATUSES = {"grounded", "unsupported", "uncertain"}
TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
ARTICLE_UNITS = {"a", "an", "the"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claimed", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--grounded", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    return label, Path(path)


def normalize_unit(text: str) -> str:
    tokens = TOKEN_RE.findall(text.lower())
    while tokens and tokens[0] in ARTICLE_UNITS:
        tokens.pop(0)
    return " ".join(tokens)


def normalize_key_part(text: str) -> str:
    return normalize_unit(text) or ""


def unit_records(group: Any) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if not isinstance(group, list):
        return records
    for item in group:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        unit = item.get("unit")
        if category not in UNIT_CATEGORIES or not isinstance(unit, str) or not unit.strip():
            continue
        target = item.get("target", "")
        records.append(
            {
                "category": category,
                "unit": unit.strip(),
                "target": target.strip() if isinstance(target, str) else "",
            }
        )
    return records


def dedup_counts(group: Any) -> tuple[int, dict[str, int], int]:
    counts = {category: 0 for category in UNIT_CATEGORIES}
    seen: set[str] = set()
    duplicate = 0
    for record in unit_records(group):
        norm = normalize_unit(record["unit"])
        if not norm:
            continue
        key = f"{record['category']}|{norm}|{normalize_key_part(record.get('target', ''))}"
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        counts[record["category"]] += 1
    return sum(counts.values()), counts, duplicate


def caption_tokens(request: dict[str, Any]) -> int:
    caption = request.get("caption", "")
    return len(TOKEN_RE.findall(caption)) if isinstance(caption, str) else 0


def read_claimed(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if not raw.get("ok") or not isinstance(raw.get("parsed"), dict):
                continue
            total, counts, duplicate = dedup_counts(raw["parsed"].get("claimed_units"))
            request = raw.get("request", {})
            rows.append(
                {
                    "label": label,
                    "caption_id": request.get("caption_id"),
                    "tokens": caption_tokens(request),
                    "dedup_units": total,
                    "duplicate_units": duplicate,
                    **{f"{category}_units": counts[category] for category in UNIT_CATEGORIES},
                }
            )
    return rows


def request_unit_lookup(request: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        unit.get("unit_id"): unit
        for unit in request.get("claimed_units", [])
        if isinstance(unit, dict) and isinstance(unit.get("unit_id"), str)
    }


def read_grounded(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if not raw.get("ok") or not isinstance(raw.get("parsed"), dict):
                continue
            lookup = request_unit_lookup(raw.get("request", {}))
            counter: Counter[str] = Counter()
            for result in raw["parsed"].get("unit_results", []):
                if not isinstance(result, dict):
                    continue
                unit = lookup.get(result.get("unit_id"), {})
                category = unit.get("category", "__unknown__")
                status = result.get("status", "__bad_status__")
                counter["valid"] += 1
                counter[status] += 1
                if status in VISUAL_STATUSES:
                    counter["visual"] += 1
                    if category in UNIT_CATEGORIES:
                        counter[f"{category}_visual"] += 1
                        counter[f"{category}_{status}"] += 1
            rows.append(
                {
                    "label": label,
                    "caption_id": raw.get("request", {}).get("caption_id"),
                    "valid": counter["valid"],
                    "visual": counter["visual"],
                    "grounded": counter["grounded"],
                    "unsupported": counter["unsupported"],
                    "uncertain": counter["uncertain"],
                    **{key: counter[key] for key in counter if "_" in key},
                }
            )
    return rows


def ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_indices(n: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(reps, n), endpoint=False)


def summarize_claimed(rows: list[dict[str, Any]], reps: int, rng: np.random.Generator) -> dict[str, Any]:
    n = len(rows)
    units = np.asarray([row["dedup_units"] for row in rows], dtype=np.float64)
    tokens = np.asarray([max(row["tokens"], 1) for row in rows], dtype=np.float64)
    dups = np.asarray([row["duplicate_units"] for row in rows], dtype=np.float64)
    idx = bootstrap_indices(n, reps, rng) if n else np.empty((0, 0), dtype=np.int64)

    def mean_metric(arr: np.ndarray) -> dict[str, float]:
        point = float(arr.mean()) if len(arr) else 0.0
        boot = arr[idx].mean(axis=1) if len(arr) else np.asarray([0.0])
        low, high = ci(boot)
        return {"mean": point, "ci95_low": low, "ci95_high": high}

    ratio = float(100.0 * units.sum() / tokens.sum()) if tokens.sum() else 0.0
    ratio_boot = 100.0 * units[idx].sum(axis=1) / tokens[idx].sum(axis=1) if n else np.asarray([0.0])
    low, high = ci(ratio_boot)
    out: dict[str, Any] = {
        "captions": n,
        "dedup_units_per_caption": mean_metric(units),
        "dedup_units_per_100_tokens": {"mean": ratio, "ci95_low": low, "ci95_high": high},
        "duplicate_units_per_caption": mean_metric(dups),
    }
    for category in UNIT_CATEGORIES:
        arr = np.asarray([row[f"{category}_units"] for row in rows], dtype=np.float64)
        out[f"{category}_per_caption"] = mean_metric(arr)
    return out


def summarize_grounded(rows: list[dict[str, Any]], reps: int, rng: np.random.Generator) -> dict[str, Any]:
    n = len(rows)
    grounded = np.asarray([row["grounded"] for row in rows], dtype=np.float64)
    unsupported = np.asarray([row["unsupported"] for row in rows], dtype=np.float64)
    uncertain = np.asarray([row["uncertain"] for row in rows], dtype=np.float64)
    visual = np.asarray([max(row["visual"], 0) for row in rows], dtype=np.float64)
    idx = bootstrap_indices(n, reps, rng) if n else np.empty((0, 0), dtype=np.int64)

    def ratio_metric(num: np.ndarray, den: np.ndarray) -> dict[str, float]:
        point = float(num.sum() / den.sum()) if den.sum() else 0.0
        if not n:
            return {"mean": point, "ci95_low": point, "ci95_high": point}
        boot_den = den[idx].sum(axis=1)
        boot = np.divide(num[idx].sum(axis=1), boot_den, out=np.zeros_like(boot_den), where=boot_den != 0)
        low, high = ci(boot)
        return {"mean": point, "ci95_low": low, "ci95_high": high}

    def mean_metric(arr: np.ndarray) -> dict[str, float]:
        point = float(arr.mean()) if len(arr) else 0.0
        boot = arr[idx].mean(axis=1) if len(arr) else np.asarray([0.0])
        low, high = ci(boot)
        return {"mean": point, "ci95_low": low, "ci95_high": high}

    out: dict[str, Any] = {
        "captions": n,
        "visual_units": int(visual.sum()),
        "grounded_units_per_caption": mean_metric(grounded),
        "grounded_precision": ratio_metric(grounded, visual),
        "unsupported_rate": ratio_metric(unsupported, visual),
        "uncertain_rate": ratio_metric(uncertain, visual),
    }
    categories: dict[str, Any] = {}
    for category in UNIT_CATEGORIES:
        den = np.asarray([row.get(f"{category}_visual", 0) for row in rows], dtype=np.float64)
        cat_grounded = np.asarray([row.get(f"{category}_grounded", 0) for row in rows], dtype=np.float64)
        cat_unsupported = np.asarray([row.get(f"{category}_unsupported", 0) for row in rows], dtype=np.float64)
        cat_uncertain = np.asarray([row.get(f"{category}_uncertain", 0) for row in rows], dtype=np.float64)
        categories[category] = {
            "visual_units": int(den.sum()),
            "grounded_precision": ratio_metric(cat_grounded, den),
            "unsupported_rate": ratio_metric(cat_unsupported, den),
            "uncertain_rate": ratio_metric(cat_uncertain, den),
        }
    out["categories"] = categories
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def fmt_metric(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.4f} [{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    payload: dict[str, Any] = {
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "claimed": {},
        "grounded": {},
    }

    claimed_tsv: list[dict[str, Any]] = []
    for item in args.claimed:
        label, path = parse_label_path(item)
        rows = read_claimed(path, label)
        summary = summarize_claimed(rows, args.bootstrap_reps, rng)
        payload["claimed"][label] = {"input": str(path), **summary}
        claimed_tsv.append(
            {
                "surface": label,
                "captions": summary["captions"],
                "cbu_per_caption_ci95": fmt_metric(summary["dedup_units_per_caption"]),
                "cbu_per_100_tokens_ci95": fmt_metric(summary["dedup_units_per_100_tokens"]),
                "object_per_caption_ci95": fmt_metric(summary["object_per_caption"]),
                "attribute_per_caption_ci95": fmt_metric(summary["attribute_per_caption"]),
                "relation_per_caption_ci95": fmt_metric(summary["relation_per_caption"]),
                "camera_per_caption_ci95": fmt_metric(summary["camera_per_caption"]),
                "lighting_per_caption_ci95": fmt_metric(summary["lighting_per_caption"]),
                "text_rendering_per_caption_ci95": fmt_metric(summary["text_rendering_per_caption"]),
            }
        )

    grounded_tsv: list[dict[str, Any]] = []
    category_tsv: list[dict[str, Any]] = []
    for item in args.grounded:
        label, path = parse_label_path(item)
        rows = read_grounded(path, label)
        summary = summarize_grounded(rows, args.bootstrap_reps, rng)
        payload["grounded"][label] = {"input": str(path), **summary}
        grounded_tsv.append(
            {
                "surface": label,
                "captions": summary["captions"],
                "visual_units": summary["visual_units"],
                "grounded_units_per_caption_ci95": fmt_metric(summary["grounded_units_per_caption"]),
                "grounded_precision_ci95": fmt_metric(summary["grounded_precision"]),
                "unsupported_rate_ci95": fmt_metric(summary["unsupported_rate"]),
                "uncertain_rate_ci95": fmt_metric(summary["uncertain_rate"]),
            }
        )
        for category, cat in summary["categories"].items():
            category_tsv.append(
                {
                    "surface": label,
                    "category": category,
                    "visual_units": cat["visual_units"],
                    "grounded_precision_ci95": fmt_metric(cat["grounded_precision"]),
                    "unsupported_rate_ci95": fmt_metric(cat["unsupported_rate"]),
                    "uncertain_rate_ci95": fmt_metric(cat["uncertain_rate"]),
                }
            )

    (out_dir / "cbu_bootstrap_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_tsv(
        out_dir / "claimed_cbu_ci.tsv",
        claimed_tsv,
        [
            "surface",
            "captions",
            "cbu_per_caption_ci95",
            "cbu_per_100_tokens_ci95",
            "object_per_caption_ci95",
            "attribute_per_caption_ci95",
            "relation_per_caption_ci95",
            "camera_per_caption_ci95",
            "lighting_per_caption_ci95",
            "text_rendering_per_caption_ci95",
        ],
    )
    write_tsv(
        out_dir / "grounded_cbu_ci.tsv",
        grounded_tsv,
        [
            "surface",
            "captions",
            "visual_units",
            "grounded_units_per_caption_ci95",
            "grounded_precision_ci95",
            "unsupported_rate_ci95",
            "uncertain_rate_ci95",
        ],
    )
    write_tsv(
        out_dir / "grounded_cbu_category_ci.tsv",
        category_tsv,
        [
            "surface",
            "category",
            "visual_units",
            "grounded_precision_ci95",
            "unsupported_rate_ci95",
            "uncertain_rate_ci95",
        ],
    )
    print(
        json.dumps(
            {"output_dir": str(out_dir), "claimed": len(claimed_tsv), "grounded": len(grounded_tsv)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
