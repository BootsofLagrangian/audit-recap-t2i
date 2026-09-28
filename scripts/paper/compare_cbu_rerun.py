#!/usr/bin/env python3
"""Compare a claimed-CBU rerun against the original responses for the same requests.

Both inputs are ``run_text_json_requests.py`` response JSONL files. Requests are
joined on request_id; each valid response is reduced to its set of lowercased
(category, unit, target) keys, the deduplication key used by
``summarize_cbu_responses.py``. The report gives, per surface, the share of
requests whose key sets match exactly and the mean claimed CBU per caption in
each run.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["request_id"]] = row
    return rows


def keys(row: dict) -> frozenset | None:
    if not row.get("ok") or row.get("parse_error") or row.get("schema_error") or row.get("parsed") is None:
        return None
    return frozenset(
        (str(u.get("category", "")).lower(), str(u.get("unit", "")).strip().lower(), str(u.get("target", "")).strip().lower())
        for u in row["parsed"].get("claimed_units", [])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--original", required=True)
    parser.add_argument("--rerun", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    original, rerun = load(Path(args.original)), load(Path(args.rerun))
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for request_id, new in rerun.items():
        old = original.get(request_id)
        if old is None:
            continue
        surface = new["request"]["surface"]
        s = stats[surface]
        s["joined"] += 1
        a, b = keys(old), keys(new)
        if a is None or b is None:
            s["invalid_either"] += 1
            continue
        s["valid_both"] += 1
        s["exact_match"] += a == b
        s["cbu_original"] += len(a)
        s["cbu_rerun"] += len(b)
        s["jaccard_sum"] += len(a & b) / len(a | b) if a | b else 1.0
    report = {}
    for surface, s in sorted(stats.items()):
        n = s["valid_both"] or 1
        report[surface] = {
            "joined": int(s["joined"]), "valid_both": int(s["valid_both"]), "invalid_either": int(s["invalid_either"]),
            "exact_match_rate": s["exact_match"] / n, "mean_jaccard": s["jaccard_sum"] / n,
            "cbu_per_cap_original": s["cbu_original"] / n, "cbu_per_cap_rerun": s["cbu_rerun"] / n,
        }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
