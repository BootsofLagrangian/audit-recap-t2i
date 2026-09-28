#!/usr/bin/env python3
"""Write the appendix tabulars derived from cbu_vqa_by_category_b64.json.

Outputs (tabular bodies only; the captions are in the paper appendix):
  tables/cc12m_denominators.tex   per-surface CC12M extraction and VQA counts
  tables/excl_count_relation.tex  cross-corpus headline without count/relation
  tables/vqa_by_type.tex          per-type support/risk and CC12M judge agreement
  tables/policy_control.tex       supported yield and risk of the captioning-policy control
  tables/vqa_mean_std.tex         every VQA cell as mean and bootstrap standard deviation
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "results"
QWEN, GEMMA = "Qwen3.5-397B-A17B-FP8", "Gemma-4-31B-IT"
CC12M_ROWS = [
    ("Ours", "ours_cc12m"),
    ("CC12M-LLaVA-NeXT", "ref_cc12m_llavanext"),
    ("PixelProse", "ref_pixelprose_cc12m"),
    ("CC12M-Qwen3-VL${}^{\\dagger}$", "ref_cc12m_qwen3vl8b"),
]
LABEL = {"ours_cc12m": "Ours", "ref_cc12m_llavanext": "LLaVA-NeXT",
         "ref_pixelprose_cc12m": "PixelProse", "ref_cc12m_qwen3vl8b": "Qwen3-VL-8B"}
CONTROL_ROWS = {"CC12M-control": [], "DataComp-control": []}  # filled from the cells present in the summary
TYPES = ["object", "attribute", "relation", "count", "style", "camera", "lighting", "text_rendering"]


def n(x: int) -> str:
    return f"${x:,}$".replace(",", "{,}")


def cell(data: dict, slice_name: str, judge: str, surface: str) -> dict:
    return next(c for c in data["cells"] if (c["slice"], c["judge"], c["surface"]) == (slice_name, judge, surface))


def has_cell(data: dict, slice_name: str, surface: str) -> bool:
    return any((c["slice"], c["surface"]) == (slice_name, surface) and c["all_types"]["responses"] for c in data["cells"])


def denominators(data: dict) -> str:
    lines = []
    for label, key in CC12M_ROWS:
        ext = data["cc12m_extraction"][key]
        q = cell(data, "CC12M", QWEN, LABEL[key])["all_types"]
        g = cell(data, "CC12M", GEMMA, LABEL[key])["all_types"]
        assert q["responses"] == g["responses"] and q["questions"] == g["questions"]
        assert q["responses"] == ext["valid"] - ext.get("valid_zero_claims", 0)
        lines.append(" & ".join([
            label, n(ext["requests"]), n(ext["valid"]), n(ext.get("invalid_at_token_cap", 0)),
            n(ext.get("valid_zero_claims", 0)), n(q["responses"]), n(q["questions"]),
            f"${q['risk']:.3f}$", f"${g['risk']:.3f}$",
        ]) + r" \\")
    return "\n".join(lines) + "\n"


def excl_count_relation(data: dict) -> str:
    lines = []
    for slice_name in ["DataComp", "LAION-pop", "PD12M", "Danbooru"]:
        for judge, short in [(QWEN, "Qwen"), (GEMMA, "Gemma")]:
            ref, ours = cell(data, slice_name, judge, "Ref"), cell(data, slice_name, judge, "Ours")
            a_r, a_o = ref["all_types"], ours["all_types"]
            k_r, k_o = ref["excluding_count_relation"], ours["excluding_count_relation"]
            lines.append(" & ".join([
                slice_name if short == "Qwen" else "", short,
                f"${a_r['supported_cap']:.2f} \\to \\mathbf{{{a_o['supported_cap']:.2f}}}$",
                f"${k_r['supported_cap']:.2f} \\to \\mathbf{{{k_o['supported_cap']:.2f}}}$",
                f"${k_o['supported_cap'] - k_r['supported_cap']:+.2f}$",
                f"${k_r['risk']:.3f} \\to \\mathbf{{{k_o['risk']:.3f}}}$",
            ]) + r" \\")
    return "\n".join(lines) + "\n"


def vqa_by_type(data: dict) -> str:
    pooled: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for c in data["cells"]:
        if c["slice"].endswith("-control"):
            continue
        group = "Ours" if c["surface"] == "Ours" else "Refs"
        for claim_type, counts in c["by_type"].items():
            pooled[(c["judge"], group, claim_type)].update(counts)

    def rates(judge: str, group: str, claim_type: str) -> str:
        counts = pooled[(judge, group, claim_type)]
        total = sum(counts.values())
        return f"${counts['yes'] / total:.3f}$ & ${counts['no'] / total:.3f}$"

    agree = data["cc12m_judge_agreement"]
    lines = []
    for claim_type in TYPES:
        name = claim_type.replace("_", "-")
        lines.append(" & ".join([
            name, rates(QWEN, "Ours", claim_type), rates(QWEN, "Refs", claim_type),
            rates(GEMMA, "Ours", claim_type), rates(GEMMA, "Refs", claim_type),
            f"${100 * agree[claim_type]['exact_rate']:.1f}\\%$",
        ]) + r" \\")
    lines.append(r"\midrule")
    lines.append(f"all types & \\multicolumn{{8}}{{c}}{{}} & ${100 * agree['__all__']['exact_rate']:.1f}\\%$ \\\\")
    return "\n".join(lines) + "\n"


def policy_control(data: dict) -> str:
    rows = [("CC12M", "Ours", "CC12M", "Ours"), ("CC12M", "Naive", "CC12M-control", "Naive"),
            ("CC12M", "Naive (greedy)", "CC12M-control", "Naive (greedy)"),
            ("DataComp", "Ours", "DataComp-control", "Ours"), ("DataComp", "Naive", "DataComp-control", "Naive"),
            ("DataComp", "Naive (greedy)", "DataComp-control", "Naive (greedy)")]
    lines = []
    for dataset, surface, slice_name, key in rows:
        if not has_cell(data, slice_name, key):
            continue
        q = cell(data, slice_name, QWEN, key)["all_types"]
        g = cell(data, slice_name, GEMMA, key)["all_types"]
        assert q["responses"] == g["responses"]
        lines.append(" & ".join([
            dataset, surface, n(q["responses"]), n(q["questions"]),
            f"${q['supported_cap']:.2f}$", f"${q['risk']:.3f}$", f"${g['supported_cap']:.2f}$", f"${g['risk']:.3f}$",
        ]) + r" \\")
    return "\n".join(lines) + "\n"


def pm(value: float, std: float, digits: int) -> str:
    return f"${value:.{digits}f} \\pm {std:.{digits}f}$"


def mean_std_cells(data: dict, slice_name: str, surface: str) -> str:
    parts = []
    for judge in (QWEN, GEMMA):
        c = cell(data, slice_name, judge, surface)["all_types"]
        parts += [pm(c["supported_cap"], c["supported_cap_std"], 2), pm(c["risk"], c["risk_std"], 3)]
    return " & ".join(parts)


def vqa_mean_std(data: dict) -> str:
    rows = [("CC12M", label, "CC12M", LABEL[key]) for label, key in CC12M_ROWS]
    rows += [(s, {"Ref": "Reference", "Ours": "Ours"}[side], s, side)
             for s in ["DataComp", "LAION-pop", "PD12M", "Danbooru"] for side in ("Ref", "Ours")]
    for slice_name, dataset in (("CC12M-control", "CC12M control"), ("DataComp-control", "DataComp control")):
        rows += [(dataset, key, slice_name, key) for key in CONTROL_ROWS[slice_name]]
    lines, last = [], None
    for dataset, surface, slice_name, key in rows:
        if last is not None and dataset != last:
            lines.append(r"\addlinespace[0.2em]")
        lines.append(f"{dataset if dataset != last else ''} & {surface} & {mean_std_cells(data, slice_name, key)} \\\\")
        last = dataset
    return "\n".join(lines) + "\n"


def main() -> int:
    data = json.loads((HERE / "cbu_vqa_by_category_b64.json").read_text())
    for c in data["cells"]:
        if c["slice"] in CONTROL_ROWS and c["surface"] not in CONTROL_ROWS[c["slice"]] and c["all_types"]["responses"]:
            CONTROL_ROWS[c["slice"]].append(c["surface"])
    out = HERE / "tables"
    out.mkdir(exist_ok=True)
    (out / "cc12m_denominators.tex").write_text(denominators(data))
    (out / "excl_count_relation.tex").write_text(excl_count_relation(data))
    (out / "vqa_by_type.tex").write_text(vqa_by_type(data))
    (out / "policy_control.tex").write_text(policy_control(data))
    (out / "vqa_mean_std.tex").write_text(vqa_mean_std(data))
    for path in sorted(out.glob("*.tex")):
        print(f"== {path.name}\n{path.read_text()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
