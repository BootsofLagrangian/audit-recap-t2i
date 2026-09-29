"""Re-render Fig 2 (CC12M case study) at 5:3 (w:h) aspect with +30% axis font.

Design:
  * Dataset colour palette is shared across left and right panels
    (Okabe-Ito: Ours=blue, CC12M-LLaVA-NeXT=vermillion, PixelProse=green).
  * Left panel: dataset encodes colour+shape; judge encodes marker fill
    (Qwen=filled, Gemma=hollow). Two separated legends in the upper-right:
    Dataset (3 entries) and Judge (2 entries, neutral grey).
  * Right panel: dataset encodes colour+shape; budget B encodes marker size.
    Two separated legends in the upper-right: Dataset (3 entries) and
    B (4 labelled entries).
  * Qwen3-VL-8B is excluded from both panels.

Inputs:
  cbu_vqa_by_category_b64.json         -> left panel (Supported CBU vs Risk), CC12M cells on the
                                          requests answered by both Judges
  cc12m_budget_frontier_plot.csv       -> right panel (Claimed CBU vs CBU/100 lex)

Outputs:
  cc12m_vqa_supported_risk_pareto_v3.{pdf,png}
  cc12m_cbu_efficiency_yield_frontier_revised.{pdf,png}

PANELS=left or PANELS=right renders one panel; the default renders both.
"""
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
EVAL = Path(os.environ.get("EVAL_DIR", ROOT / "results"))
# Optional directory of Helvetica-like fonts (e.g. TeX Gyre Heros) for hosts without Helvetica.
for _font in glob.glob(os.path.join(os.environ.get("EXTRA_FONT_DIR", "/nonexistent"), "*.otf")):
    mpl.font_manager.fontManager.addfont(_font)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "TeX Gyre Heros", "Arial", "DejaVu Sans"],
    "font.size": 11.0,
    "axes.labelsize": 12.0,
    "axes.titlesize": 12.0,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9.0,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DATASETS = ["Ours", "CC12M-LLaVA-NeXT", "PixelProse"]
DATASET_MARKER = {"Ours": "o", "CC12M-LLaVA-NeXT": "s", "PixelProse": "^"}
DATASET_COLOR  = {
    "Ours":             "#0072B2",  # blue       (shared L/R)
    "CC12M-LLaVA-NeXT": "#D55E00",  # vermillion (shared L/R)
    "PixelProse":       "#009E73",  # bluish green (shared L/R)
}
JUDGE_LABEL = {"Qwen3.5-397B-A17B-FP8": "Qwen", "Gemma-4-31B-it": "Gemma", "Gemma-4-31B-IT": "Gemma"}
LABEL_MAP   = {"LLaVA-NeXT": "CC12M-LLaVA-NeXT"}  # summaries use bare LLaVA-NeXT
PANELS = set(os.environ.get("PANELS", "left,right").split(","))


# =====================================================================
# LEFT panel: Supported CBU per caption  vs  Risk (lower better)
#   * dataset = colour + shape
#   * judge   = marker fill (Qwen=filled, Gemma=hollow)
# =====================================================================
cells = json.load(open(EVAL / "cbu_vqa_by_category_b64.json"))["cells"]
fig, ax = plt.subplots(figsize=(5.0, 3.0))

by_label = defaultdict(list)  # label -> [(judge, sup, risk)]
for cell in cells:
    if cell["slice"] != "CC12M" or cell["surface"] == "Qwen3-VL-8B":
        continue
    label = LABEL_MAP.get(cell["surface"], cell["surface"])
    judge = JUDGE_LABEL.get(cell["judge"], cell["judge"])
    by_label[label].append((judge, cell["all_types"]["supported_cap"], cell["all_types"]["risk"]))
assert sorted(by_label) == sorted(DATASETS) and all(len(v) == 2 for v in by_label.values()), by_label

# Gray connector across the two judges for each dataset.
for label, pts in by_label.items():
    if len(pts) >= 2:
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        ax.plot(xs, ys, "-", color="#999999", linewidth=0.9, alpha=0.55, zorder=2)

for label, pts in by_label.items():
    color = DATASET_COLOR[label]
    marker = DATASET_MARKER[label]
    for judge, sup, risk in pts:
        if judge == "Qwen":
            ax.scatter(sup, risk, marker=marker, s=140,
                       facecolors=color, edgecolors="black",
                       linewidths=0.7, alpha=0.95, zorder=3)
        else:  # Gemma -> hollow: face=white, thick coloured edge
            ax.scatter(sup, risk, marker=marker, s=140,
                       facecolors="white", edgecolors=color,
                       linewidths=1.6, alpha=0.95, zorder=3)

ax.set_xlabel(r"Supported CBU per caption ( $\uparrow$ better)")
ax.set_ylabel(r"Risk ( $\downarrow$ better)")
ax.grid(True, alpha=0.25, linestyle=":")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
y_lo, y_hi = ax.get_ylim()
ax.set_ylim(y_lo, y_hi + (y_hi - y_lo) * 0.18)

dataset_handles = [
    Line2D([0], [0], marker=DATASET_MARKER[lbl], color="w",
           markerfacecolor=DATASET_COLOR[lbl], markeredgecolor="black",
           markeredgewidth=0.7, markersize=8, label=lbl)
    for lbl in DATASETS
]
judge_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor="#555555", markeredgecolor="black",
           markeredgewidth=0.7, markersize=8, label="Qwen Judge"),
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor="white", markeredgecolor="#555555",
           markeredgewidth=1.6, markersize=8, label="Gemma Judge"),
]
leg1 = ax.legend(handles=dataset_handles, title="Dataset",
                 loc="upper right", bbox_to_anchor=(0.995, 0.995),
                 framealpha=0.95, borderpad=0.35, labelspacing=0.3,
                 handletextpad=0.35, fontsize=8.0, title_fontsize=8.5)
ax.add_artist(leg1)
ax.legend(handles=judge_handles, title="Judge",
          loc="upper right", bbox_to_anchor=(0.995, 0.66),
          framealpha=0.95, borderpad=0.35, labelspacing=0.3,
          handletextpad=0.35, fontsize=8.0, title_fontsize=8.5)

plt.tight_layout(pad=0.4)
out_left = EVAL / "cc12m_vqa_supported_risk_pareto_v3.pdf"
if "left" in PANELS:
    plt.savefig(out_left, bbox_inches="tight")
    plt.savefig(str(out_left).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_left}")
plt.close(fig)


# =====================================================================
# RIGHT panel: Claimed CBU per caption  vs  CBU per 100 lex
#   * dataset = colour + shape (palette matches LEFT)
#   * B       = marker size (with labelled budget legend)
# =====================================================================
rows = list(csv.DictReader(open(EVAL / "cc12m_budget_frontier_plot.csv")))
fig, ax = plt.subplots(figsize=(5.0, 3.0))

budget_to_size = {16: 55, 32: 100, 48: 155, 64: 220}
traj = defaultdict(list)
for r in rows:
    raw = r["label"]
    if raw == "Qwen3-VL-8B":
        continue  # excluded from right panel per user
    label = LABEL_MAP.get(raw, raw)
    traj[label].append((int(r["budget"]),
                        float(r["cbu_per_cap"]),
                        float(r["cbu_per_100tok"])))

for label in DATASETS:
    points = sorted(traj[label])
    Xs = [p[1] for p in points]
    Ys = [p[2] for p in points]
    color = DATASET_COLOR[label]
    marker = DATASET_MARKER[label]
    ax.plot(Xs, Ys, "-", color=color, linewidth=1.4, alpha=0.6, zorder=2)
    for B, x, y in points:
        ax.scatter(x, y, marker=marker, s=budget_to_size[B], c=color,
                   edgecolors="black", linewidths=0.7, alpha=0.95, zorder=3)

ax.set_xlabel(r"Claimed CBU per caption ( $\uparrow$ better)")
ax.set_ylabel(r"CBU per 100 lex ( $\uparrow$ better)")
ax.grid(True, alpha=0.25, linestyle=":")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
y_lo, y_hi = ax.get_ylim()
ax.set_ylim(y_lo, y_hi + (y_hi - y_lo) * 0.20)

dataset_handles = [
    Line2D([0], [0], marker=DATASET_MARKER[lbl], color="w",
           markerfacecolor=DATASET_COLOR[lbl], markeredgecolor="black",
           markeredgewidth=0.7, markersize=8, label=lbl)
    for lbl in DATASETS
]
budget_marker_size = {16: 5, 32: 7, 48: 9, 64: 11}
budget_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor="#888888", markeredgecolor="black",
           markeredgewidth=0.6, markersize=budget_marker_size[B],
           label=f"$B$={B}")
    for B in [16, 32, 48, 64]
]

leg1 = ax.legend(handles=dataset_handles, title="Dataset",
                 loc="upper right", bbox_to_anchor=(0.995, 0.995),
                 framealpha=0.95, borderpad=0.35, labelspacing=0.3,
                 handletextpad=0.35, fontsize=8.0, title_fontsize=8.5)
ax.add_artist(leg1)
ax.legend(handles=budget_handles, title="Budget",
          loc="upper right", bbox_to_anchor=(0.995, 0.66),
          ncol=2, framealpha=0.95, borderpad=0.35, labelspacing=0.25,
          handletextpad=0.3, columnspacing=0.6,
          fontsize=8.0, title_fontsize=8.5)

plt.tight_layout(pad=0.4)
out_right = EVAL / "cc12m_cbu_efficiency_yield_frontier_revised.pdf"
if "right" in PANELS:
    plt.savefig(out_right, bbox_inches="tight")
    plt.savefig(str(out_right).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_right}")
plt.close(fig)
