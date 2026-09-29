"""Refined twinx + bar_mass — Apple-pastel palette, 2-decimal JSD ticks,
cleaner legend, less visual conflict between bars and JSD layer.

Values are the means over the seven prompt pools of the per-pool block means in
prompt_support_bootstrap_b64_n2_250k_2026-04-24.tsv (B = 64, n-gram n = 2).
"""
import csv
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import numpy as np

# Optional directory of Helvetica-like fonts (e.g. TeX Gyre Heros) for hosts without Helvetica.
for _font in glob.glob(os.path.join(os.environ.get("EXTRA_FONT_DIR", "/nonexistent"), "*.otf")):
    mpl.font_manager.fontManager.addfont(_font)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "TeX Gyre Heros", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "0.3",
    "xtick.labelsize": 9,
    "ytick.labelsize": 8.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT = os.environ.get("EVAL_DIR", str(Path(__file__).resolve().parents[2] / "results"))
SRC = Path(OUT) / "prompt_support_bootstrap_b64_n2_250k_2026-04-24.tsv"

POOLS = (
    "civitai_flux_prompts_aconexx", "flux_improved_k_mktr", "flux_prompts_chrisgoringe",
    "flux_prompts_regpeter", "pickapic_rankings", "sd_prompts_2m_andyyang", "sdxl_refiner_prompts_falah",
)
PAIRS = (
    ("datacomp_recap_llava15_paired_url", "DataComp"), ("laion_pop_llama32_paired", "LAION-pop"),
    ("pd12m_full_paired", "PD12M"), ("danbooru2023_florence2_paired", "Danbooru"),
)


def pool_means(metric, column):
    with SRC.open() as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t") if r["metric"] == metric and r["prompt_pool"] in POOLS]
    out = []
    for key, _ in PAIRS:
        values = [float(r[column]) for r in rows if r["comparison"] == key]
        assert len(values) == len(POOLS), (key, metric, len(values))
        out.append(sum(values) / len(values))
    return out


datasets = [label for _, label in PAIRS]
mass_o = pool_means("prompt_mass_on_caption_support", "local_block_mean")
mass_r = pool_means("prompt_mass_on_caption_support", "reference_block_mean")
jsd_o = pool_means("jsd", "local_block_mean")
jsd_r = pool_means("jsd", "reference_block_mean")

# Apple-pastel-leaning colour-blind-safe pair
C_OURS_BAR = "#7FB3F6"   # soft cobalt
C_REF_BAR  = "#FFB48A"   # warm peach
C_OURS_DOT = "#1F5E9F"   # deep blue (JSD layer accent)
C_REF_DOT  = "#C4633A"   # rich terracotta (JSD layer accent)


def save(fig, name):
    p = f"{OUT}/teaser_right_{name}.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}")


def gen_twinx_refined():
    fig, ax = plt.subplots(figsize=(3.9, 2.7))
    x = np.arange(len(datasets))
    w = 0.36

    # Mass bars (left y) — pastel solid fills, soft black edge
    bars_o = ax.bar(x - w/2, mass_o, w, color=C_OURS_BAR,
                    edgecolor="0.25", linewidth=0.7, zorder=2)
    bars_r = ax.bar(x + w/2, mass_r, w, color=C_REF_BAR,
                    edgecolor="0.25", linewidth=0.7, zorder=2)

    # Numeric value labels above bars only — reader sees which axis instantly
    for bars in (bars_o, bars_r):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.013,
                    f"{b.get_height():.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="0.25", zorder=3)

    ax.set_ylabel(r"Prompt-mass support  ($\uparrow$)", color="0.2")
    ax.set_ylim(0, 0.78)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18, linestyle=":", zorder=0)

    # JSD on twin y — filled dots in deeper Ours/Ref shades, no overlap with bars
    ax2 = ax.twinx()
    ax2.plot(x, jsd_o, "o-", color=C_OURS_DOT, markeredgecolor="white",
             markeredgewidth=1.0, linewidth=1.4, markersize=7.5, zorder=4)
    ax2.plot(x, jsd_r, "s--", color=C_REF_DOT, markeredgecolor="white",
             markeredgewidth=1.0, linewidth=1.4, markersize=7.5, zorder=4)
    ax2.set_ylabel(r"n-gram JSD  ($\downarrow$)", color="0.2")
    ax2.set_ylim(0.40, 0.58)
    ax2.set_yticks([0.40, 0.45, 0.50, 0.55])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # Single legend at top, 4 entries, well-spaced
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=C_OURS_BAR, edgecolor="0.25", linewidth=0.7,
              label="Ours · support"),
        Patch(facecolor=C_REF_BAR, edgecolor="0.25", linewidth=0.7,
              label="Ref · support"),
        Line2D([0], [0], marker="o", color=C_OURS_DOT,
               markeredgecolor="white", markeredgewidth=1.0,
               markersize=7, linewidth=1.4, label="Ours · JSD"),
        Line2D([0], [0], marker="s", color=C_REF_DOT,
               markeredgecolor="white", markeredgewidth=1.0,
               linestyle="--",
               markersize=7, linewidth=1.4, label="Ref · JSD"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False,
               handletextpad=0.45, columnspacing=1.1, fontsize=8.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    save(fig, "v_twinx_v3")


def gen_bar_mass_refined():
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(len(datasets))
    w = 0.36

    bars_o = ax.bar(x - w/2, mass_o, w, color=C_OURS_BAR,
                    edgecolor="0.25", linewidth=0.7, label="Ours", zorder=2)
    bars_r = ax.bar(x + w/2, mass_r, w, color=C_REF_BAR,
                    edgecolor="0.25", linewidth=0.7, label="Ref", zorder=2)

    # Numeric value labels above bars
    for bars in (bars_o, bars_r):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.013,
                    f"{b.get_height():.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="0.25")

    ax.set_ylabel(r"Prompt-mass support  ($\uparrow$)", color="0.2")
    ax.set_ylim(0, 0.74)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18, linestyle=":", zorder=0)

    leg = ax.legend(loc="upper right", frameon=False, handletextpad=0.4,
                    handlelength=1.4)

    plt.tight_layout()
    save(fig, "v_bar_mass_v2")


if __name__ == "__main__":
    gen_twinx_refined()
