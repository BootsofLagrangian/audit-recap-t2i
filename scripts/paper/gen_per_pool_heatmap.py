"""Per-pool prompt-support heatmap for appendix.

Twin signed heatmaps (mass | direction-aligned JSD) sharing rows.
Each panel has a right-side row margin (per-pool wins/7 + mean Δ̄)
and a bottom column margin (per-slice wins/7 + mean Δ̄).
"""
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Sans-serif mathtext + upright default keeps in-label $\bar{\Delta}$
    # consistent with Helvetica body text instead of falling back to
    # serif/italic math glyphs.
    "mathtext.fontset": "stixsans",
    "mathtext.default": "regular",
})

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/prompt_support_bootstrap_b64_n2_250k_2026-04-24.tsv"
OUT = ROOT / "results/per_pool_prompt_support_heatmap.pdf"

POOL_LABEL = {
    "civitai_flux_prompts_aconexx":  "FLUX-aconexx",
    "flux_improved_k_mktr":           "FLUX-mktr",
    "flux_prompts_chrisgoringe":      "FLUX-chrisgoringe",
    "flux_prompts_regpeter":          "FLUX-regpeter",
    "pickapic_rankings":              "Pick-a-Pic",
    "sd_prompts_2m_andyyang":         "SD-andyyang",
    "sdxl_refiner_prompts_falah":     "SDXL-falah",
}

SLICE_ORDER = [
    ("datacomp_recap_llava15_paired_url", "DataComp",   "cross"),
    ("laion_pop_llama32_paired",          "LAION-pop",  "cross"),
    ("pd12m_full_paired",                 "PD12M",      "cross"),
    ("danbooru2023_florence2_paired",     "Danbooru",   "cross"),
    ("cc12m_llavanext_paired",            "LLaVA-NeXT", "cc12m"),
    ("cc12m_pixelprose_paired",           "PixelProse", "cc12m"),
    ("cc12m_qwen3vl8b_paired",            "Qwen3-VL$^{\\dagger}$", "cc12m"),
]

VLIM = 0.10
ZERO_BAND = 0.01

deltas = {("mass", p, s): np.nan for p in POOL_LABEL for s, _, _ in SLICE_ORDER}
deltas.update({("jsd", p, s): np.nan for p in POOL_LABEL for s, _, _ in SLICE_ORDER})

with open(SRC) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        pool = row["prompt_pool"]; slc  = row["comparison"]; m = row["metric"]
        d = float(row["delta_mean_local_minus_reference"])
        if pool not in POOL_LABEL: continue
        if m == "prompt_mass_on_caption_support":
            deltas[("mass", pool, slc)] = d
        elif m == "jsd":
            deltas[("jsd", pool, slc)] = d

pool_keys_sorted = sorted(
    POOL_LABEL.keys(),
    key=lambda p: -np.nanmean([deltas[("mass", p, s)] for s, _, _ in SLICE_ORDER]),
)
slice_keys = [s for s, _, _ in SLICE_ORDER]
slice_disp = [d for _, d, _ in SLICE_ORDER]
group_split = sum(1 for _, _, g in SLICE_ORDER if g == "cross")

mass_mat = np.array([[deltas[("mass", p, s)] for s in slice_keys] for p in pool_keys_sorted])
jsd_mat  = np.array([[-deltas[("jsd",  p, s)] for s in slice_keys] for p in pool_keys_sorted])
n_rows, n_cols = mass_mat.shape

# Paper-standard muted diverging palette (ColorBrewer RdBu_r) through neutral zero.
# Continuous gradient avoids the saturation jolt of the prior endpoint-explicit cmap;
# in-cell numbers use luminance-based auto-contrast (no halo).
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "paper_div",
    [
        (0.00, "#2166ac"),  # muted blue (vmin = -0.10)
        (0.25, "#67a9cf"),
        (0.50, "#f7f7f7"),  # near-neutral at zero
        (0.75, "#ef8a62"),
        (1.00, "#b2182b"),  # muted red (vmax = +0.10)
    ],
    N=256,
)
norm = mpl.colors.Normalize(vmin=-VLIM, vmax=VLIM)


def _auto_text_color(v: float) -> str:
    """Dark text on light cells, light text on dark cells (Rec. 601 luminance)."""
    rgba = cmap(norm(v))
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if lum < 0.55 else "#1a1a1a"

fig = plt.figure(figsize=(9.0, 3.9))
gs = fig.add_gridspec(
    nrows=2, ncols=5,
    width_ratios=[n_cols, 1.1, n_cols, 1.1, 0.35],
    height_ratios=[n_rows, 0.9],
    wspace=0.40, hspace=0.55,
    left=0.09, right=0.96, top=0.84, bottom=0.10,
)


def draw_heatmap(ax, mat, title, show_ylabels):
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i, j]
            if np.isnan(v): continue
            sat = abs(v) > VLIM
            label = "0.00" if abs(v) < ZERO_BAND else f"{v:+.2f}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=7, color=_auto_text_color(v),
                    fontweight="bold" if sat else "normal")

    # Thin white minor-grid between cells lifts cell separation without heavy borders.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(slice_disp, rotation=35, ha="right")
    ax.set_yticks(range(n_rows))
    if show_ylabels:
        ax.set_yticklabels([POOL_LABEL[p] for p in pool_keys_sorted])
    else:
        ax.set_yticklabels([])

    ax.axvline(group_split - 0.5, color="black", linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.set_title(title, fontsize=9, pad=22)

    # Group labels above the heatmap (between title and top of cells)
    ax.text((group_split - 1) / 2, -0.85, "Cross-corpus",
            ha="center", va="center", fontsize=7.5, style="italic",
            color="0.30", clip_on=False)
    ax.text(group_split + (n_cols - group_split - 1) / 2, -0.85, "CC12M",
            ha="center", va="center", fontsize=7.5, style="italic",
            color="0.30", clip_on=False)
    return im


def draw_row_margin(ax, mat):
    means = np.nanmean(mat, axis=1)
    wins  = np.sum(mat > ZERO_BAND, axis=1)
    ax.set_xlim(0, 1); ax.set_ylim(n_rows - 0.5, -0.5)
    for i in range(n_rows):
        ax.text(0.5, i - 0.22, f"{wins[i]}/{n_cols}",
                ha="center", va="center", fontsize=7, family="monospace")
        ax.text(0.5, i + 0.22, f"{means[i]:+.2f}",
                ha="center", va="center", fontsize=7, family="monospace",
                color="#b2182b" if means[i] > 0 else "#2166ac")
    ax.text(0.5, -0.85, r"wins / $\bar\Delta$",
            ha="center", va="center", fontsize=7, style="italic",
            color="0.30", clip_on=False)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def draw_col_margin(ax, mat):
    means = np.nanmean(mat, axis=0)
    wins  = np.sum(mat > ZERO_BAND, axis=0)
    ax.set_xlim(-0.5, n_cols - 0.5); ax.set_ylim(0, 1)
    for j in range(n_cols):
        ax.text(j, 0.72, f"{wins[j]}/{n_rows}",
                ha="center", va="center", fontsize=7, family="monospace")
        ax.text(j, 0.28, f"{means[j]:+.2f}",
                ha="center", va="center", fontsize=7, family="monospace",
                color="#b2182b" if means[j] > 0 else "#2166ac")
    ax.axvline(group_split - 0.5, color="black", linewidth=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


ax_mass     = fig.add_subplot(gs[0, 0])
ax_mass_rm  = fig.add_subplot(gs[0, 1])
ax_jsd      = fig.add_subplot(gs[0, 2])
ax_jsd_rm   = fig.add_subplot(gs[0, 3])
ax_cb       = fig.add_subplot(gs[0, 4])
ax_mass_cm  = fig.add_subplot(gs[1, 0])
ax_jsd_cm   = fig.add_subplot(gs[1, 2])

im = draw_heatmap(ax_mass, mass_mat,
                  "Δ prompt-mass support  (Ours − Ref)",
                  show_ylabels=True)
draw_heatmap(ax_jsd, jsd_mat,
             "JSD win delta  (−Δ JSD)", show_ylabels=False)
draw_row_margin(ax_mass_rm, mass_mat)
draw_row_margin(ax_jsd_rm,  jsd_mat)
draw_col_margin(ax_mass_cm, mass_mat)
draw_col_margin(ax_jsd_cm,  jsd_mat)

cbar = fig.colorbar(im, cax=ax_cb)
cbar.ax.tick_params(labelsize=7)
cbar.set_ticks([-VLIM, -ZERO_BAND, ZERO_BAND, VLIM])
cbar.set_ticklabels([f"−{VLIM:.2f}", f"−{ZERO_BAND:.2f}",
                     f"+{ZERO_BAND:.2f}", f"+{VLIM:.2f}"])
cbar.set_label("signed Δ  (red = Ours win)", fontsize=7.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, bbox_inches="tight")
plt.savefig(str(OUT).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print(f"saved {OUT}")
