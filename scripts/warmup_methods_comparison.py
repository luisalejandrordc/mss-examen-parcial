# ══════════════════════════════════════════════════════════════════════════════
# Comparison between five methods for identifying the warm-up point of a variable
# ══════════════════════════════════════════════════════════════════════════════

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from warmup.detection import (
    method_backward_cusum,
    method_ci_width,
    method_forward_cusum,
    method_relative_change,
    method_welch,
)

# Available Variables: "interarrival_time", "crushing_time", "mineral_load"
VARIABLE = "interarrival_time"

# ── Load data ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
csv_path = SCRIPT_DIR.parent / "data" / "mining_simulation.csv"

df = pd.read_csv(csv_path)
data = df[VARIABLE].to_numpy(dtype=float)
n = len(df)

# ── Running statistics ─────────────────────────────────────────────────────────
running_avg = np.array([data[: i + 1].mean() for i in range(n)])
running_std = np.array([data[: i + 1].std(ddof=1) if i > 0 else 0.0 for i in range(n)])
ci_margin = 1.96 * running_std / np.sqrt(np.arange(1, n + 1))


# ══════════════════════════════════════════════════════════════════════════════
# Run all methods
# ══════════════════════════════════════════════════════════════════════════════
wu1 = method_relative_change(data)
wu2 = method_welch(data)
wu3 = method_ci_width(data)
wu4 = method_forward_cusum(data)
wu5 = method_backward_cusum(data)

print("=" * 55)
print(f"  {'Method':<35} {'Warm-up':>8}")
print("=" * 55)
print(
    f"  {'1. Relative Change (' + str(wu1.diagnostics["threshold"] * 100) + '%, w=10)':<35} {wu1.warmup:>8}"
)
print(f"  {'2. Welch Graphical (±5% band)':<35} {wu2.warmup:>8}")
print(f"  {'3. CI Width Stabilization':<35} {wu3.warmup:>8}")
print(f"  {'4. Forward CUSUM (k=0.5σ, h=4σ)':<35} {wu4.warmup:>8}")
print(f"  {'5. Backward CUSUM (k=0.5σ, h=4σ)':<35} {wu5.warmup:>8}")
print("=" * 55)

idx = np.arange(n)

# ══════════════════════════════════════════════════════════════════════════════
# Plot — 5-panel figure, one per method
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "m1": "#e06c75",  # red
    "m2": "#61afef",  # blue
    "m3": "#98c379",  # green
    "m4": "#e5c07b",  # yellow
    "m5": "#c678dd",  # purple
    "avg": "#abb2bf",
    "raw": "#4b5263",
    "warmup": "#ff6b6b",
}

fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor("#1e2127")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

PANEL_BG = "#282c34"


def style_ax(ax, title, color):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="#abb2bf", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3e4451")
    ax.set_title(title, color=color, fontsize=10, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.15, color="#abb2bf")
    ax.set_xlabel("Observation", color="#abb2bf", fontsize=8)


# ── Panel 1: Relative Change ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Method 1 — Relative Change Criterion", COLORS["m1"])
ax1.plot(idx, running_avg, color=COLORS["avg"], lw=1.5, label="Running avg")
ax1.axvline(
    wu1.warmup,
    color=COLORS["m1"],
    lw=2,
    linestyle="--",
    label=f"Warm-up = {wu1.warmup}",
)
ax1b = ax1.twinx()
ax1b.plot(
    idx,
    wu1.diagnostics["rel_change"] * 100,
    color=COLORS["m1"],
    lw=0.8,
    alpha=0.6,
    linestyle=":",
    label="Rel. change (%)",
)
ax1b.axhline(
    wu1.diagnostics["threshold"] * 100,
    color=COLORS["m1"],
    lw=0.6,
    linestyle="-.",
    alpha=0.5,
)
ax1b.tick_params(colors="#abb2bf", labelsize=7)
ax1b.set_ylim(0, 20)
ax1b.set_ylabel("Rel. change (%)", color=COLORS["m1"], fontsize=7)
ax1.set_ylabel("Running avg", color="#abb2bf", fontsize=8)
lines1, labels1 = ax1.get_legend_handles_labels()
lines1b, labels1b = ax1b.get_legend_handles_labels()
ax1.legend(
    lines1 + lines1b,
    labels1 + labels1b,
    loc="upper right",
    fontsize=7,
    facecolor=PANEL_BG,
    labelcolor="#abb2bf",
    edgecolor="#3e4451",
)

# ── Panel 2: Welch ────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Method 2 — Welch's Graphical Method", COLORS["m2"])
ax2.plot(idx, running_avg, color=COLORS["avg"], lw=1, alpha=0.5, label="Running avg")
ax2.plot(
    idx, wu2.diagnostics["smoothed"], color=COLORS["m2"], lw=2, label="Smoothed avg"
)
ax2.axhline(
    wu2.diagnostics["grand_mean"],
    color=COLORS["m2"],
    lw=0.8,
    linestyle="--",
    alpha=0.6,
    label=f"Grand mean={wu2.diagnostics["grand_mean"]:.3f}",
)
ax2.fill_between(
    idx,
    wu2.diagnostics["grand_mean"] - wu2.diagnostics["band"],
    wu2.diagnostics["grand_mean"] + wu2.diagnostics["band"],
    color=COLORS["m2"],
    alpha=0.08,
    label="±5% stability band",
)
ax2.axvline(
    wu2.warmup,
    color=COLORS["m2"],
    lw=2,
    linestyle="--",
    label=f"Warm-up = {wu2.warmup}",
)
ax2.set_ylabel("Value", color="#abb2bf", fontsize=8)
ax2.legend(
    loc="upper right",
    fontsize=7,
    facecolor=PANEL_BG,
    labelcolor="#abb2bf",
    edgecolor="#3e4451",
)

# ── Panel 3: CI Width ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Method 3 — CI Width Stabilization", COLORS["m3"])
ax3.plot(idx, running_avg, color=COLORS["avg"], lw=1.5, label="Running avg")
ax3.fill_between(
    idx,
    running_avg - wu3.diagnostics["ci_margin"],
    running_avg + wu3.diagnostics["ci_margin"],
    color=COLORS["m3"],
    alpha=0.15,
    label="95% CI",
)
ax3.axvline(
    wu3.warmup,
    color=COLORS["m3"],
    lw=2,
    linestyle="--",
    label=f"Warm-up = {wu3.warmup}",
)
ax3b = ax3.twinx()
ax3b.plot(
    idx,
    wu3.diagnostics["rel_reduction"] * 100,
    color=COLORS["m3"],
    lw=1,
    linestyle=":",
    alpha=0.7,
    label="Rel. reduction (%)",
)
ax3b.axhline(
    wu3.diagnostics["threshold"] * 100,
    color=COLORS["m3"],
    lw=0.6,
    linestyle="-.",
    alpha=0.5,
)
ax3b.tick_params(colors="#abb2bf", labelsize=7)
ax3b.set_ylim(0, 20)
ax3b.set_ylabel("Rel. reduction (%)", color=COLORS["m3"], fontsize=7)
ax3.set_ylabel("Running avg", color="#abb2bf", fontsize=8)
lines3, labels3 = ax3.get_legend_handles_labels()
lines3b, labels3b = ax3b.get_legend_handles_labels()
ax3.legend(
    lines3 + lines3b,
    labels3 + labels3b,
    loc="upper right",
    fontsize=7,
    facecolor=PANEL_BG,
    labelcolor="#abb2bf",
    edgecolor="#3e4451",
)

# ── Panel 4: Forward CUSUM ────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Method 4 — Forward CUSUM Control Chart", COLORS["m4"])
ax4.plot(idx, wu4.diagnostics["C_plus"], color=COLORS["m4"], lw=1.5, label="C⁺ (upper)")
ax4.plot(
    idx,
    wu4.diagnostics["C_minus"],
    color="#e06c75",
    lw=1.5,
    linestyle="--",
    label="C⁻ (lower)",
)
ax4.axhline(
    wu4.diagnostics["h"],
    color=COLORS["m4"],
    lw=1,
    linestyle="-.",
    alpha=0.7,
    label=f"Decision limit h={wu4.diagnostics["h"]:.2f}",
)
ax4.axvline(
    wu4.warmup,
    color=COLORS["m4"],
    lw=2,
    linestyle="--",
    label=f"Warm-up = {wu4.warmup}",
)
ax4.set_ylabel("CUSUM statistic", color="#abb2bf", fontsize=8)
ax4.legend(
    loc="upper right",
    fontsize=7,
    facecolor=PANEL_BG,
    labelcolor="#abb2bf",
    edgecolor="#3e4451",
)

# ── Panel 5: Backward CUSUM ────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, "Method 5 — Backward CUSUM Control Chart", COLORS["m5"])
ax5.plot(idx, wu5.diagnostics["C_plus"], color=COLORS["m5"], lw=1.5, label="C⁺ (upper)")
ax5.plot(
    idx,
    wu5.diagnostics["C_minus"],
    color="#e06c75",
    lw=1.5,
    linestyle="--",
    label="C⁻ (lower)",
)
ax5.axhline(
    wu5.diagnostics["h"],
    color=COLORS["m5"],
    lw=1,
    linestyle="-.",
    alpha=0.7,
    label=f"Decision limit h={wu5.diagnostics["h"]:.2f}",
)
ax5.axvline(
    wu5.warmup,
    color=COLORS["m5"],
    lw=2,
    linestyle="--",
    label=f"Warm-up = {wu5.warmup}",
)
ax5.set_ylabel("CUSUM statistic", color="#abb2bf", fontsize=8)
ax5.legend(
    loc="upper right",
    fontsize=7,
    facecolor=PANEL_BG,
    labelcolor="#abb2bf",
    edgecolor="#3e4451",
)

# ── Panel 6: Summary comparison ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, "Summary — All Methods on Running Average", "#e8d5a3")
ax6.plot(idx, running_avg, color=COLORS["avg"], lw=1.5, zorder=1, label="Running avg")

warmups = [wu1, wu2, wu3, wu4, wu5]
labels = [
    "M1 Rel.Change",
    "M2 Welch",
    "M3 CI Width",
    "M4 Forward CUSUM",
    "M5 Backward CUSUM",
]
cols = [COLORS["m1"], COLORS["m2"], COLORS["m3"], COLORS["m4"], COLORS["m5"]]

for wu, lbl, col in zip(warmups, labels, cols):
    ax6.axvline(
        wu.warmup,
        color=col,
        lw=1.5,
        linestyle="--",
        alpha=0.85,
        label=f"{lbl}={wu.warmup}",
    )

ax6.set_ylabel("Running avg", color="#abb2bf", fontsize=8)
ax6.legend(
    loc="upper right",
    fontsize=6.5,
    facecolor=PANEL_BG,
    labelcolor="#abb2bf",
    edgecolor="#3e4451",
)

# ── Super title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "Warm-up Point Detection — Comparative Analysis",
    color="#e8d5a3",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

plt.savefig(
    "warmup_comparison.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
)
plt.show()
print("\nFigure saved as warmup_comparison.png")
