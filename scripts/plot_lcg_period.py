# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL VALIDATION: LCG Full Period & Lag-1 Autocorrelation
# ──────────────────────────────────────────────────────────────────────────────
# Description: Generates the full period of an LCG to visualize the complete
#              sequence distribution (Panel A) and evaluates sequential
#              independence using a Lag-1 scatter plot (Panel B).
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from prng.generators import linear_congruential_generator

# ── Constants & Theming ───────────────────────────────────────────────────────
COLORS = {
    "bg": "#1e2127",
    "panel": "#282c34",
    "grid": "#3e4451",
    "text": "#abb2bf",
    "gold": "#e8d5a3",
    "blue": "#61afef",
    "green": "#98c379",
    "red": "#e06c75",
    "orange": "#e5c07b",
    "purple": "#c678dd",
    "teal": "#56b6c2",
    "peach": "#ffc9b9",
    "rose": "#f49cbb",
    "white": "#ffffff",
}

# Set global matplotlib parameters for consistency
plt.rcParams.update(
    {
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["panel"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "text.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.4,
        "font.family": "sans-serif",
        "savefig.dpi": 300,
    }
)


def style_axis(ax, title, color=COLORS["gold"]):
    """Applies consistent styling to a given matplotlib axis."""
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(COLORS["grid"])
    ax.set_title(title, color=color, fontsize=10, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.25)


def main():
    # ── 1. Generate Pseudorandom Sequence ─────────────────────────────────────
    seed, a, c, g = 55, 37, 19, 7
    m = 2**g

    # Generate the sequence and convert to a NumPy array for efficient slicing
    sequence = linear_congruential_generator(seed, a, c, m)

    n_samples = len(sequence)  # Should equal m (128) for a full period LCG
    idx = np.arange(1, n_samples + 1)

    # ── 2. Plotting Setup ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        f"Full Period of the Linear Congruential Generator \n"
        f"($X_0={seed},\\ a={a},\\ c={c},\\ m=2^{{{g}}}={m}$)",
        color=COLORS["gold"],
        fontsize=14,
        fontweight="bold",
    )

    # Create a custom bright gradient
    theme_cmap = LinearSegmentedColormap.from_list(
        "ThemeGlow", [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]
    )

    # ── Panel A: Sequence Plot (Gradient Theme) ───────────────────────────────
    ax1 = axes[0]
    style_axis(ax1, "A — Complete Sequence ($r_i$ vs $i$)", COLORS["gold"])

    # Plot all points, coloring them chronologically by their Index (i)
    ax1.scatter(
        idx,
        sequence,
        c=idx,
        cmap=theme_cmap,
        s=20,
        alpha=0.9,
        zorder=3,
    )

    # Expected mean reference line
    ax1.axhline(
        0.5,
        color=COLORS["white"],
        lw=1.5,
        linestyle="--",
        alpha=0.8,
        label="Expected Mean ($0.5$)",
    )

    ax1.set_xlabel("Iteration ($i$)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Pseudorandom Value ($r_i$)", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, n_samples + 2)
    ax1.set_ylim(-0.04, 1.10)
    ax1.legend(
        fontsize=9,
        facecolor=COLORS["panel"],
        edgecolor=COLORS["grid"],
        labelcolor=COLORS["text"],
    )

    # ── Panel B: Lag-1 Scatter Plot (ri vs r_{i+1}) ───────────────────────────
    ax2 = axes[1]
    style_axis(ax2, "B — Lag-1 Scatter Plot ($r_i$ vs $r_{i+1}$)", COLORS["gold"])

    # Shift sequence by 1 to compare current vs next
    r_curr = sequence[:-1]
    r_next = sequence[1:]

    # Color by position in sequence to show temporal flow using a colormap
    sc = ax2.scatter(
        r_curr,
        r_next,
        c=np.arange(len(r_curr)),
        cmap=theme_cmap,
        s=22,
        alpha=0.9,
        zorder=3,
    )

    # Configure Colorbar
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Sequence Index ($i$)", color=COLORS["text"], fontsize=9, fontweight="bold"
    )
    cbar.ax.yaxis.set_tick_params(color=COLORS["text"], labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["text"])

    ax2.set_xlabel("Current Value ($r_i$)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Next Value ($r_{i+1}$)", fontsize=10, fontweight="bold")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
