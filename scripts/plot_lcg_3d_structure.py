# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL VALIDATION: LCG 3D Lattice Structure (Marsaglia Effect)
# ──────────────────────────────────────────────────────────────────────────────
# Description: Generates sequential triplets (r_i, r_{i+1}, r_{i+2}) from an LCG
#              and plots them in 3D space to visually demonstrate the Marsaglia
#              hyperplane flaw inherent to all linear congruential generators.
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
    "white": "#ffffff",
}

plt.rcParams.update(
    {
        "figure.facecolor": COLORS["panel"],
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


def style_3d_axis(ax):
    """Applies dark theme styling to 3D Matplotlib panes."""
    # Set the background color of the 3D panels
    ax.xaxis.set_pane_color(COLORS["panel"])
    ax.yaxis.set_pane_color(COLORS["panel"])
    ax.zaxis.set_pane_color(COLORS["panel"])

    # Set the color of the bounding box lines
    ax.xaxis.line.set_color(COLORS["grid"])
    ax.yaxis.line.set_color(COLORS["grid"])
    ax.zaxis.line.set_color(COLORS["grid"])


def main():
    # ── 1. Generate Pseudorandom Sequence ─────────────────────────────────────
    # Note: Using m=1024 (from your Q6 script) instead of m=128 here.
    # A larger sequence makes the 3D parallel planes much denser and easier to see!
    seed, a, c, g = 83, 141, 53, 10
    m = 2**g

    sequence = linear_congruential_generator(seed, a, c, m)
    seq_arr = np.array(sequence)

    # ── 2. Extract Triplets for 3D Space ──────────────────────────────────────
    # We need (x, y, z) = (r_i, r_{i+1}, r_{i+2})
    r_curr = seq_arr[:-2]
    r_next = seq_arr[1:-1]
    r_next2 = seq_arr[2:]

    # ── 3. Plotting Setup ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(
        f"LCG 3D Lattice Structure: The Marsaglia Effect\n"
        f"($X_0={seed},\\ a={a},\\ c={c},\\ m=2^{{{g}}}={m}$)",
        color=COLORS["white"],
        fontsize=14,
        fontweight="bold",
        # y=0.95,
    )

    # Add 3D subplot
    ax = fig.add_subplot(111, projection="3d")
    style_3d_axis(ax)

    # Custom continuous gradient
    theme_cmap = LinearSegmentedColormap.from_list(
        "ThemeGlow", [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]
    )

    # ── 4. Scatter Plot ───────────────────────────────────────────────────────
    sc = ax.scatter(
        r_curr,
        r_next,
        r_next2,
        c=np.arange(len(r_curr)),
        cmap=theme_cmap,
        s=15,
        alpha=0.85,
        zorder=3,
    )

    # Configure Colorbar
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.05)
    cbar.set_label(
        "Sequence Index ($i$)", color=COLORS["text"], fontsize=9, fontweight="bold"
    )
    cbar.ax.yaxis.set_tick_params(color=COLORS["text"], labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["text"])

    # Label Axes
    ax.set_xlabel("Current Value ($r_i$)", fontsize=10, fontweight="bold", labelpad=10)
    ax.set_ylabel("Next Value ($r_{i+1}$)", fontsize=10, fontweight="bold", labelpad=10)
    ax.set_zlabel(
        "Next-Next Value ($r_{i+2}$)", fontsize=10, fontweight="bold", labelpad=10
    )

    # Limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Adjust the camera angle specifically to look "down the barrel" of the planes
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
