# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL VALIDATION: Chi-Squared Uniformity Test
# ──────────────────────────────────────────────────────────────────────────────
# Description: Generates a sequence using an LCG and evaluates its empirical
#              distribution against a uniform distribution using a Chi-squared
#              goodness-of-fit test.
# ══════════════════════════════════════════════════════════════════════════════
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

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
    # ── 1. Generate Pseudorandom Numbers ──────────────────────────────────────
    n_samples = 100
    rs_q6 = linear_congruential_generator(seed=83, a=141, c=53, m=1024, N=n_samples)

    # ── 2. Calculate Chi-Squared Statistics ───────────────────────────────────
    m_int = int(math.sqrt(n_samples))
    expected = n_samples / m_int
    alpha = 0.05
    df = m_int - 1
    chi_crit = chi2.ppf(1 - alpha, df)

    # Define bins and calculate observed frequencies using numpy's native C-backend
    edges = np.linspace(0, 1, m_int + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    observed, _ = np.histogram(rs_q6, bins=edges)

    # Calculate contributions and total test statistic
    contribs = (expected - observed) ** 2 / expected
    chi2_0 = contribs.sum()

    # From Binomial Distribution Var(Oi) = np(1 - p) = E(1 - 1/m)
    std_dev = math.sqrt(expected * (1 - 1 / m_int))

    # ── 3. Plotting Setup ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Statistical Validation: Uniformity Test (95% Confidence Level)",
        color=COLORS["gold"],
        fontsize=14,
        fontweight="bold",
    )

    # X-axis tick labels formatted for standard mathematical interval notation
    xtick_labels = [f"[{edges[i]:.1f}, {edges[i+1]:.1f})" for i in range(m_int)]

    # ── Panel A: Histogram of Observed Frequencies ────────────────────────────
    ax1 = axes[0]
    style_axis(ax1, "A — Observed vs Expected Frequencies")

    # Highlight bars that deviate heavily from the expected value
    bar_colors = [
        (
            COLORS["blue"]
            if expected - std_dev <= o <= expected + std_dev
            else COLORS["red"]
        )
        for o in observed
    ]

    bars1 = ax1.bar(
        centers,
        observed,
        width=0.08,
        color=bar_colors,
        alpha=0.80,
        edgecolor=COLORS["panel"],
        linewidth=0.6,
        zorder=3,
    )

    # Annotate each bar with its observed value
    for bar, o_val in zip(bars1, observed):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{int(o_val)}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["text"],
        )

    # Expected frequency line
    expected_handle = ax1.axhline(
        expected,
        color=COLORS["green"],
        lw=2,
        linestyle="--",
        zorder=4,
    )

    # Tolerance band ±1 std
    tolerance_handle = ax1.axhline(
        expected + std_dev,
        color=COLORS["green"],
        lw=1,
        linestyle=":",
        alpha=0.7,
    )
    ax1.axhline(
        expected - std_dev, color=COLORS["green"], lw=1, linestyle=":", alpha=0.7
    )
    ax1.fill_between(
        [0, 1],
        expected - std_dev,
        expected + std_dev,
        color=COLORS["green"],
        alpha=0.08,
    )

    ax1.set_xlabel("Sub-interval", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=10, fontweight="bold")
    ax1.set_xticks(centers)
    ax1.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylim(0, max(observed) + 3)
    ax1.legend(
        [bars1, expected_handle, tolerance_handle],
        [
            "Observed ($O_i$)",
            f"Expected ($E_i = {expected:.2f}".rstrip("0").rstrip(".") + ")$",
            "Tolerance ($E_i \\pm \\sigma$)",
        ],
        fontsize=9,
        facecolor=COLORS["panel"],
        edgecolor=COLORS["grid"],
        labelcolor=COLORS["text"],
    )

    # ── Panel B: Per-interval Chi-Squared Contributions ───────────────────────
    ax2 = axes[1]
    style_axis(ax2, "B — Contributions to $\\chi^2_0$ Statistic")

    critical_level = chi_crit / m_int  # The theoretical fair share per interval
    contrib_colors = [
        COLORS["red"] if c > critical_level else COLORS["orange"] for c in contribs
    ]

    bars2 = ax2.bar(
        centers,
        contribs,
        width=0.08,
        color=contrib_colors,
        alpha=0.85,
        edgecolor=COLORS["panel"],
        linewidth=0.6,
        zorder=3,
    )

    # Annotate each bar with its mathematical contribution
    for bar, c_val in zip(bars2, contribs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{c_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["text"],
        )

    # Critical level line
    critical_level_handle = ax2.axhline(
        critical_level,
        color=COLORS["rose"],
        lw=1.5,
        linestyle="--",
        zorder=4,
    )

    # Observed level line
    observed_level_handle = ax2.axhline(
        chi2_0 / m_int,
        color=COLORS["purple"],
        lw=1.5,
        linestyle=":",
        alpha=0.7,
    )

    ax2.set_xlabel("Sub-interval", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Contribution", fontsize=10, fontweight="bold")
    ax2.set_xticks(centers)
    ax2.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, max(contribs.max(), critical_level) * 1.35)
    ax2.legend(
        [bars2, critical_level_handle, observed_level_handle],
        [
            "Bin Contributions $(E_i - O_i)^2 / E_i$",
            f"Critical Level ($\\chi^2_{{crit}}/m = {critical_level:.2f}$)",
            f"Observed Level ($\\chi^2_0/m = {chi2_0/m_int:.2f}$)",
        ],
        fontsize=9,
        facecolor=COLORS["panel"],
        edgecolor=COLORS["grid"],
        labelcolor=COLORS["text"],
    )

    # ── 4. Summary Results Box ────────────────────────────────────────────────
    test_passed = chi2_0 < chi_crit
    result_text = "Do not reject $H_0$" if test_passed else "Reject $H_0$"
    box_color = COLORS["green"] if test_passed else COLORS["red"]

    summary = (
        f"$\\chi^2_0 = {chi2_0:.3f}$\n"
        f"$\\chi^2_{{0.05,\\,9}} = {chi_crit:.3f}$\n"
        f"{result_text}"
    )

    ax2.text(
        0.96,
        0.96,
        summary,
        transform=ax2.transAxes,
        color=box_color,
        fontsize=9,
        va="top",
        ha="right",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=COLORS["panel"],
            edgecolor=box_color,
            alpha=0.9,
        ),
    )

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.14)
    plt.show()


if __name__ == "__main__":
    main()
