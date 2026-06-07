"""
Warm-up Point Detection Methods for Steady-State Analysis
Compares five methods for identifying the warm-up period in simulation output.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

# ── Load data ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
csv_path = SCRIPT_DIR.parent / "data" / "data_2.csv"

df = pd.read_csv(csv_path, header=None)
n = len(df)
data = df[0].to_numpy(dtype=float)

# ── Running statistics ─────────────────────────────────────────────────────────
running_avg = np.array([data[: i + 1].mean() for i in range(n)])
running_std = np.array([data[: i + 1].std(ddof=1) if i > 0 else 0.0 for i in range(n)])
ci_margin = 1.96 * running_std / np.sqrt(np.arange(1, n + 1))


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — Relative Change Criterion
# ══════════════════════════════════════════════════════════════════════════════
def method_relative_change(running_avg, threshold=0.015, window=None):
    """
    Warm-up ends at the first index i such that all relative changes
    in [i, i+window) are below `threshold`.

    relative_change[i] = |avg[i] - avg[i-1]| / avg[i-1]

    Pros:  Simple, intuitive, easy to tune via threshold.
    Cons:  Sensitive to the threshold value; may trigger late if a single
           spike resets the window.
    """
    if window is None:
        window = max(5, n // 10)

    rel_change = np.abs(np.diff(running_avg) / running_avg[:-1])
    rel_change = np.concatenate([[np.nan], rel_change])  # Pad with NaN at position 0

    warmup = n - 1  # default: never detected
    for i in range(1, n - window):
        if np.all(rel_change[i : i + window] < threshold):
            warmup = i
            break
    return warmup, rel_change


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — Welch's Graphical Method
# ══════════════════════════════════════════════════════════════════════════════
def method_welch(data, smoothing_window=None, stability_window=None):
    """
    Welch (1983): smooth the running average with a moving average of width `w`,
    then find where the smoothed curve stops oscillating.

    Here we detect the warm-up as the first point where the smoothed curve
    enters a band of ±5% around the grand mean and stays there.

    Pros:  Standard academic reference; visually clear.
    Cons:  Window size w must be chosen manually; purely graphical in origin.
    """
    grand_mean = data.mean()
    if smoothing_window is None:
        smoothing_window = max(5, n // 20)  # Controls noise reduction
    if stability_window is None:
        stability_window = max(5, n // 10)  # Controls persistence requirement

    # Smooth the running average with a centered moving window
    smoothed = np.asarray(
        pd.Series(running_avg)
        .rolling(window=smoothing_window, center=True, min_periods=1)
        .mean()
    )

    band = 0.05 * grand_mean
    warmup = n - 1  # default: never detected
    for i in range(n - stability_window):
        segment = smoothed[i : i + stability_window]
        if np.all(np.abs(segment - grand_mean) < band):
            warmup = i
            break
    return warmup, smoothed, grand_mean, band


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3 — Confidence Interval Width Stabilization
# ══════════════════════════════════════════════════════════════════════════════
def method_ci_width(running_std, threshold=0.02, window=None):
    """
    Detects warm-up as the point where the 95% CI width stops shrinking
    by more than `threshold` between consecutive observations.

    CI width = 2 * 1.96 * std / sqrt(n)

    The relative reduction in width:
        delta_w[i] = (w[i-1] - w[i]) / w[i-1]

    Warm-up ends when delta_w < threshold for a sustained window.

    Pros:  Statistically grounded; directly linked to estimation precision.
    Cons:  CI width always shrinks with n (law of large numbers), so the
           threshold must be calibrated carefully.
    """
    if window is None:
        window = max(5, n // 10)  # Controls persistence requirement

    widths = 2 * 1.96 * running_std / np.sqrt(np.arange(1, n + 1))
    widths[0] = widths[1] if len(widths) > 1 else 0  # avoid div/0

    rel_reduction = np.abs(np.diff(widths) / widths[:-1])
    rel_reduction = np.concatenate([[np.nan], rel_reduction])

    warmup = n - 1  # default: never detected
    for i in range(1, n - window):
        if np.all(rel_reduction[i : i + window] < threshold):
            warmup = i
            break
    return warmup, widths, rel_reduction


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 4 — FORWARD CUSUM (Argmax Peak Detection)
# ══════════════════════════════════════════════════════════════════════════════
def method_forward_cusum(data, k_factor=0.5):
    """
    CUSUM tracks deviations of individual observations from a target (here,
    the grand mean mu_0).  Two one-sided statistics are maintained:

        C_plus[i]  = max(0, C_plus[i-1]  + (x[i] - mu_0) - k*sigma)
        C_minus[i] = max(0, C_minus[i-1] - (x[i] - mu_0) - k*sigma)

    where k = k_factor * sigma is the allowance (slack).

    Mechanism for Warm-up Detection:
        Standard Tabular CUSUM is designed for forward-looking process
        monitoring (detecting when a stable process breaks). When applied
        retrospectively to simulation warm-up, it suffers from "CUSUM Inertia"
        or the "drain-out effect."

        If we define the warm-up end as the LAST index where a signal fires
        (exceeds threshold h = 4*sigma), we overestimate the warm-up period
        because the CUSUM statistic requires time to "drain" back to zero
        after the transient phase ends.

        Instead, the exact moment the transient state ends and steady state
        begins is mathematically located at the PEAK (argmax) of the CUSUM
        chart. This is the precise transition point where accumulated
        transient errors stop growing and begin to diminish.

    Pros: Very sensitive to sustained mean shifts; avoids the drain-out lag effect.
    Cons:  Requires estimating mu_0 and sigma (chicken-and-egg problem);
           we use the second half of data as a proxy for the stable state.
    """
    stable_half = data[n // 2 :]
    mu0 = stable_half.mean()
    sigma = stable_half.std(ddof=1)
    k = k_factor * sigma
    h = 4 * sigma  # decision limit

    C_plus = np.zeros(n)
    C_minus = np.zeros(n)
    for i in range(1, n):
        C_plus[i] = max(0, C_plus[i - 1] + (data[i] - mu0) - k)
        C_minus[i] = max(0, C_minus[i - 1] - (data[i] - mu0) - k)

    max_plus = np.max(C_plus)
    max_minus = np.max(C_minus)

    warmup = 0
    if max_plus > h or max_minus > h:
        if max_plus > max_minus:
            warmup = int(np.argmax(C_plus))
        else:
            warmup = int(np.argmax(C_minus))

    return warmup, C_plus, C_minus, h, mu0, sigma


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 5 — BACKWARD CUSUM (Reverse Cumulative Sum)
# ══════════════════════════════════════════════════════════════════════════════
def method_backward_cusum(data, k_factor=0.5):
    """
    By reading the timeline backwards, the simulation starts in a stable state
    and the initial warm-up transient becomes a sudden "break" at the end of the
    reversed array.

    Mechanism:
        1. Reverses the data array.
        2. Calibrates stable parameters (mu0, sigma) using the FIRST half of
           the reversed array (which is the stable LAST half of the original).
        3. Calculates standard C_plus and C_minus statistics.
        4. Finds the FIRST index where a threshold (h) is crossed.
        5. Maps this reversed break-point back to the original timeline.
    """
    rev_data = data[::-1]
    stable_half = rev_data[: n // 2]
    mu0 = stable_half.mean()
    sigma = stable_half.std(ddof=1)
    k = k_factor * sigma
    h = 4 * sigma  # decision limit

    C_plus = np.zeros(n)
    C_minus = np.zeros(n)
    for i in range(1, n):
        C_plus[i] = max(0, C_plus[i - 1] + (rev_data[i] - mu0) - k)
        C_minus[i] = max(0, C_minus[i - 1] - (rev_data[i] - mu0) - k)

    signals = np.where((C_plus > h) | (C_minus > h))[0]
    warmup = n - int(signals[0]) if len(signals) > 0 else 0

    return warmup, C_plus[::-1], C_minus[::-1], h, mu0, sigma


# ══════════════════════════════════════════════════════════════════════════════
# Run all methods
# ══════════════════════════════════════════════════════════════════════════════
wu1, rel_change = method_relative_change(running_avg)
wu2, smoothed, gm, band = method_welch(data)
wu3, ci_widths, rel_red = method_ci_width(running_std)
wu4, f_Cp, f_Cm, f_h, f_mu0, f_sig = method_forward_cusum(data)
wu5, b_Cp, b_Cm, b_h, b_mu0, b_sig = method_backward_cusum(data)

print("=" * 55)
print(f"  {'Method':<35} {'Warm-up':>8}")
print("=" * 55)
print(f"  {'1. Relative Change (1.5%, w=10)':<35} {wu1:>8}")
print(f"  {'2. Welch Graphical (±5% band)':<35} {wu2:>8}")
print(f"  {'3. CI Width Stabilization':<35} {wu3:>8}")
print(f"  {'4. Forward CUSUM (k=0.5σ, h=4σ)':<35} {wu4:>8}")
print(f"  {'5. Backward CUSUM (k=0.5σ, h=4σ)':<35} {wu4:>8}")
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

fig = plt.figure(figsize=(16, 14))
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
ax1.fill_between(
    idx,
    running_avg - ci_margin,
    running_avg + ci_margin,
    color=COLORS["avg"],
    alpha=0.08,
)
ax1.axvline(wu1, color=COLORS["m1"], lw=2, linestyle="--", label=f"Warm-up = {wu1}")
ax1b = ax1.twinx()
ax1b.plot(
    idx,
    rel_change * 100,
    color=COLORS["m1"],
    lw=0.8,
    alpha=0.6,
    linestyle=":",
    label="Rel. change %",
)
ax1b.axhline(1.5, color=COLORS["m1"], lw=0.6, linestyle="-.", alpha=0.5)
ax1b.tick_params(colors="#abb2bf", labelsize=7)
ax1b.set_ylabel("Rel. change (%)", color=COLORS["m1"], fontsize=7)
ax1b.set_ylim(0, 50)
ax1.set_ylabel("Running avg", color="#abb2bf", fontsize=8)
ax1.legend(
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
ax2.plot(idx, smoothed, color=COLORS["m2"], lw=2, label="Smoothed avg (Welch)")
ax2.axhline(
    gm,
    color=COLORS["m2"],
    lw=0.8,
    linestyle="--",
    alpha=0.6,
    label=f"Grand mean={gm:.3f}",
)
ax2.fill_between(
    idx,
    gm - band,
    gm + band,
    color=COLORS["m2"],
    alpha=0.08,
    label="±5% stability band",
)
ax2.axvline(wu2, color=COLORS["warmup"], lw=2, linestyle="--", label=f"Warm-up = {wu2}")
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
    running_avg - ci_margin,
    running_avg + ci_margin,
    color=COLORS["m3"],
    alpha=0.15,
    label="95% CI",
)
ax3.axvline(wu3, color=COLORS["m3"], lw=2, linestyle="--", label=f"Warm-up = {wu3}")
ax3b = ax3.twinx()
ax3b.plot(
    idx, ci_widths, color=COLORS["m3"], lw=1, linestyle=":", alpha=0.7, label="CI width"
)
ax3b.tick_params(colors="#abb2bf", labelsize=7)
ax3b.set_ylabel("CI width", color=COLORS["m3"], fontsize=7)
ax3.set_ylabel("Running avg", color="#abb2bf", fontsize=8)
ax3.legend(
    loc="upper right",
    fontsize=7,
    facecolor=PANEL_BG,
    labelcolor="#abb2bf",
    edgecolor="#3e4451",
)

# ── Panel 4: Forward CUSUM ────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Method 4 — Forward CUSUM Control Chart", COLORS["m4"])
ax4.plot(idx, f_Cp, color=COLORS["m4"], lw=1.5, label="C⁺ (upper)")
ax4.plot(idx, f_Cm, color="#e06c75", lw=1.5, linestyle="--", label="C⁻ (lower)")
ax4.axhline(
    f_h,
    color=COLORS["m4"],
    lw=1,
    linestyle="-.",
    alpha=0.7,
    label=f"Decision limit h={f_h:.2f}",
)
ax4.axvline(wu4, color=COLORS["warmup"], lw=2, linestyle="--", label=f"Warm-up = {wu4}")
ax4.set_ylabel("CUSUM statistic", color="#abb2bf", fontsize=8)
ax4.legend(
    loc="upper right",
    fontsize=7,
    facecolor=PANEL_BG,
    labelcolor="#abb2bf",
    edgecolor="#3e4451",
)
ax4.text(
    0.02,
    0.08,
    f"μ₀={f_mu0:.3f}  σ={f_sig:.3f}",
    transform=ax4.transAxes,
    color="#abb2bf",
    fontsize=7,
)

# ── Panel 5: Backward CUSUM ────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, "Method 5 — Backward CUSUM Control Chart", COLORS["m5"])
ax5.plot(idx, b_Cp, color=COLORS["m5"], lw=1.5, label="C⁺ (upper)")
ax5.plot(idx, b_Cm, color="#e06c75", lw=1.5, linestyle="--", label="C⁻ (lower)")
ax5.axhline(
    b_h,
    color=COLORS["m5"],
    lw=1,
    linestyle="-.",
    alpha=0.7,
    label=f"Decision limit h={b_h:.2f}",
)
ax5.axvline(wu5, color=COLORS["warmup"], lw=2, linestyle="--", label=f"Warm-up = {wu5}")
ax5.set_ylabel("CUSUM statistic", color="#abb2bf", fontsize=8)
ax5.legend(
    loc="upper right",
    fontsize=7,
    facecolor=PANEL_BG,
    labelcolor="#abb2bf",
    edgecolor="#3e4451",
)
ax5.text(
    0.02,
    0.08,
    f"μ₀={b_mu0:.3f}  σ={b_sig:.3f}",
    transform=ax5.transAxes,
    color="#abb2bf",
    fontsize=7,
)

# ── Panel 6: Summary comparison ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, "Summary — All Methods on Running Average", "#e8d5a3")
ax6.plot(idx, running_avg, color=COLORS["avg"], lw=1.5, zorder=1, label="Running avg")
ax6.fill_between(
    idx,
    running_avg - ci_margin,
    running_avg + ci_margin,
    color=COLORS["avg"],
    alpha=0.06,
)

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
    ax6.axvline(wu, color=col, lw=1.5, linestyle="--", alpha=0.85, label=f"{lbl}={wu}")

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
