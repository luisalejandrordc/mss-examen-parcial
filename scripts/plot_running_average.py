# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL VALIDATION: Running Average Stabilization
# ──────────────────────────────────────────────────────────────────────────────
# Description: Calculates and plots the running average of a process variable
#              to visually identify system stabilization (warm-up phase).
# ══════════════════════════════════════════════════════════════════════════════
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def main():
    # ══════════════════════════════════════════════════════════════════════════
    # 1. Manage Data
    # ══════════════════════════════════════════════════════════════════════════
    LOAD = False  # Toggle to load existing CSV data or generate randomly

    # Variables for mining_simulation.csv: "interarrival_time", "crushing_time", "mineral_load"
    # Variables for processing_time_x.csv: "processing_time"
    VARIABLE = "processing_time"

    if LOAD:
        # ── Load existing data ────────────────────────────────────────────────
        SCRIPT_DIR = Path(__file__).resolve().parent
        csv_path = SCRIPT_DIR.parent / "data" / "processing_time_1.csv"

        df = pd.read_csv(csv_path)
        data = df[VARIABLE].to_numpy(dtype=float)
        mu = None
    else:
        # ── Generate random exponential-like data ─────────────────────────────
        np.random.seed(123)
        r = np.random.rand(100)
        mu = 12  # Theoretical mean
        data = -mu * np.log(1 - r)
        data = data.round(3)

    # ── Compute running average ───────────────────────────────────────────────
    running_avg = np.cumsum(data) / np.arange(1, len(data) + 1)
    idx = np.arange(len(data))
    grand_mean = float(np.mean(data))

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Plotting Setup
    # ══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 6))

    # Main running average line
    ax.plot(idx, running_avg, color=COLORS["blue"], lw=2, label="Running Average")

    # Grand Mean reference line to show convergence
    ax.axhline(
        grand_mean,
        color=COLORS["green"],
        lw=1.5,
        linestyle="--",
        alpha=0.6,
        label=f"Grand Mean ($\\mu \\approx {grand_mean:.2f}$)",
    )

    # Theoretical Mean (only when data was randomly generated)
    if mu:
        ax.axhline(
            mu,
            color=COLORS["red"],
            lw=1.5,
            linestyle="--",
            alpha=0.6,
            label=f"Theoretical Mean ($\\mu \\approx {mu}$)",
        )

    # Transient State: 30% - Steady State: 70%
    plt.axvspan(0, 30, alpha=0.15, color=COLORS["blue"], label="Transient State")
    plt.axvspan(30, 100, alpha=0.05, color=COLORS["blue"], label="Steady State")

    # Axis formatting
    ax.grid(True)
    ax.set_xlabel(
        "Observation Index ($i$)", fontsize=10, fontweight="bold", labelpad=10
    )
    ax.set_ylabel("Running Average", fontsize=10, fontweight="bold", labelpad=10)

    # Title styling
    ax.set_title(
        "Crusher Prossesing Time Stabilization Curve",
        color=COLORS["gold"],
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Legend
    ax.legend(
        loc="upper right",
        fontsize=10,
        facecolor=COLORS["panel"],
        edgecolor=COLORS["grid"],
        labelcolor=COLORS["text"],
    )

    # ── Render ────────────────────────────────────────────────────────────────
    plt.show()


if __name__ == "__main__":
    main()
