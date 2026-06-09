# ══════════════════════════════════════════════════════════════════════════════
# Calculate and plot the running average of a process variable
# ══════════════════════════════════════════════════════════════════════════════

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Variables for mining_simulation.csv: "interarrival_time", "crushing_time", "mineral_load"
# Variables for processing_time_x.csv: "processing_time"
VARIABLE = "processing_time"

# ── Load data ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
csv_path = SCRIPT_DIR.parent / "data" / "processing_time_1.csv"

df = pd.read_csv(csv_path)
data = df[VARIABLE].to_numpy(dtype=float)
running_avg = np.cumsum(data) / np.arange(1, len(data) + 1)
idx = np.arange(len(data))

# ── Plot data ──────────────────────────────────────────────────────────────────
COLORS = {
    "m1": "#e06c75",  # red
    "m2": "#61afef",  # blue
    "m3": "#98c379",  # green
    "m4": "#e5c07b",  # yellow
    "m5": "#c678dd",  # purple
    "avg": "#abb2bf",
    "panel": "#282c34",
}

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#1e2127")

ax.plot(idx, running_avg, color=COLORS["m2"], lw=1.5, label="Running avg")

ax.set_facecolor(COLORS["panel"])
ax.tick_params(colors="#abb2bf", labelsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor("#3e4451")
ax.grid(True, alpha=0.15, color="#abb2bf")
ax.set_xlabel("Observation", color="#abb2bf", fontsize=10)
ax.set_ylabel("Running average", color="#abb2bf", fontsize=10)
ax.legend(
    loc="upper right",
    fontsize=10,
    facecolor=COLORS["panel"],
    labelcolor="#abb2bf",
    edgecolor="#3e4451",
)

ax.set_title(
    "Crusher Processing Time Stabilization Curve",
    color="#e8d5a3",
    fontsize=14,
    fontweight="bold",
    y=1.04,
)

plt.show()
