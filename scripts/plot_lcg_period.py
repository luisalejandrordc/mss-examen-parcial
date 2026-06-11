"""
Visualization scripts for Q3 (LCG full period) and Q6 (Chi-squared uniformity).
Run from the directory containing data_1.csv.

Generates:
    plot_q3_lcg_period.png  —  Q3: sequence plot + lag-1 scatter
    plot_q6_uniformity.png  —  Q6: frequency histogram + chi^2 contributions
"""

import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ── Shared style ───────────────────────────────────────────────────────────────
BG = "#1e2127"
PANEL = "#282c34"
GRID = "#3e4451"
TEXT = "#abb2bf"
GOLD = "#e8d5a3"
BLUE = "#61afef"
GREEN = "#98c379"
RED = "#e06c75"
ORANGE = "#e5c07b"
PURPLE = "#c678dd"
TEAL = "#56b6c2"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "grid.color": GRID,
        "grid.alpha": 0.4,
        "font.family": "serif",
    }
)


def style(ax, title, color=GOLD):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.set_title(title, color=color, fontsize=10, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.25)


# ══════════════════════════════════════════════════════════════════════════════
# Q3 — LCG Full Period
# ══════════════════════════════════════════════════════════════════════════════
g, a, c, X0 = 7, 37, 19, 55
m = 2**g

X = X0
sequence = []
seen = set()
while True:
    X = (a * X + c) % m
    if X in seen:
        break
    seen.add(X)
    sequence.append(X / (m - 1))

N = len(sequence)  # should be 128 = m
idx = np.arange(1, N + 1)

fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.patch.set_facecolor(BG)
fig3.suptitle(
    f"Q3 — Período Completo del Generador ACL  "
    f"($X_0={X0},\\ a={a},\\ c={c},\\ m=2^{{{g}}}={m}$)",
    color=GOLD,
    fontsize=12,
    fontweight="bold",
    y=1.01,
)

# ── Panel A: Sequence plot ────────────────────────────────────────────────────
ax = axes[0]
style(ax, "A — Secuencia completa $r_i$ vs $i$")

# Colour points by quintile so density is visible even when they overlap
quintile = (np.array(sequence) * 5).astype(int).clip(0, 4)
colours = [BLUE, GREEN, ORANGE, RED, PURPLE]
for q, col in enumerate(colours):
    mask = quintile == q
    ax.scatter(
        idx[mask],
        np.array(sequence)[mask],
        color=col,
        s=18,
        alpha=0.85,
        zorder=3,
        label=f"$r_i \\in [{q/5:.1f}, {(q+1)/5:.1f})$",
    )

ax.axhline(0.5, color=GOLD, lw=1, linestyle="--", alpha=0.6, label="$E[r]=0.5$")
ax.set_xlabel("Iteración $i$", fontsize=9)
ax.set_ylabel("$r_i$", fontsize=9)
ax.set_xlim(0, N + 2)
ax.set_ylim(-0.04, 1.04)
ax.legend(
    fontsize=7,
    facecolor=PANEL,
    edgecolor=GRID,
    labelcolor=TEXT,
    loc="upper right",
    ncol=2,
)

# Annotate period
ax.annotate(
    f"Período completo\n$N = m = {N}$",
    xy=(N, sequence[-1]),
    xytext=(N - 30, 0.15),
    color=GOLD,
    fontsize=8,
    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1),
)

# ── Panel B: Lag-1 scatter  (ri vs r_{i+1}) ──────────────────────────────────
ax = axes[1]
style(ax, "B — Diagrama de Rezago 1:  $r_i$ vs $r_{i+1}$")

r_curr = np.array(sequence[:-1])
r_next = np.array(sequence[1:])

# Colour by position in sequence to show temporal flow
sc = ax.scatter(
    r_curr, r_next, c=np.arange(len(r_curr)), cmap="plasma", s=22, alpha=0.85, zorder=3
)

cbar = fig3.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Índice $i$", color=TEXT, fontsize=8)
cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)

ax.set_xlabel("$r_i$", fontsize=9)
ax.set_ylabel("$r_{i+1}$", fontsize=9)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")

# Reference diagonal (would indicate autocorrelation)
ax.plot(
    [0, 1],
    [0, 1],
    color=GRID,
    lw=0.8,
    linestyle=":",
    alpha=0.5,
    label="diagonal (ref.)",
)
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

ax.text(
    0.03,
    0.95,
    "Cobertura uniforme\n→ sin estructura de celosía visible",
    transform=ax.transAxes,
    color=GREEN,
    fontsize=8,
    va="top",
)

fig3.tight_layout()
# fig3.savefig(
#     "plot_q3_lcg_period.png",
#     dpi=150,
#     bbox_inches="tight",
#     facecolor=fig3.get_facecolor(),
# )
# print("Saved: plot_q3_lcg_period.png")


# ══════════════════════════════════════════════════════════════════════════════
# Q6 — Chi-squared Uniformity Test
# ══════════════════════════════════════════════════════════════════════════════
from pathlib import Path

import pandas as pd
from scipy import stats

csv_path = Path(__file__).resolve().parent.parent / "data" / "data_1.csv"
rs_q6 = pd.read_csv(csv_path).to_numpy()
n = len(rs_q6)

m_int = int(math.sqrt(n))  # 10 sub-intervals
E = n / m_int  # expected per interval = 10
alpha = 0.05
df = m_int - 1  # degrees of freedom = 9
chi_crit = stats.chi2.ppf(1 - alpha, df)  # 16.919

# Observed frequencies and contributions
edges = np.linspace(0, 1, m_int + 1)
centers = (edges[:-1] + edges[1:]) / 2
observed = np.array(
    [
        np.sum(
            (rs_q6 >= edges[i])
            & (rs_q6 < edges[i + 1] if i < m_int - 1 else rs_q6 <= edges[i + 1])
        )
        for i in range(m_int)
    ],
    dtype=float,
)
contribs = (E - observed) ** 2 / E
chi2_0 = contribs.sum()

fig6, axes = plt.subplots(1, 2, figsize=(14, 5))
fig6.patch.set_facecolor(BG)
fig6.suptitle(
    f"Q6 — Prueba Chi-Cuadrada de Uniformidad  "
    f"($n={n},\\ m={m_int},\\ \\alpha={alpha}$)",
    color=GOLD,
    fontsize=12,
    fontweight="bold",
    y=1.01,
)

# ── Panel A: Histogram of observed frequencies ───────────────────────────────
ax = axes[0]
style(ax, "A — Frecuencias Observadas vs Esperadas")

bar_colors = [RED if abs(O - E) > 3 else BLUE for O in observed]
bars = ax.bar(
    centers,
    observed,
    width=0.09,
    color=bar_colors,
    alpha=0.80,
    edgecolor=PANEL,
    linewidth=0.6,
    zorder=3,
    label="$O_i$ (observada)",
)

# Expected frequency line
ax.axhline(
    E, color=GREEN, lw=2, linestyle="--", zorder=4, label=f"$E_i = n/m = {E:.0f}$"
)

# Tolerance band ±1 std under Poisson approximation (sqrt(E))
ax.axhline(
    E + math.sqrt(E),
    color=GREEN,
    lw=0.8,
    linestyle=":",
    alpha=0.5,
    label=f"$E_i \\pm \\sqrt{{E_i}}$",
)
ax.axhline(E - math.sqrt(E), color=GREEN, lw=0.8, linestyle=":", alpha=0.5)
ax.fill_between([0, 1], E - math.sqrt(E), E + math.sqrt(E), color=GREEN, alpha=0.05)

# Annotate each bar with Oi value
for bar, O in zip(bars, observed):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"{int(O)}",
        ha="center",
        va="bottom",
        fontsize=8,
        color=TEXT,
    )

ax.set_xlabel("Sub-intervalo", fontsize=9)
ax.set_ylabel("Frecuencia", fontsize=9)
ax.set_xticks(centers)
ax.set_xticklabels(
    [f"[{e:.1f},{edges[i+1]:.1f})" for i, e in enumerate(edges[:-1])],
    rotation=45,
    ha="right",
    fontsize=7,
)
ax.set_ylim(0, max(observed) + 3)
ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

# ── Panel B: Per-interval chi^2 contributions ────────────────────────────────
ax = axes[1]
style(ax, "B — Contribuciones al Estadístico $\\chi^2_0$")

contrib_colors = [RED if c > chi_crit / m_int else ORANGE for c in contribs]
bars2 = ax.bar(
    centers,
    contribs,
    width=0.09,
    color=contrib_colors,
    alpha=0.85,
    edgecolor=PANEL,
    linewidth=0.6,
    zorder=3,
    label="$(E_i - O_i)^2 / E_i$",
)

# Budget line: fair share per interval
budget = chi_crit / m_int
ax.axhline(
    budget,
    color=GOLD,
    lw=1.5,
    linestyle="--",
    zorder=4,
    label=f"$\\chi^2_{{crit}}/m = {budget:.3f}$ (cuota por intervalo)",
)

# Total chi^2_0 annotation
ax.axhline(
    chi2_0 / m_int,
    color=RED,
    lw=1.2,
    linestyle=":",
    alpha=0.7,
    label=f"$\\chi^2_0/m = {chi2_0/m_int:.3f}$ (promedio real)",
)

# Annotate each bar
for bar, c in zip(bars2, contribs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{c:.2f}",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=TEXT,
    )

ax.set_xlabel("Sub-intervalo", fontsize=9)
ax.set_ylabel("$(E_i - O_i)^2 / E_i$", fontsize=9)
ax.set_xticks(centers)
ax.set_xticklabels(
    [f"[{e:.1f},{edges[i+1]:.1f})" for i, e in enumerate(edges[:-1])],
    rotation=45,
    ha="right",
    fontsize=7,
)
ax.set_ylim(0, max(contribs) * 1.35)
ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

# Summary box
summary = (
    f"$\\chi^2_0 = {chi2_0:.3f}$\n"
    f"$\\chi^2_{{0.05,\\,9}} = {chi_crit:.3f}$\n"
    f"$\\chi^2_0 < \\chi^2_{{crit}}$ → No rechazar $H_0$ ✓"
)
ax.text(
    0.97,
    0.97,
    summary,
    transform=ax.transAxes,
    color=GREEN,
    fontsize=8.5,
    va="top",
    ha="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL, edgecolor=GREEN, alpha=0.9),
)

fig6.tight_layout()
# fig6.savefig(
#     "plot_q6_uniformity.png",
#     dpi=150,
#     bbox_inches="tight",
#     facecolor=fig6.get_facecolor(),
# )
# print("Saved: plot_q6_uniformity.png")
plt.show()
