# ══════════════════════════════════════════════════════════════════════════════
# Warm-up Point Detection Methods for Steady-State Analysis
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

__all__ = [
    "method_relative_change",
    "method_welch",
    "method_ci_width",
    "method_forward_cusum",
    "method_backward_cusum",
]


@dataclass
class WarmupResult:
    """Stores Warm-up Point information"""

    warmup: int
    diagnostics: dict[str, Any]


def _validate_input(data: np.ndarray):
    if len(data) < 2:
        raise ValueError("data must contain at least two observations")


def running_average(data: np.ndarray) -> np.ndarray:
    return np.cumsum(data) / np.arange(1, len(data) + 1)


def running_standard_deviation(data: np.ndarray) -> np.ndarray:
    n = np.arange(1, len(data) + 1)
    cumsum = np.cumsum(data)
    cumsum_sq = np.cumsum(data**2)
    variance = (cumsum_sq - (cumsum**2) / n) / np.where(n == 1, 1, n - 1)
    std_dev = np.sqrt(variance)
    std_dev[0] = 0.0
    return std_dev


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — Relative Change Criterion
# ══════════════════════════════════════════════════════════════════════════════
def method_relative_change(
    data: np.ndarray, threshold: float = 0.015, window: Optional[int] = None
) -> WarmupResult:
    """
    Warm-up ends at the first index i such that all relative changes
    in [i, i+window) are below `threshold`.

    relative_change[i] = |avg[i] - avg[i-1]| / avg[i-1]

    Pros:  Simple, intuitive, easy to tune via threshold.
    Cons:  Sensitive to the threshold value; may trigger late if a single
           spike resets the window.
    """
    _validate_input(data)
    n = len(data)
    if window is None:
        window = max(5, n // 10)  # Controls persistence requirement

    running_avg = running_average(data)
    rel_change = np.abs(np.diff(running_avg) / running_avg[:-1])
    rel_change = np.concatenate([[np.nan], rel_change])  # Pad with NaN at position 0

    warmup = n - 1  # default: never detected
    for i in range(1, n - window):
        if np.all(rel_change[i : i + window] < threshold):
            warmup = i
            break

    return WarmupResult(warmup, {"rel_change": rel_change})


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — Welch's Graphical Method
# ══════════════════════════════════════════════════════════════════════════════
def method_welch(
    data: np.ndarray,
    smoothing_window: Optional[int] = None,
    stability_window: Optional[int] = None,
) -> WarmupResult:
    """
    Welch (1983): smooth the running average with a moving average of width `w`,
    then find where the smoothed curve stops oscillating.

    Here we detect the warm-up as the first point where the smoothed curve
    enters a band of ±5% around the grand mean and stays there.

    Pros:  Standard academic reference; visually clear.
    Cons:  Window size w must be chosen manually; purely graphical in origin.
    """
    _validate_input(data)
    n = len(data)
    if smoothing_window is None:
        smoothing_window = max(5, n // 20)  # Controls noise reduction
    if stability_window is None:
        stability_window = max(5, n // 10)  # Controls persistence requirement

    grand_mean = data.mean()
    band = 0.05 * grand_mean

    # Smooth the running average with a centered moving window
    running_avg = running_average(data)
    smoothed = np.asarray(
        pd.Series(running_avg)
        .rolling(window=smoothing_window, center=True, min_periods=1)
        .mean()
    )

    warmup = n - 1  # default: never detected
    for i in range(n - stability_window):
        segment = smoothed[i : i + stability_window]
        if np.all(np.abs(segment - grand_mean) < band):
            warmup = i
            break

    return WarmupResult(
        warmup, {"smoothed": smoothed, "grand_mean": grand_mean, "band": band}
    )


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3 — Confidence Interval Width Stabilization
# ══════════════════════════════════════════════════════════════════════════════
def method_ci_width(
    data: np.ndarray, threshold: float = 0.02, window: Optional[int] = None
) -> WarmupResult:
    """
    Detects warm-up as the point where the 95% CI width stops shrinking
    by more than `threshold` (relative) between consecutive observations.

    CI width = 2 * 1.96 * std / sqrt(n)

    The relative reduction in width:
        delta_w[i] = (w[i-1] - w[i]) / w[i-1]

    Warm-up ends when delta_w < threshold for a sustained window.

    Pros:  Statistically grounded; directly linked to estimation precision.
    Cons:  CI width always shrinks with n (law of large numbers), so the
           threshold must be calibrated carefully.
    """
    _validate_input(data)
    n = len(data)
    if window is None:
        window = max(5, n // 10)  # Controls persistence requirement

    running_std = running_standard_deviation(data)

    ci_margin = 1.96 * running_std / np.sqrt(np.arange(1, n + 1))
    widths = ci_margin * 2
    widths[0] = widths[1] if len(widths) > 1 else 0  # avoid div/0

    rel_reduction = np.abs(np.diff(widths) / widths[:-1])
    rel_reduction = np.concatenate([[np.nan], rel_reduction])

    warmup = n - 1
    for i in range(1, n - window):
        if np.all(rel_reduction[i : i + window] < threshold):
            warmup = i
            break

    return WarmupResult(
        warmup,
        {"ci_margin": ci_margin, "widths": widths, "rel_reduction": rel_reduction},
    )


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 4 — FORWARD CUSUM (Argmax Peak Detection)
# ══════════════════════════════════════════════════════════════════════════════
def method_forward_cusum(data: np.ndarray, k_factor: float = 0.5) -> WarmupResult:
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
    _validate_input(data)
    n = len(data)
    stable_half = data[n // 2 :]
    mu0 = stable_half.mean()
    sigma = stable_half.std(ddof=1)
    k = k_factor * sigma
    h = 4 * sigma  # heuristic decision limit

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

    return WarmupResult(warmup, {"C_plus": C_plus, "C_minus": C_minus, "h": h})


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 5 — BACKWARD CUSUM (Reverse Cumulative Sum)
# ══════════════════════════════════════════════════════════════════════════════
def method_backward_cusum(data: np.ndarray, k_factor: float = 0.5) -> WarmupResult:
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
    _validate_input(data)
    n = len(data)
    rev_data = data[::-1]
    stable_half = rev_data[: n // 2]
    mu0 = stable_half.mean()
    sigma = stable_half.std(ddof=1)
    k = k_factor * sigma
    h = 4 * sigma  # heuristic decision limit

    C_plus = np.zeros(n)
    C_minus = np.zeros(n)
    for i in range(1, n):
        C_plus[i] = max(0, C_plus[i - 1] + (rev_data[i] - mu0) - k)
        C_minus[i] = max(0, C_minus[i - 1] - (rev_data[i] - mu0) - k)

    signals = np.where((C_plus > h) | (C_minus > h))[0]
    warmup = n - int(signals[0]) if len(signals) > 0 else 0

    return WarmupResult(
        warmup, {"C_plus": C_plus[::-1], "C_minus": C_minus[::-1], "h": h}
    )
