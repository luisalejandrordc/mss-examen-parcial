# ══════════════════════════════════════════════════════════════════════════════
# Steady-State Analysis: Warm-up Point Detection Methods
# Provides statistical and graphical algorithms to identify the end of the
# transient-state in simulation data, marking the transition to steady-state.
# ══════════════════════════════════════════════════════════════════════════════

from typing import Optional

import numpy as np
import pandas as pd

from .models import CIWidth, Cusum, RelativeChange, WarmupResult, Welch
from .statistics import running_average, running_standard_deviation

__all__ = [
    "method_relative_change",
    "method_welch",
    "method_ci_width",
    "method_forward_cusum",
    "method_backward_cusum",
]


def _validate_input(data: np.ndarray):
    if len(data) < 2:
        raise ValueError("data must contain at least two observations")


def method_relative_change(
    data: np.ndarray, threshold: float = 0.05, window: Optional[int] = None
) -> WarmupResult[RelativeChange]:
    """
    Detects warm-up completion when the relative change between consecutive
    running averages stays below a `threshold` for a specified `window`.
    """
    _validate_input(data)
    n = len(data)
    if window is None:
        window = max(5, n // 10)

    running_avg = running_average(data)
    rel_change = np.abs(np.diff(running_avg) / running_avg[:-1])
    rel_change = np.concatenate([[np.nan], rel_change])

    warmup = n - 1
    for i in range(1, n - window):
        if np.all(rel_change[i : i + window] < threshold):
            warmup = i
            break

    return WarmupResult(warmup, RelativeChange(rel_change, threshold))


def method_welch(
    data: np.ndarray,
    smoothing_window: Optional[int] = None,
    stability_window: Optional[int] = None,
) -> WarmupResult[Welch]:
    """
    Applies Welch's method (1983) by applying a moving average to the running
    average, identifying where the smoothed curve stabilizes within a ±5%
    band of the grand mean for a set `stability_window`.
    """
    _validate_input(data)
    n = len(data)
    if smoothing_window is None:
        smoothing_window = max(5, n // 15)
    if stability_window is None:
        stability_window = max(5, n // 10)

    grand_mean = data.mean()
    band = 0.05 * grand_mean

    running_avg = running_average(data)
    smoothed = np.asarray(
        pd.Series(running_avg)
        .rolling(window=smoothing_window, center=True, min_periods=1)
        .mean()
    )

    warmup = n - 1
    for i in range(n - stability_window):
        segment = smoothed[i : i + stability_window]
        if np.all(np.abs(segment - grand_mean) < band):
            warmup = i
            break

    return WarmupResult(warmup, Welch(smoothed, grand_mean, band))


def method_ci_width(
    data: np.ndarray, threshold: float = 0.05, window: Optional[int] = None
) -> WarmupResult[CIWidth]:
    """
    Identifies steady-state as the point where the relative reduction in the
    95% confidence interval width falls below a `threshold` for a set `window`.
    """
    _validate_input(data)
    n = len(data)
    if window is None:
        window = max(5, n // 10)

    running_std = running_standard_deviation(data)

    ci_margin = 1.96 * running_std / np.sqrt(np.arange(1, n + 1))
    widths = ci_margin * 2
    widths[0] = widths[1] if len(widths) > 1 else 0

    rel_reduction = np.abs(np.diff(widths) / widths[:-1])
    rel_reduction = np.concatenate([[np.nan], rel_reduction])

    warmup = n - 1
    for i in range(1, n - window):
        if np.all(rel_reduction[i : i + window] < threshold):
            warmup = i
            break

    return WarmupResult(warmup, CIWidth(ci_margin, rel_reduction, threshold))


def method_forward_cusum(
    data: np.ndarray, k_factor: float = 0.4
) -> WarmupResult[Cusum]:
    """
    Uses a forward Tabular CUSUM to monitor deviations from the mean.
    To avoid "drain-out" lag, the exact end of the transient state is marked
    at the global maximum (argmax) of the CUSUM chart.
    """
    _validate_input(data)
    n = len(data)
    running_avg = running_average(data)
    mu0 = running_avg.mean()
    sigma = running_avg.std(ddof=1)
    k = k_factor * sigma
    h = 4 * sigma

    c_plus = np.zeros(n)
    c_minus = np.zeros(n)
    for i in range(1, n):
        c_plus[i] = max(0, c_plus[i - 1] + (running_avg[i] - mu0) - k)
        c_minus[i] = max(0, c_minus[i - 1] - (running_avg[i] - mu0) - k)

    max_plus = np.max(c_plus)
    max_minus = np.max(c_minus)

    warmup = 0
    if max_plus > h or max_minus > h:
        if max_plus > max_minus:
            warmup = int(np.argmax(c_plus))
        else:
            warmup = int(np.argmax(c_minus))

    return WarmupResult(warmup, Cusum(c_plus, c_minus, h))


def method_backward_cusum(
    data: np.ndarray, k_factor: float = 0.4
) -> WarmupResult[Cusum]:
    """
    Analyzes the timeline in reverse. By treating the stable end of the
    simulation as the baseline, this method detects the warm-up transient
    as the first threshold breach in the reversed CUSUM array.
    """
    _validate_input(data)
    n = len(data)
    running_avg = running_average(data)[::-1]
    mu0 = running_avg.mean()
    sigma = running_avg.std(ddof=1)
    k = k_factor * sigma
    h = 4 * sigma

    c_plus = np.zeros(n)
    c_minus = np.zeros(n)
    for i in range(1, n):
        c_plus[i] = max(0, c_plus[i - 1] + (running_avg[i] - mu0) - k)
        c_minus[i] = max(0, c_minus[i - 1] - (running_avg[i] - mu0) - k)

    signals = np.where((c_plus > h) | (c_minus > h))[0]
    warmup = n - int(signals[0]) if len(signals) > 0 else 0

    return WarmupResult(warmup, Cusum(c_plus[::-1], c_minus[::-1], h))
