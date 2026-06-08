import numpy as np


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
