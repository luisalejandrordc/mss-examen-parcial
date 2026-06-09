from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class WarmupResult(Generic[T]):
    warmup: int
    info: T


@dataclass
class RelativeChange:
    rel_change: np.ndarray
    threshold: float


@dataclass
class Welch:
    smoothed: np.ndarray
    grand_mean: float
    band: float


@dataclass
class CIWidth:
    ci_margin: np.ndarray
    rel_reduction: np.ndarray
    threshold: float


@dataclass
class Cusum:
    c_plus: np.ndarray
    c_minus: np.ndarray
    h: float
