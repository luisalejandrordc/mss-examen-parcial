# ══════════════════════════════════════════════════════════════════════════════
# PseudoRandom Number Generators (PRNG)

# Non-Congruential Methods:
#   - Middle-Square Method
#   - Middle-Product Method
#   - Constant Multiplier Method

# Congruential Methods:
#   Linear:
#       - Linear Congruential Generator (LCG)
#       - Multiplicative Congruential Generator
#       - Additive Congruential Generator
#   Non-Linear:
#       - Quadratic Congruential Generator
#       - Blum Blum Shub Generator (Cryptographic PRNG)
# ══════════════════════════════════════════════════════════════════════════════

from typing import Optional

import numpy as np


def linear_congruential_generator(
    seed: int, a: int, c: int, m: int, N: Optional[int] = None, precision: int = 4
) -> np.ndarray:
    if N is None:
        N = m
    x_i = seed
    seen = set()
    sequence = np.empty(m)
    n = 0
    while x_i not in seen and n < N:
        seen.add(x_i)
        x_i = (a * x_i + c) % m
        r_i = round(x_i / (m - 1), precision)
        sequence[n] = r_i
        n += 1
    return sequence[:n]
