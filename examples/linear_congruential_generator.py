# ══════════════════════════════════════════════════════════════════════════════
# Linear Congruential Generator (LCG)
# It is one of the oldest most widely used pseudorandom number
# generators (PRNG) in computer science.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np

# ── Parameters ────────────────────────────────────────────────────────────────
x_0 = 55
a = 37
c = 19
m = 128

# ── Initial Conditions ────────────────────────────────────────────────────────
x_i = x_0
seen = set()
results = np.empty(m)
n = 0

while x_i not in seen:
    seen.add(x_i)
    x_i = (a * x_i + c) % m
    r_i = round(x_i / (m - 1), 4)
    results[n] = r_i
    n += 1

results = results[:n]

print("=" * 34)
print(f"  {'Statistic':<20} {'Value':>8}")
print("=" * 34)
print(f"  {'Count':<20} {n:>8}")
print(f"  {'Mean':<20} {results.mean():>8.4f}")
print(f"  {'Minimum':<20} {results.min():>8.4f}")
print(f"  {'Maximum':<20} {results.max():>8.4f}")
print(f"  {'Std Deviation':<20} {results.std():>8.4f}")
print("=" * 34)

print(results)
