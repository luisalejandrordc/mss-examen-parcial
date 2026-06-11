# ════════════════════════════════════════════════════════════════════
# Linear Congruential Generator (LCG)
# It is one of the oldest most widely used pseudorandom
# number generators (PRNG) in computer science.
# ════════════════════════════════════════════════════════════════════

import time

import numpy as np

# ── Parameters ──────────────────────────────────────────────────────
seed = 55
a = 37
c = 19
g = 7
m = 2**g

# ── Algorithm ───────────────────────────────────────────────────────
start = time.time()
x_i = seed
seen = set()
sequence = np.empty(m)
n = 0
while x_i not in seen:
    seen.add(x_i)
    x_i = (a * x_i + c) % m
    r_i = round(x_i / (m - 1), 4)
    sequence[n] = r_i
    n += 1
sequence = sequence[:n]
end = time.time()

# ── Results ─────────────────────────────────────────────────────────
print(sequence)
print()
print("=" * 34)
print(f"  {'Statistic':<20} {'Value':>8}")
print("=" * 34)
print(f"  {'Count':<20} {n:>8}")
print(f"  {'Mean':<20} {sequence.mean():>8.4f}")
print(f"  {'Minimum':<20} {sequence.min():>8.4f}")
print(f"  {'Maximum':<20} {sequence.max():>8.4f}")
print(f"  {'Variation':<20} {sequence.var():>8.4f}")
print(f"  {'Std. Deviation':<20} {sequence.std():>8.4f}")
print("=" * 34)
print()
print(f"Elapsed Time: {(end-start)/1000:.8f} ms")
