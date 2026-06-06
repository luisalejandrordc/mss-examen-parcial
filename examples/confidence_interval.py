import numpy as np

samples = np.array([4.8, 5.1, 4.9, 5.0, 5.2])

mean = np.mean(samples)
std = np.std(samples, ddof=1)
margin = 1.96 * std / np.sqrt(len(samples))

print(f"95% CI: [{mean - margin:.3f}, {mean + margin:.3f}]")
