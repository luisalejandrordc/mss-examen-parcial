# Example using pandas

import pandas as pd

data = pd.Series([10, 12, 15, 14, 13])

running_avg = data.expanding().mean()

print(running_avg)

# Example using numpy

import numpy as np

np.random.seed(123)
r = np.random.rand(100)
data = -12 * np.log(1 - r)
data = data.round(3)

running_avg = np.cumsum(data) / np.arange(1, len(data) + 1)
running_avg = running_avg.round(3)

print(data)
print(running_avg)
