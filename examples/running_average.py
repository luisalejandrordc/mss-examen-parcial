# Example using pandas

import pandas as pd

data = pd.Series([10, 12, 15, 14, 13])

running_avg = data.expanding().mean()

print(running_avg)

# Example using numpy

import numpy as np

data = np.array([10, 12, 15, 14, 13])

running_avg = np.cumsum(data) / np.arange(1, len(data) + 1)

print(running_avg)
