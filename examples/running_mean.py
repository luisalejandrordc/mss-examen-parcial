import pandas as pd

data = pd.Series([10, 12, 15, 14, 13])

running_mean = data.expanding().mean()

print(running_mean)
