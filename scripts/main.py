from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
csv_path = SCRIPT_DIR.parent / "data" / "data_1.csv"

df = pd.read_csv(csv_path, header=None)
df.columns = ["data"]
df["running_avg"] = df["data"].expanding().mean()
# data["running_avg"] = data[0].cumsum() / (data.index + 1) # alternative way
df["running_std"] = df["data"].expanding().std()

print(df)

# Determining the point where the process change
# from a transient state to a steady state
relative_change = abs(df["running_avg"].diff() / df["running_avg"].shift(1))
THRESHOLD = 0.015  # 1.5%
warm_up = 100
for i in range(len(df)):
    window = relative_change.iloc[i : i + 10]
    if (window < THRESHOLD).all():
        warm_up = i
        print("Based on the relative change criterion, steady state starts at", i)
        break

# Confidence interval stabilization
margin = 1.96 * df["running_std"] / np.sqrt(df.index + 1)
upper = df["running_avg"] + margin
lower = df["running_avg"] - margin

# Creating the plot
plt.figure(figsize=(10, 5))
plt.plot(df["running_avg"], label="Running Mean")
plt.axvline(x=warm_up, color="red", linestyle="--", linewidth=1, label="Warm-up Point")
plt.fill_between(
    range(len(df)), lower, upper, alpha=0.3, label="95% Confidence Interval"
)
plt.xlabel("Observation")
plt.ylabel("Mean")
plt.legend()
plt.grid(True)
plt.title("Graphical Analysis")
plt.show()
