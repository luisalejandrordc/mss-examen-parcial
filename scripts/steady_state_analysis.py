from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
csv_path = SCRIPT_DIR.parent / "data" / "data_1.csv"

df = pd.read_csv(csv_path, header=None)
df.columns = ["data"]
df["running_avg"] = df["data"].expanding().mean()
# data["running_avg"] = data["data"].cumsum() / (data.index + 1) # alternative way
df["running_std"] = df["data"].expanding().std()

# print(df)

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


# Plotting Running Average and Relative Change
fig, ax1 = plt.subplots(figsize=(10, 5))

# Left axis: running average
ax1.plot(df["running_avg"], color="blue", label="Running Average")
ax1.axvline(x=warm_up, color="red", linestyle="--", linewidth=1, label="Warm-up Point")
ax1.fill_between(
    range(len(df)),
    lower,
    upper,
    color="blue",
    alpha=0.05,
    label="95% Confidence Interval",
)
ax1.set_xlabel("Observation")
ax1.set_ylabel("Running Average")
ax1.grid(True)

# Right axis: relative change %
ax2 = ax1.twinx()
ax2.plot(
    relative_change * 100, color="green", linestyle="--", label="Relative Change (%)"
)
ax2.set_ylabel("Relative Change (%)")

# Forcing both y-axes limits
ax1.set_ylim(bottom=0, top=1)
ax2.set_ylim(bottom=0, top=50)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

# Align tick positions
from matplotlib.ticker import LinearLocator

ax1.yaxis.set_major_locator(LinearLocator(11))
ax2.yaxis.set_major_locator(LinearLocator(11))

ax1.grid(True, alpha=0.3)
plt.title("Steady State Analysis")
plt.show()


# Plotting Running Average only
# plt.figure(figsize=(10, 5))
# plt.plot(df["running_avg"], color="blue", label="Running Mean")
# plt.axvline(x=warm_up, color="red", linestyle="--", linewidth=1, label="Warm-up Point")
# plt.fill_between(
#     range(len(df)), lower, upper, alpha=0.3, label="95% Confidence Interval"
# )
# plt.xlabel("Observation")
# plt.ylabel("Mean")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.title("Steady State Analysis")
# plt.show()
