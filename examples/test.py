import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from prng.generators import linear_congruential_generator

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

results = linear_congruential_generator(seed=83, a=141, c=53, m=1024)

ax.scatter(results[:-2], results[1:-1], results[2:], s=15)

ax.set_xlabel(r"$r_i$")
ax.set_ylabel(r"$r_{i+1}$")
ax.set_zlabel(r"$r_{i+2}$")
ax.set_title("3D Structure of LCG")
plt.show()

import numpy as np

np.random.seed(20)
r = np.random.rand(100)
x = -10 * np.log(1 - r)
x = x.round(3)

import pandas as pd

df = pd.DataFrame(x, columns=["processing_time"])
df.to_csv("output.csv", index=False)
# np.savetxt("output.csv", x, delimiter=",", fmt="%f", header="processing_time")

# print(r)
print(x)
print(x.mean())
