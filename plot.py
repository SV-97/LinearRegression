import matplotlib.pyplot as plt
from collections import namedtuple

from math import ceil

Data = namedtuple("Data", ["x", "y", "label"])

names = [("error", "Error"), ("norm_gradient_w", "||∇w||"), ("learnrate", r"Learnrate $\eta$")]
vals = {}
for (name, label) in names:
    with open(f"{name}.txt", "r") as f:
        raw = f.readlines()

    y = [float(a) for a in raw if a not in ("", "Inf", "NaN")]
    x = list(range(len(y)))
    vals[name] = Data(x, y, label)

cols = 2
rows = ceil(len(names) / 2)
for (i, d) in enumerate(vals.values()):
    plt.subplot(rows, cols, i+1)
    plt.plot(d.x, d.y, label=d.label)
    plt.xlabel("Iterations")
    plt.ylabel(d.label)
    plt.legend()

plt.show()