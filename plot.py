import matplotlib.pyplot as plt
from collections import namedtuple

from math import ceil

Data = namedtuple("Data", ["x", "y", "label"])

names = [("error", r"Restfehler $E_D$"), ("norm_gradient_w",
                                          "Parameter Gradient ||∇w||"), ]  # ("learning_rate", r"Lern Rate $\eta$")]
vals = {}
for (name, label) in names:
    with open(f"{name}.txt", "r") as f:
        raw = f.readlines()

    y = [float(a) for a in raw if a not in ("", "Inf", "NaN")]
    x = list(range(len(y)))
    vals[name] = Data(x, y, label)

cols = 2
rows = ceil(3 / 2)
for (i, d) in enumerate(vals.values()):
    plt.subplot(rows, cols, i+1)
    plt.semilogx(d.x, d.y, label=d.label)
    plt.xlabel("Iteration")
    plt.ylabel(d.label)
    plt.legend()
    plt.grid()
"""
plt.subplot(rows, cols, rows*cols)
d1 = vals["learning_rate"]
d2 = vals["norm_gradient_w"]
label = r"Lern Rate $ \cdot ||\nabla w||$"
plt.semilogx(d1.x, list(
    map(lambda x: x[0]*x[1], zip(d1.y, d2.y))), label=label)
plt.xlabel("Iteration")
plt.ylabel(label)
plt.legend()"""

plt.show()
