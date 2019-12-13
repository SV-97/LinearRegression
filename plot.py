import matplotlib.pyplot as plt
from collections import namedtuple

from math import ceil

Data = namedtuple("Data", ["x", "y", "label"])

names = [("error", r"Error $E_D$"), ("norm_gradient_w", "Parameter Gradient ||∇w||"), ("learnrate", r"Learning Rate $\eta$")]
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
    plt.semilogy(d.x, d.y, label=d.label)
    plt.xlabel("Iteration")
    plt.ylabel(d.label)
    plt.legend()

plt.subplot(rows, cols, rows*cols)
d1 = vals["learnrate"]
d2 = vals["norm_gradient_w"]
label = r"Learning rate $ \cdot ||\nabla w||$"
plt.semilogy(d1.x, list(map(lambda x: x[0]*x[1], zip(d1.y, d2.y))), label=label)
plt.xlabel("Iteration")
plt.ylabel(label)
plt.legend()

plt.show()