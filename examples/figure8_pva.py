"""Phase-plane pairs (p-v, v-a, p-a) per axis of the figure-8: x = 3 sin(w t), y = 1.5 sin(2 w t), period 12 s.

    uv run examples/figure8_pva.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PERIOD = 12.0
W = 2 * np.pi / PERIOD
AMP = np.array([3.0, 1.5])
FREQ = np.array([1.0, 2.0])

t = np.linspace(0, PERIOD, 1000)
w = W * FREQ
phase = np.outer(t, w)
p = AMP * np.sin(phase)
v = AMP * w * np.cos(phase)
a = -AMP * w**2 * np.sin(phase)

pairs = (("p", "v", p, v), ("v", "a", v, a), ("p", "a", p, a))
fig, ax = plt.subplots(2, 3, figsize=(14, 8))
for row, axis in enumerate("xy"):
    for col, (n1, n2, X1, X2) in enumerate(pairs):
        sc = ax[row, col].scatter(X1[:, row], X2[:, row], c=t, s=3, cmap="viridis")
        ax[row, col].plot(X1[0, row], X2[0, row], "ko")
        ax[row, col].set_xlabel(f"{n1}{axis}")
        ax[row, col].set_ylabel(f"{n2}{axis}")
        ax[row, col].set_title(f"{n2}{axis} vs {n1}{axis}")
        ax[row, col].grid(alpha=0.3)
fig.subplots_adjust(hspace=0.35, wspace=0.35)
fig.colorbar(sc, ax=ax, label="t [s]  (black dot = t=0)")
out = Path("logs/figure8_pva.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120)
print(f"saved {out}")
