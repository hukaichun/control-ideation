"""Plot one or more episode CSVs (position vs reference, commanded vs actual velocity).

    uv run tools/plot_log.py logs/a.csv [logs/b.csv ...] [-o out.png]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+", type=Path)
ap.add_argument("-o", "--out", type=Path)
args = ap.parse_args()

fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
for i, path in enumerate(args.logs):
    d = np.genfromtxt(path, delimiter=",", names=True)
    c = f"C{i}"
    for row, (ax_name, p, r, v, cmd) in enumerate(
        [("north", "n", "ref_n", "vn", "cmd_vn"),
         ("east", "e", "ref_e", "ve", "cmd_ve"),
         ("down", "d", "ref_d", "vd", "cmd_vd")]
    ):
        axes[row, 0].plot(d["t"], d[p], c, label=path.stem)
        axes[row, 0].plot(d["t"], d[r], "k--", lw=0.8, label="reference" if i == 0 else None)
        axes[row, 0].set_ylabel(f"{ax_name} [m]")
        axes[row, 1].plot(d["t"], d[v], c, label=path.stem)
        axes[row, 1].plot(d["t"], d[cmd], c, ls=":", lw=0.8, label=f"{path.stem} cmd")
        axes[row, 1].set_ylabel(f"v {ax_name} [m/s]")
    err = np.sqrt((d["n"] - d["ref_n"]) ** 2 + (d["e"] - d["ref_e"]) ** 2 + (d["d"] - d["ref_d"]) ** 2)
    print(f"{path.stem}: position RMSE {np.sqrt(np.mean(err**2)):.3f} m, max {err.max():.3f} m")

axes[0, 0].set_title("position")
axes[0, 1].set_title("velocity (dotted = command)")
axes[0, 0].legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("t [s]")
for ax in axes.flat:
    ax.grid(alpha=0.3)
fig.tight_layout()
out = args.out or args.logs[0].with_suffix(".png")
fig.savefig(out, dpi=120)
print(f"saved {out}")
