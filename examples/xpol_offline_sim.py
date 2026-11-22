"""Run .xpol images offline on the flat integrator chain, the same integration ext_policy does.

    uv run examples/xpol_offline_sim.py build/policies/hold.xpol build/policies/circle.xpol
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uavlab.xpol import Policy

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="+", type=Path)
ap.add_argument("--duration", type=float, default=40.0)
args = ap.parse_args()

DT = 0.01
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for i, path in enumerate(args.images):
    pol = Policy(path)
    x = np.zeros(12)
    x[2] = -5.0                      # start hovering at the origin, 5 m up
    xs = []
    for _ in range(int(args.duration / DT)):
        u = pol(x)
        x[9:12] += u * DT            # same order as ext_policy: snap -> jerk -> acc
        x[6:9] += x[9:12] * DT
        x[3:6] += x[6:9] * DT
        x[0:3] += x[3:6] * DT
        xs.append(x.copy())
    xs = np.array(xs)
    t = np.arange(len(xs)) * DT
    r = np.hypot(xs[:, 0], xs[:, 1])
    print(f"{pol.name} (crc 0x{pol.crc:08x}): final p = {np.round(xs[-1, :3], 3)}, "
          f"radius last 10 s {r[t > args.duration - 10].min():.3f}..{r[t > args.duration - 10].max():.3f} m, "
          f"max |a| {np.linalg.norm(xs[:, 6:9], axis=1).max():.2f} m/s^2")
    ax[0].plot(xs[:, 1], xs[:, 0], f"C{i}", label=pol.name)
    ax[1].plot(t, xs[:, 2], f"C{i}", label=f"{pol.name} z")
ax[0].set_xlabel("east [m]")
ax[0].set_ylabel("north [m]")
ax[0].set_aspect("equal")
ax[0].legend()
ax[1].set_xlabel("t [s]")
ax[1].set_ylabel("down [m]")
ax[1].legend()
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
out = Path("logs/xpol_offline_sim.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110)
print(f"saved {out}")
