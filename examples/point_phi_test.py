"""Experiment: the 'path' is a single point, phi = p - p* (PointELPolicy, no energy term). What trajectory does the
drone fly from hover / from a moving start, near and far, with fast (wp 16) and slow (wp 3) poles?

    PYTHONPATH=. uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python point_phi_test.py
"""
import numpy as np

import flat_el_policy as fe
import quad_sim as qs

quad = qs.Quad()
P0 = np.array([0.0, 0.0, 5.0])
CASES = [("2 m ahead, hover start, wp 16", [2, 0, 0], [0, 0, 0], 16.0),
         ("10 m ahead, hover start, wp 16", [10, 0, 0], [0, 0, 0], 16.0),
         ("10 m ahead, hover start, wp 3", [10, 0, 0], [0, 0, 0], 3.0),
         ("10 m above, hover start, wp 3", [0, 0, 10], [0, 0, 0], 3.0),
         ("10 m below, hover start, wp 3", [0, 0, -10], [0, 0, 0], 3.0),
         ("10 m ahead, start 10 m/s sideways, wp 3", [10, 0, 0], [0, 10, 0], 3.0),
         ("10 m ahead, start 10 m/s sideways, wp 16", [10, 0, 0], [0, 10, 0], 16.0),
         ("10 m behind, start 10 m/s forward, wp 3", [-10, 0, 0], [10, 0, 0], 3.0)]
logs = []
for name, off, v0, wp in CASES:
    ps = P0 + np.array(off, float)
    pol = fe.PointELPolicy(ps, wp=wp)
    log = fe.fly(quad, pol, P0, np.array(v0, float), np.zeros(3), np.zeros(3), 8.0, mode="B")
    logs.append((name, ps, log))
    d = log[:, 1]
    done = np.where(d > 0.1)[0]
    t_arr = log[done[-1] + 1, 0] if len(done) and done[-1] + 1 < len(log) else (0.0 if not len(done) else np.nan)
    seg = ps - P0
    lam = np.clip((log[:, 4:7] - P0) @ seg / (seg @ seg), 0, 1)
    off_line = np.linalg.norm(log[:, 4:7] - P0 - lam[:, None] * seg, axis=1).max()
    tilt = np.degrees(np.arccos(np.clip(log[:, 14], -1, 1)))
    print(f"{name:42s}: finished {log[-1, 0] >= 7.99} | within 0.1 m from t = {t_arr:.2f} s | max off the straight line "
          f"{off_line:.2f} m | max tilt {tilt.max():.0f} deg, upside down {100 * (log[:, 14] < 0).mean():.1f}% | "
          f"sat {100 * log[:, 8].mean():.0f}% | max |w| {log[:, 7].max():.1f} rad/s", flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
for (name, ps, log), c in zip(logs, plt.cm.tab10(np.arange(len(logs)))):
    for ax, (i, j) in zip(axs, ((0, 1), (0, 2))):
        ax.plot(log[:, 4 + i], log[:, 4 + j], color=c, lw=1.2, label=name)
        ax.plot(ps[i], ps[j], "x", color=c, ms=9)
for ax, lab in zip(axs, (("x m", "y m"), ("x m", "z m"))):
    ax.plot(P0[0], P0[1] if lab[1] == "y m" else P0[2], "ko", ms=6)
    ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1]); ax.set_aspect("equal"); ax.grid(alpha=0.3)
axs[0].set_title("top view (start = black dot, target = x)"); axs[1].set_title("side view")
axs[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
fig.savefig("../logs/point_phi.png", dpi=120, bbox_inches="tight")
