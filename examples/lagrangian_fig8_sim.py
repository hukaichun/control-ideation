"""Simulate the state feedback u(x) = -5 w^2 a - 4 w^4 p (same law on both axes).

Flat coordinates: each axis is a chain of integrators, state [p, v, a, j], input u = snap.
Started on the figure-8 x = 3 sin(w t), y = 1.5 sin(2 w t), it should reproduce it exactly.

    uv run examples/lagrangian_fig8_sim.py
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
DT = 0.002
T_END = 3 * PERIOD


def ref(t):
    """Planned figure-8: derivatives 0..4 -> (..., 5, 2)."""
    w = W * FREQ
    ph = np.multiply.outer(np.asarray(t, dtype=float), w)
    return np.stack([AMP * w**k * np.sin(ph + k * np.pi / 2) for k in range(5)], axis=-2)


def u_of_x(x):
    p, a = x[0], x[2]
    return -5 * W**2 * a - 4 * W**4 * p


def f(x):
    return np.vstack([x[1:], u_of_x(x)])


def simulate(x0):
    ts = np.arange(0, T_END, DT)
    xs, us = [], []
    x = x0.copy()
    for _ in ts:
        xs.append(x.copy())
        us.append(u_of_x(x))
        k1 = f(x)
        k2 = f(x + DT / 2 * k1)
        k3 = f(x + DT / 2 * k2)
        k4 = f(x + DT * k3)
        x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ts, np.array(xs), np.array(us)


if __name__ == "__main__":
    x0 = ref(0.0)[:4]
    ts, xs, us = simulate(x0)
    R = ref(ts)
    pos_err = np.linalg.norm(xs[:, 0] - R[:, 0], axis=1)
    u_err = np.linalg.norm(us - R[:, 4], axis=1)
    print(f"nominal start: max |p - p*(t)| = {pos_err.max():.2e} m, max |u(x(t)) - u*(t)| = {u_err.max():.2e}")

    x0p = x0.copy()
    x0p[0] += [0.3, 0.2]                    # start 0.36 m off
    _, xp, _ = simulate(x0p)
    pert_err = np.linalg.norm(xp[:, 0] - R[:, 0], axis=1)
    print(f"perturbed start (0.36 m off): |p - p*(t)| at end = {pert_err[-1]:.2f} m, max = {pert_err.max():.2f} m")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].plot(*R[:, 0].T, "k--", lw=2.5, label="planned figure-8")
    ax[0].plot(*xs[:, 0].T, "C0", lw=1, label="u(x), ideal start")
    ax[0].plot(*xp[:, 0].T, "C3", lw=0.8, label="u(x), start 0.36 m off")
    ax[0].set_aspect("equal")
    ax[0].set_title(f"XY path, {T_END / PERIOD:.0f} laps")
    ax[0].legend(fontsize=8)

    ax[1].plot(ts, R[:, 4, 0], "k--", lw=2.5, label="u*(t) x")
    ax[1].plot(ts, R[:, 4, 1], "k:", lw=2.5, label="u*(t) y")
    ax[1].plot(ts, us[:, 0], "C0", lw=1, label="u(x(t)) x")
    ax[1].plot(ts, us[:, 1], "C1", lw=1, label="u(x(t)) y")
    ax[1].set_title("snap: u(x(t)) vs u*(t)")
    ax[1].set_xlabel("t [s]")
    ax[1].legend(fontsize=8)

    ax[2].semilogy(ts, pos_err + 1e-16, "C0", label="ideal start")
    ax[2].semilogy(ts, pert_err, "C3", label="start 0.36 m off")
    ax[2].set_title("|p − p*(t)| [m]")
    ax[2].set_xlabel("t [s]")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    out = Path("logs/lagrangian_fig8_sim.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
