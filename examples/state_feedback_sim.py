"""Planned trajectory rewritten as a pure state feedback u(x), in flat coordinates (no PX4).

Each horizontal axis is a chain of integrators: state [p, v, a, j], input u = snap.
Planned trajectory: figure-8 x*(s), with feedforward u*(s) from flatness.

  time-based : u = u*(t)    - K (x - x*(t))
  state-based: u = u*(s(x)) - K (x - x*(s(x))),   s(x) = argmin_s |x - x*(s)|_W

s(x) is computed from the state only (global search over the curve), so u is a function of x alone.

    uv run examples/state_feedback_sim.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PERIOD = 12.0                   # s, one figure-8
W0 = 2 * np.pi / PERIOD
AMP = np.array([3.0, 1.5])      # m
FREQ = np.array([1.0, 2.0])     # x: sin(w s), y: sin(2 w s)
LAM = 2.0                       # closed-loop pole (s + LAM)^4 per axis
K = np.array([LAM**4, 4 * LAM**3, 6 * LAM**2, 4 * LAM])     # gains on [p, v, a, j]
WVEC = np.array([1, LAM**-2, LAM**-4, LAM**-6])[:, None]    # projection metric

DT = 0.004
T_END = 30.0
HOLD = (4.0, 7.0)               # vehicle is held in place during this window


def ref(s):
    """Derivatives 0..5 of the planned position at parameter s -> (..., 6, 2)."""
    w = W0 * FREQ
    phase = np.multiply.outer(np.asarray(s, dtype=float), w)
    return np.stack([AMP * w**k * np.sin(phase + k * np.pi / 2) for k in range(6)], axis=-2)


S_TAB = np.linspace(0, PERIOD, 3000, endpoint=False)
X_TAB = ref(S_TAB)[:, :4, :]    # (N, 4, 2) state along the curve


def project(x):
    """s(x): parameter of the closest curve point to state x. Depends on x only."""
    s = S_TAB[np.argmin(((X_TAB - x) ** 2 * WVEC).sum(axis=(1, 2)))]
    for _ in range(4):          # Newton refinement of d/ds |x - x*(s)|^2 = 0
        R = ref(s)
        e = R[:4] - x
        g = (e * WVEC * R[1:5]).sum()
        h = (R[1:5] * WVEC * R[1:5]).sum() + (e * WVEC * R[2:6]).sum()
        if h <= 0:
            break
        s -= np.clip(g / h, -0.05, 0.05)
    return s % PERIOD


def u_state(x, t):
    R = ref(project(x))
    return R[4] - K @ (x - R[:4])


def u_time(x, t):
    R = ref(t)
    return R[4] - K @ (x - R[:4])


def f(x, u):
    return np.vstack([x[1:], u])


def simulate(controller, x0, hold=None):
    x = x0.copy()
    ts = np.arange(0, T_END, DT)
    xs, us = [], []
    for t in ts:
        u = controller(x, t)
        xs.append(x.copy())
        us.append(u)
        if hold and hold[0] <= t < hold[1]:
            x = np.vstack([x[:1], np.zeros((3, 2))])     # pinned: p frozen, v = a = j = 0
            continue
        k1 = f(x, u)
        k2 = f(x + DT / 2 * k1, controller(x + DT / 2 * k1, t + DT / 2))
        k3 = f(x + DT / 2 * k2, controller(x + DT / 2 * k2, t + DT / 2))
        k4 = f(x + DT * k3, controller(x + DT * k3, t + DT))
        x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ts, np.array(xs), np.array(us)


def dist_to_path(P):
    """Position distance to the planned path (geometry only)."""
    return np.min(np.linalg.norm(P[:, None, :] - X_TAB[None, :, 0, :], axis=2), axis=1)


if __name__ == "__main__":
    x_start = ref(0.0)[:4]

    # A. On the curve: u(x*(t)) must equal u*(t).
    tc = np.linspace(0, PERIOD, 500)
    ident_err = np.array([np.linalg.norm(u_state(ref(t)[:4], 0) - ref(t)[4]) for t in tc])
    ts, xa, ua = simulate(u_state, x_start)
    ustar = ref(ts)[:, 4, :]
    print(f"A. on-curve  max|u(x*(t)) - u*(t)| = {ident_err.max():.2e}")
    print(f"A. closed loop from ideal start: max|u(x(t)) - u*(t)| = {np.linalg.norm(ua - ustar, axis=1).max():.2e}, "
          f"max|p - p*(t)| = {np.linalg.norm(xa[:, 0] - ref(ts)[:, 0], axis=1).max():.2e} m")

    # B. Vehicle pinned for 3 s, then released.
    _, xt, _ = simulate(u_time, x_start, HOLD)
    _, xs, _ = simulate(u_state, x_start, HOLD)
    after = ts >= HOLD[1]
    for name, X in (("time-based ", xt), ("state-based", xs)):
        print(f"B. {name}: after release  max dist to path {dist_to_path(X[after, 0]).max():.2f} m, "
              f"max |a| {np.linalg.norm(X[after, 2], axis=1).max():.2f} m/s^2, "
              f"max |v| {np.linalg.norm(X[after, 1], axis=1).max():.2f} m/s")

    # C. Start off the curve, at rest.
    x_off = np.zeros((4, 2))
    x_off[0] = [-1.0, 1.5]
    _, xc, _ = simulate(u_state, x_off)
    print(f"C. off-curve start: dist to path {dist_to_path(xc[:1, 0])[0]:.2f} m -> "
          f"{dist_to_path(xc[-1:, 0])[0]:.3f} m at t={T_END:.0f}s")

    nominal_a = np.linalg.norm(ref(S_TAB)[:, 2, :], axis=1).max()
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(*X_TAB[:, 0, :].T, "k--", lw=1, label="planned path")
    ax[0, 0].plot(*xt[:, 0].T, "C3", label="time-based, pinned 3 s")
    ax[0, 0].plot(*xs[:, 0].T, "C0", label="state-based, pinned 3 s")
    ax[0, 0].plot(*xc[:, 0].T, "C2", lw=1, label="state-based, off-curve start")
    ax[0, 0].plot(*xs[ts.searchsorted(HOLD[0]), 0], "ko", label="pinned here")
    ax[0, 0].set_aspect("equal")
    ax[0, 0].set_title("XY path")
    ax[0, 0].legend(fontsize=8)

    for X, c, lab in ((xt, "C3", "time-based"), (xs, "C0", "state-based")):
        ax[0, 1].plot(ts, dist_to_path(X[:, 0]), c, label=lab)
        ax[1, 0].plot(ts, np.linalg.norm(X[:, 2], axis=1), c, label=lab)
    ax[1, 0].axhline(nominal_a, color="k", ls=":", lw=1, label="max |a| of planned path")
    for a in (ax[0, 1], ax[1, 0]):
        a.axvspan(*HOLD, color="0.85", label="pinned")
        a.set_xlabel("t [s]")
        a.legend(fontsize=8)
    ax[0, 1].set_title("distance to path [m]")
    ax[1, 0].set_title("|acceleration| [m/s²]  (≈ tilt demand)")

    ax[1, 1].semilogy(tc, ident_err + 1e-16)
    ax[1, 1].set_title("on the curve: |u(x*(t)) − u*(t)|")
    ax[1, 1].set_xlabel("t [s]")
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    out = Path("logs/state_feedback_sim.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
