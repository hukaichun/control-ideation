"""Fit L = 1/2|a|^2 + G(p, v) with G = sum_k theta_k * RBF_k(p, v) to a planned trajectory, then fly the
resulting time-independent law u(x) = snap = d/dt G_v - G_p on the flat chain (2D, x/y).

E-L of L(p, v, a):  G_p - d/dt G_v + snap = 0  ->  u = G_vp v + G_vv a - G_p, linear in theta.
Fit: least squares  snap*(t) = sum_k theta_k psi_k(x*(t))  over samples of the planned trajectory.

    uv run examples/rbf_lagrangian_fit.py
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T = 20.0                       # period of the planned loop [s]
W0 = 2 * np.pi / T
WAYPOINTS = np.array([[0, 4], [3, 3], [4, 0], [2, -2], [-1, -3], [-4, -1], [-3, 2]], float)


# ---------- planned trajectory: periodic trig interpolation through waypoints ----------
def plan(wp):
    n = len(wp)
    k = np.arange(-(n // 2), n // 2 + 1)
    ti = np.arange(n) * T / n
    C = np.exp(-1j * np.outer(k, W0 * ti)) @ wp / n
    return k, C


def traj(t, plan_kc, d):
    """d-th time derivative of the planned trajectory, shape (len(t), 2)."""
    k, C = plan_kc
    return ((np.exp(1j * np.outer(t, k * W0)) * (1j * k * W0) ** d) @ C).real


# ---------- Gaussian RBF features in (p, v) ----------
class RBF:
    def __init__(self, cp, cv, sp, sv):
        self.cp, self.cv, self.sp, self.sv = cp, cv, sp, sv

    def psi(self, p, v, a):
        """psi_k(x) = d/dt grad_v phi_k - grad_p phi_k, shape (N, K, 2)."""
        dp = p[:, None, :] - self.cp[None]
        dv = v[:, None, :] - self.cv[None]
        phi = np.exp(-0.5 * ((dp / self.sp) ** 2).sum(-1) - 0.5 * ((dv / self.sv) ** 2).sum(-1))
        rp, rv = dp / self.sp ** 2, dv / self.sv ** 2
        rp_v = (rp * v[:, None]).sum(-1, keepdims=True)      # rp . v
        rv_a = (rv * a[:, None]).sum(-1, keepdims=True)      # rv . a
        # G_vp v = phi rv (rp.v);  G_vv a = phi (rv (rv.a) - a/sv^2);  -G_p = phi rp
        return phi[..., None] * (rv * (rp_v + rv_a) - a[:, None] / self.sv ** 2 + rp)

    def u(self, theta, p, v, a):
        return np.einsum("nkd,k->nd", self.psi(p, v, a), theta)


def fit(rbf, p, v, a, snap, ridge):
    A = rbf.psi(p, v, a).transpose(0, 2, 1).reshape(-1, len(rbf.cp))
    b = snap.reshape(-1)
    AtA = A.T @ A
    theta = np.linalg.solve(AtA + ridge * np.trace(AtA) / len(AtA) * np.eye(len(AtA)), A.T @ b)
    return theta


def simulate(rbf, theta, x0, t_end, dt=0.005):
    def f(x):
        p, v, a, j = x[:2], x[2:4], x[4:6], x[6:]
        return np.concatenate([v, a, j, rbf.u(theta, p[None], v[None], a[None])[0]])

    n = int(t_end / dt)
    xs = np.empty((n + 1, 8))
    xs[0] = x = x0
    for i in range(n):
        k1 = f(x)
        k2 = f(x + dt / 2 * k1)
        k3 = f(x + dt / 2 * k2)
        k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs[i + 1] = x
    return np.arange(n + 1) * dt, xs


def nearest_dist(p, ref):
    return np.min(np.linalg.norm(p[:, None] - ref[None], axis=-1), axis=1)


def main(K, width, ridge, periods, out, offsets):
    pk = plan(WAYPOINTS)
    ts = np.linspace(0, T, 2000, endpoint=False)
    p, v, a, j, s = (traj(ts, pk, d) for d in range(5))

    tc = np.linspace(0, T, K, endpoint=False)
    cp, cv = traj(tc, pk, 0), traj(tc, pk, 1)
    sp = width * np.mean(np.linalg.norm(np.diff(cp, axis=0), axis=1))
    sv = width * np.mean(np.linalg.norm(np.diff(cv, axis=0), axis=1))
    if offsets:  # also place centers off the curve, +-1 width along each p and v axis
        D = np.concatenate([np.zeros((1, 4)), np.eye(4), -np.eye(4)]) * np.array([sp, sp, sv, sv])
        cp = (cp[:, None, :] + D[None, :, :2]).reshape(-1, 2)
        cv = (cv[:, None, :] + D[None, :, 2:]).reshape(-1, 2)
    rbf = RBF(cp, cv, sp, sv)
    theta = fit(rbf, p, v, a, s, ridge)

    u_fit = rbf.u(theta, p, v, a)
    rel = np.linalg.norm(u_fit - s) / np.linalg.norm(s)
    print(f"K={len(cp)}  sp={sp:.2f} m  sv={sv:.2f} m/s  fit residual (relative) = {rel:.2e}")

    x0 = np.concatenate([p[0], v[0], a[0], j[0]])
    x0_pert = x0 + np.array([0.3, -0.2, 0, 0, 0, 0, 0, 0])
    t_end = periods * T
    t1, x1 = simulate(rbf, theta, x0, t_end)
    t2, x2 = simulate(rbf, theta, x0_pert, t_end)

    ref_dense = traj(np.linspace(0, T, 4000, endpoint=False), pk, 0)
    e_time = np.linalg.norm(x1[:, :2] - traj(t1, pk, 0), axis=1)
    e_path1 = nearest_dist(x1[:, :2], ref_dense)
    e_path2 = nearest_dist(x2[:, :2], ref_dense)
    for lab, e in (("ideal start, distance to path", e_path1), ("perturbed start, distance to path", e_path2)):
        per = [e[(t1 >= k * T) & (t1 < (k + 1) * T)].max() for k in range(periods)]
        print(f"{lab}: max per period = " + ", ".join(f"{x:.3f}" for x in per) + " m")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    ax[0].plot(p[:, 0], p[:, 1], "k-", lw=3, alpha=0.25, label="planned")
    ax[0].plot(*WAYPOINTS.T, "ks", ms=5, label="waypoints")
    ax[0].plot(x1[:, 0], x1[:, 1], "C0-", lw=1, label="u(x), ideal start")
    ax[0].plot(x2[:, 0], x2[:, 1], "C3-", lw=0.8, alpha=0.8, label="u(x), perturbed start")
    ax[0].plot(*cp.T, "C2.", ms=2, label="RBF centers (p)")
    ax[0].set_aspect("equal")
    ax[0].set_title(f"xy, {periods} periods")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(ts, s[:, 0], "k-", lw=2.5, alpha=0.3, label="snap* x")
    ax[1].plot(ts, s[:, 1], "k--", lw=2.5, alpha=0.3, label="snap* y")
    ax[1].plot(ts, u_fit[:, 0], "C0-", lw=1, label="u(x*) x")
    ax[1].plot(ts, u_fit[:, 1], "C1--", lw=1, label="u(x*) y")
    ax[1].set_title(f"fit along trajectory, rel. residual {rel:.1e}")
    ax[1].set_xlabel("t [s]")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].semilogy(t1, e_time + 1e-9, "C0:", label="ideal: |p - p*(t)|")
    ax[2].semilogy(t1, e_path1 + 1e-9, "C0-", label="ideal: distance to path")
    ax[2].semilogy(t2, e_path2 + 1e-9, "C3-", label="perturbed: distance to path")
    ax[2].set_xlabel("t [s]")
    ax[2].set_ylabel("[m]")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"plot -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=60, help="number of RBF centers along the trajectory")
    ap.add_argument("--width", type=float, default=2.0, help="RBF width / center spacing")
    ap.add_argument("--ridge", type=float, default=1e-10)
    ap.add_argument("--periods", type=int, default=3)
    ap.add_argument("--no-offsets", action="store_true", help="centers only on the trajectory")
    ap.add_argument("-o", "--out", type=Path, default=Path("logs/rbf_lagrangian_fit.png"))
    a = ap.parse_args()
    main(a.K, a.width, a.ridge, a.periods, a.out, not a.no_offsets)
