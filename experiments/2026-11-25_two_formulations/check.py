"""Checks before any optimisation (user, 2026-11-25).
1. physical(): |w| and torque against CPC's own states and rotor thrusts (tracks 0, 1, 5).
2. Chains: on an analytic test path p(u) and speed (du/dt)^2 = r(u), the w chain (w = 1/sqrt(r)) and the q chain
   give the same f, j, s, and both match f, j, s from direct differentiation in time (u(t) by integration).
3. Limit rows: sign agrees with the physical margins at 20000 random (f, j, s), c in [0.01, 40].
4. Gradients of both limit functions vs central differences.
    uv run --with casadi --with pyyaml --with scipy --with numpy python check.py
"""
import math

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

import common as C
import q_version as QV
import w_version as WV


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def cpc(k):
    d = np.load(C.ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")
    x, u = d["x"], d["u"]
    J = np.array(C.Q["inertia"], float)
    l, cq = C.ARM, C.Q["torque_coeff"]
    MIX = np.array([[l, -l, -l, l], [-l, -l, l, l], [cq, -cq, cq, -cq]])
    e3 = np.array([0, 0, 1.0])
    F, Jk, S, wx, tx = [], [], [], [], []
    for i in range(len(x)):
        R = quat_R(x[i, 6:10]); w = x[i, 10:13]
        wd = np.linalg.solve(J, MIX @ u[i] - np.cross(w, J @ w))
        c = u[i].sum() / C.MASS
        F.append(c * R[:, 2]); Jk.append(c * R @ np.cross(w, e3))
        S.append(c * R @ (np.cross(wd, e3) + np.cross(w, np.cross(w, e3))))
        wx.append(np.linalg.norm(w[:2])); tx.append(np.sqrt(2) * J[0, 0] * np.linalg.norm(wd[:2]))
    F, Jk, S = (np.array(a).T for a in (F, Jk, S))
    w, tr, c = C.physical(F, Jk, S)
    ta = 4 * C.ARM * np.minimum(C.TMAX - C.MASS * c / 4, C.MASS * c / 4)
    print(f"1. CPC track {k}: |w_xy| error max {np.abs(w - wx).max():.1e} rad/s, torque error max "
          f"{np.abs(tr * ta - tx).max():.1e} N m (CPC has |w_z| <= 0.3: not in our heading-free formulas)")


def chains():
    u = ca.SX.sym("u")
    p = ca.vertcat(10 * ca.cos(2 * np.pi * u), 6 * ca.sin(4 * np.pi * u), 3 + ca.sin(2 * np.pi * u))
    r = 0.004 * (1.5 + ca.sin(2 * np.pi * u) * 0.8 + 0.3 * ca.cos(6 * np.pi * u))     # (du/dt)^2
    L = 50.0
    Q = r * L ** 2
    w = 1 / ca.sqrt(r)
    D = lambda e, k: e if k == 0 else D(ca.jacobian(e, u), k - 1)
    Pd = ca.vertcat(*[D(p, k) for k in range(1, 5)])
    fw = ca.Function("fw", [u], list(WV.CHAIN(ca.vertcat(Pd, *[D(w, k) for k in range(4)]))))
    fq = ca.Function("fq", [u], list(QV.chain(L)(ca.vertcat(Pd, *[D(Q, k) for k in range(4)]))))
    pf = ca.Function("p", [u], [D(p, 2), D(p, 1), D(p, 3), D(p, 4)])
    rf = ca.Function("r", [u], [ca.sqrt(r)])
    # direct: integrate du/dt = sqrt(r), differentiate positions numerically in time
    sol = solve_ivp(lambda t, y: [float(rf(y[0]))], [0, 30], [0.0], dense_output=True, rtol=1e-13, atol=1e-14, method="DOP853")
    err_wq, err_dir = 0.0, 0.0
    for t0 in np.linspace(1, 25, 7):
        uu = float(sol.sol(t0)[0])
        f1, j1, s1 = (np.array(v).ravel() for v in fw(uu))
        f, fu, fuu, rr, ru = (np.array(v).ravel() for v in fq(uu))
        j2, s2 = np.sqrt(rr) * fu, rr * fuu + ru * fu / 2
        err_wq = max(err_wq, np.abs(np.r_[f1 - f, j1 - j2, s1 - s2]).max() / np.abs(np.r_[f1, j1, s1]).max())
        h = 2e-3
        P = lambda t: np.array(ca.Function("pp", [u], [p])(float(sol.sol(t)[0]))).ravel()
        ts = t0 + h * np.arange(-3, 4)
        X = np.array([P(t) for t in ts])
        # finite-difference weights for derivatives 2, 3, 4 on 7 points
        V = np.vander(ts - t0, 7, increasing=True).T
        def fd(k):
            b = np.zeros(7); b[k] = math.factorial(k)
            return np.linalg.solve(V, b) @ X
        fd_ = [fd(2) + [0, 0, C.G], fd(3), fd(4)]
        err_dir = max(err_dir, max(np.abs(a - b).max() / max(np.abs(b).max(), 1) for a, b in zip(fd_, (f1, j1, s1))))
    print(f"2. chains on an analytic path: w chain vs q chain max relative difference {err_wq:.1e}; "
          f"vs direct time differentiation {err_dir:.1e} (finite differences, h = 2 ms)")


def signs():
    """Rows must equal the physical margins computed independently by physical()."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20000):
        c = 10 ** rng.uniform(-2, 1.6)
        f = rng.normal(size=3); f *= c / np.linalg.norm(f)
        j, s = rng.normal(0, 50, 3), rng.normal(0, 3000, 3)
        g = np.array(WV.LIM(f, j, s)).ravel()
        w, tr, cc = C.physical(f[:, None], j[:, None], s[:, None])
        tq = tr[0] * min(C.TMAX - C.MASS * c / 4, C.MASS * c / 4)
        ref = np.array([C.WMAX - w[0], 4 * C.TMAX / C.MASS - cc[0], C.TMAX - C.MASS * c / 4 - tq, C.MASS * c / 4 - tq])
        worst = max(worst, np.max(np.abs(g - ref) / np.maximum(np.abs(ref), 1e-12)))
    print(f"3. 20000 random points: limit rows vs physical margins, max relative difference {worst:.1e}")


def grads():
    rng = np.random.default_rng(1)
    for nm, F, dims in (("w", WV.LIM, (3, 3, 3)), ("q", QV.LIM, (3, 3, 3, 1, 1))):
        xs = [ca.SX.sym(f"x{i}", d) for i, d in enumerate(dims)]
        X = ca.vertcat(*xs)
        split = lambda v: [v[sum(dims[:i]):sum(dims[:i + 1])] for i in range(len(dims))]
        G = ca.Function("G", [X], [F(*split(X))])
        Jf = ca.Function("J", [X], [ca.jacobian(F(*split(X)), X)])
        worst = 0
        for _ in range(20):
            x0 = np.r_[rng.normal(0, 5, 3) + [0, 0, 10], rng.normal(0, 30, 3), rng.normal(0, 300, 3),
                       rng.uniform(0.001, 0.1, len(dims) - 3)][:X.shape[0]]
            Ja = np.array(Jf(x0)); Jn = np.zeros_like(Ja)
            for q in range(len(x0)):
                h = 1e-6 * max(1, abs(x0[q])); xp, xm = x0.copy(), x0.copy(); xp[q] += h; xm[q] -= h
                Jn[:, q] = (np.array(G(xp)).ravel() - np.array(G(xm)).ravel()) / (2 * h)
            worst = max(worst, np.max(np.abs(Ja - Jn).max(1) / np.maximum(np.abs(Ja).max(1), 1e-300)))
        print(f"4. [{nm}] limit gradients vs central differences, 20 points: max error per row {worst:.1e}")


if __name__ == "__main__":
    for k in (0, 1, 5):
        cpc(k)
    chains()
    signs()
    grads()
