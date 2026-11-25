"""Is the lifted problem really hard to optimise? (user, 2026-11-25) The vertical-climb toy (toy.py, 8 intervals),
solved with plain projected gradient descent + Armijo backtracking on the penalty function
    f = T + mu/2 (|E|^2 + |ends|^2),   box bounds by projection, mu raised step by step.
Two variants: raw equations, and every equation divided by its typical size (eq_scale of run.py).
Reference: IPOPT on the same problem (toy.py): T 1.5392 s (8 intervals).

    cd experiments/2026-11-25_lifted_own_optimizer
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python toy_gd.py
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

HERE = Path(__file__).resolve().parent
argv, sys.argv = sys.argv, sys.argv[:1]
spec = importlib.util.spec_from_file_location("toy", HERE / "toy.py")
TY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TY)
sys.argv = argv
LF = TY.LF
G, D, V0 = TY.G, TY.D, TY.V0
N_INT = int(os.environ.get("NI", 8))
STEPS = int(os.environ.get("STEPS", 20000))


def problem(n_int, scaled):
    M = TY.SPS * n_int
    u = (np.arange(M) + 0.5) / M
    B = [TY.basis(n_int, u, d) for d in range(4)]
    nb = B[0].shape[1]
    x = ca.MX.sym("x", nb + 3 * nb + 20 * M)
    Qc, Hc, Y = x[:nb], ca.reshape(x[nb:4 * nb], nb, 3), ca.reshape(x[4 * nb:], 20, M)
    r = [np.tile(np.array([0, 0, D])[:, None], (1, M))] + [np.zeros((3, M))] * 3
    q = [ca.mtimes(ca.DM(B[d]), Qc).T for d in range(4)]
    Hm = [ca.mtimes(ca.DM(B[d]), Hc).T for d in range(3)]
    E = LF.EQ.map(M)(ca.vertcat(*[ca.DM(a) for a in r], *q), ca.vertcat(*Hm), Y)
    if scaled:
        sc = np.r_[np.ones(3), np.full(3, 1 / G), np.full(3, 1e-2), np.full(3, 1e-3), 1.0, 1e-1, 1e-2,
                   1 / (4 * LF.TMAX), np.full(3, 1.0), D ** 2]
        E = E * ca.repmat(ca.DM(sc), 1, M)
    T = ca.sum2(1 / Y[19, :]) / M
    ends = ca.mtimes(ca.DM(TY.basis(n_int, np.array([0.0, 1.0]), 0)), Qc) - (V0 / D) ** 2
    if scaled:
        ends = ends * D ** 2
    mu = ca.MX.sym("mu")
    f = T + mu / 2 * (ca.sumsqr(E) + ca.sumsqr(ends))
    F = ca.Function("f", [x, mu], [f, T, ca.mmax(ca.fabs(E)), ca.mmax(ca.fabs(ends))])
    Fg = ca.Function("fg", [x, mu], [f, ca.gradient(f, x)])
    lb = np.r_[np.full(nb, 1e-8), np.full(3 * nb, -np.inf), np.tile(np.r_[np.full(9, -np.inf), -LF.WXY, -LF.WXY, -LF.WZ,
                                                                        np.full(3, -np.inf), np.full(4, LF.TMIN), 1e-6], M)]
    ub = np.r_[np.full(nb, np.inf), np.full(3 * nb, np.inf), np.tile(np.r_[np.full(9, np.inf), LF.WXY, LF.WXY, LF.WZ,
                                                                         np.full(3, np.inf), np.full(4, LF.TMAX), np.inf], M)]
    q0 = np.full(nb, (V0 / D) ** 2)
    y0 = np.r_[1, 0, 0, 0, 1, 0, G, 0, 0, 0, 0, 0, 0, 0, 0, np.full(4, LF.MASS * G / 4), V0 / D]
    x0 = np.r_[q0, np.tile([1.0, 0, 0], (nb, 1)).T.ravel(), np.tile(y0, M)]
    return F, Fg, lb, ub, x0


def gd(F, Fg, lb, ub, x0, log):
    x = np.clip(x0, lb, ub)
    a = 1e-6
    t0 = time.perf_counter()
    for mu in (1e1, 1e2, 1e3, 1e4, 1e5):
        for it in range(STEPS // 5):
            f, g = (np.array(v).ravel() for v in Fg(x, mu))
            f = float(f[0])
            a *= 2
            while True:
                xn = np.clip(x - a * g, lb, ub)
                fn = float(F(xn, mu)[0])
                if fn <= f - 1e-4 * (g @ (x - xn)) or a < 1e-30:
                    break
                a *= 0.5
            if a < 1e-30:
                break
            x = xn
        _, T, e, en = (float(v) for v in F(x, mu))
        log(f"    mu {mu:.0e}: {it + 1} steps, T {T:.4f} s, max |eq| {e:.1e}, ends {en:.1e}, last step {a:.1e}, "
            f"{time.perf_counter() - t0:.0f} s")
    return x


def main():
    a1 = 4 * LF.TMAX / LF.MASS - G
    vp = np.sqrt(V0 ** 2 + 2 * a1 * G * D / (a1 + G))
    print(f"toy, {N_INT} intervals, analytic T* {(vp - V0) * (1 / a1 + 1 / G):.4f} s, IPOPT on this problem 1.5392 s, "
          f"start T {D / V0:.1f} s; projected gradient descent + Armijo, {STEPS} steps")
    for scaled in (False, True):
        print(f"  equations {'scaled' if scaled else 'raw'}:")
        gd(*problem(N_INT, scaled), print)


if __name__ == "__main__" and not os.environ.get("REDUCED"):
    main()


def reduced(n_int, log):
    """Same toy without lifting: only q's coefficients; vertical climb, so c = D q'/2 + g exactly (no division), rotors
    = m c / 4 in [T_min, T_max]. Same objective (sample sum), same projected gradient descent + Armijo."""
    M = TY.SPS * n_int
    u = (np.arange(M) + 0.5) / M
    B0, B1 = (TY.basis(n_int, u, d) for d in range(2))
    nb = B0.shape[1]
    Be = TY.basis(n_int, np.array([0.0, 1.0]), 0)
    cmax = 4 * LF.TMAX / LF.MASS
    x = ca.MX.sym("x", nb)
    mu = ca.MX.sym("mu")
    q = ca.mtimes(ca.DM(B0), x)
    c = D * ca.mtimes(ca.DM(B1), x) / 2 + G
    T = ca.sum1(1 / ca.sqrt(q)) / M
    viol = ca.vertcat(ca.fmax(0, c - cmax), ca.fmax(0, -c)) / cmax
    ends = (ca.mtimes(ca.DM(Be), x) - (V0 / D) ** 2) * D ** 2
    f = T + mu / 2 * (ca.sumsqr(viol) + ca.sumsqr(ends))
    F = ca.Function("f", [x, mu], [f, T, ca.mmax(viol), ca.mmax(ca.fabs(ends))])
    Fg = ca.Function("fg", [x, mu], [f, ca.gradient(f, x)])
    lb, ub = np.full(nb, 1e-8), np.full(nb, np.inf)
    x0 = np.full(nb, (V0 / D) ** 2)
    xs = gd(F, Fg, lb, ub, x0, log)
    uf = np.linspace(0, 1, 200001)
    Tf = float(np.trapezoid(1 / np.sqrt(np.maximum(TY.basis(n_int, uf, 0) @ xs, 1e-16)), uf))
    log(f"    exact time of the result {Tf:.4f} s")


if __name__ == "__main__" and os.environ.get("REDUCED"):
    print(f"  without lifting ({N_INT} intervals, only q):")
    reduced(N_INT, print)
