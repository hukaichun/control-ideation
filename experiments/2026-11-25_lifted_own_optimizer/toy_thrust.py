"""Vertical-climb toy redone with the thrust as the search space (user, 2026-11-25).

Decision variables: control points P_j of the collective thrust c(u) (per unit mass), a clamped cubic B-spline in u
(c in C2 -> q in C3), each P_j in [0, c_max]: by the convex hull the thrust is within its limits everywhere.
Speed from the thrust, analytically: path z = D u, a = D u'' = D q'/2, c = a + g  =>
    q(u) = q0 + (2/D) (int_0^u c du' - g u)          (int of a B-spline: a B-spline, exact)
Ends: q(1) = q(0) = (v0/D)^2  <=>  int_0^1 c du = g   (one linear equality).
Objective: T = int_0^1 du / sqrt(q), Gauss-Legendre per knot interval (no sample sum).
Optimiser: projected gradient descent + Armijo backtracking. Projection onto {box} and {a.P = g} by bisection on the
multiplier. Steps that would make a control point of q non-positive are cut back (q > 0 everywhere by its convex hull).
Analytic optimum: bang-bang, T* = (vp - v0)(1/a1 + 1/g).

    cd experiments/2026-11-25_lifted_own_optimizer
    uv run --with scipy --with numpy python toy_thrust.py
"""
import time

import numpy as np
from scipy.interpolate import BSpline

G = 9.81
D, V0 = 10.0, 1.0
MASS = 0.85
TMAX = 3.3 * MASS * G / 4
CMAX = 4 * TMAX / MASS
DEG = 3
STEPS = 20000


def setup(n_int):
    t = np.r_[np.zeros(DEG), np.linspace(0, 1, n_int + 1), np.ones(DEG)]
    nb = len(t) - DEG - 1
    # quadrature points (8 Gauss points per interval)
    xg, wg = np.polynomial.legendre.leggauss(8)
    e = np.linspace(0, 1, n_int + 1)
    uq = ((e[:-1, None] + e[1:, None]) / 2 + (e[1:, None] - e[:-1, None]) / 2 * xg[None, :]).ravel()
    wq = ((e[1:, None] - e[:-1, None]) / 2 * wg[None, :]).ravel()
    A = np.zeros((len(uq), nb))                  # A[:, j] = int_0^u B_j
    a1 = np.zeros(nb)                            # int_0^1 B_j
    Aq = []                                      # q's B-spline coefficients from P (for the positivity check)
    for j in range(nb):
        c = np.zeros(nb)
        c[j] = 1
        Bi = BSpline(t, c, DEG).antiderivative()
        A[:, j] = Bi(uq)
        a1[j] = Bi(1.0)
        Aq.append(Bi.c[:len(Bi.t) - Bi.k - 1])
    Aq = np.array(Aq).T                          # coefficients of int B_j in the degree-4 basis
    tq = BSpline(t, np.zeros(nb), DEG).antiderivative().t
    one = np.zeros(Aq.shape[0])                  # coefficients of u in that basis (Greville abscissae)
    kq = DEG + 1
    one = np.array([tq[i + 1:i + kq + 1].mean() for i in range(Aq.shape[0])])
    return dict(t=t, nb=nb, uq=uq, wq=wq, A=A, a1=a1, Aq=Aq, one=one)


def q_of(S, P):
    q0 = (V0 / D) ** 2
    return q0 + 2 / D * (S["A"] @ P - G * S["uq"])


def qcoef(S, P):
    return (V0 / D) ** 2 + 2 / D * (S["Aq"] @ P - G * S["one"])


def T_and_grad(S, P):
    q = q_of(S, P)
    T = S["wq"] @ q ** -0.5
    dTdq = -0.5 * S["wq"] * q ** -1.5
    return T, (2 / D) * (S["A"].T @ dTdq)


def project(S, y):
    a = S["a1"]
    lo, hi = np.min((y - CMAX) / a) - 1.0, np.max(y / a) + 1.0      # a > 0: all at CMAX at lo, all at 0 at hi
    for _ in range(200):                         # a . clip(y - lam a) is decreasing in lam
        lam = (lo + hi) / 2
        if a @ np.clip(y - lam * a, 0, CMAX) > G:
            lo = lam
        else:
            hi = lam
    P = np.clip(y - (lo + hi) / 2 * a, 0, CMAX)
    assert abs(a @ P - G) < 1e-9, a @ P - G
    return P


def solve(n_int):
    S = setup(n_int)
    P = project(S, np.full(S["nb"], G))          # start: hover thrust (constant speed v0)
    alpha = 1.0
    t0 = time.perf_counter()
    why = "max steps"
    for it in range(STEPS):
        T, g = T_and_grad(S, P)
        alpha *= 2
        ok = False
        npos = narm = 0
        while alpha >= 1e-16:
            Pn = project(S, P - alpha * g)
            if np.all(qcoef(S, Pn) > 0):
                Tn = T_and_grad(S, Pn)[0]
                if Tn <= T + 1e-4 * g @ (Pn - P):
                    ok = True
                    break
                narm += 1
            else:
                npos += 1
            alpha *= 0.5
        if not ok:
            why = f"line search failed (q<=0 rejections {npos}, Armijo rejections {narm}), |proj step| {np.linalg.norm(project(S, P - 1e-8 * g) - P):.1e}"
            break
        if np.linalg.norm(Pn - P) < 1e-13:
            P = Pn
            why = "converged (step < 1e-13)"
            break
        P = Pn
    print("      stop:", why)
    T = T_and_grad(S, P)[0]
    return T, it + 1, time.perf_counter() - t0, P, S


def main():
    a1 = CMAX - G
    x1 = G * D / (a1 + G)
    vp = np.sqrt(V0 ** 2 + 2 * a1 * x1)
    Ts = (vp - V0) * (1 / a1 + 1 / G)
    print(f"analytic: full thrust to {x1:.3f} m, then zero thrust; peak {vp:.3f} m/s, T* = {Ts:.4f} s")
    for n_int in (4, 8, 16, 32, 64, 128):
        T, it, dt, P, S = solve(n_int)
        uu = np.linspace(0, 1, 100001)
        c = BSpline(S["t"], P, DEG)(uu)
        print(f"  {n_int:3d} intervals ({S['nb']} variables): T {T:.4f} s ({100 * (T / Ts - 1):+.3f}%), {it} steps, "
              f"{dt * 1000:.0f} ms | thrust on a dense grid [{c.min():.3f}, {c.max():.3f}] (limits [0, {CMAX:.2f}]), "
              f"end speed check q(1) {qcoef(S, P)[-1] * D ** 2:.6f} (= v0^2 {V0 ** 2})")


if __name__ == "__main__":
    main()
