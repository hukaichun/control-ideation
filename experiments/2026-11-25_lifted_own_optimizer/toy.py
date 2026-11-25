"""One-segment toy with an analytic answer (user, 2026-11-25: check that the lifted formulation is not wrong).

Vertical climb of D m along r(u) = (0, 0, D u), u in [0, 1], speed v0 at both ends. Body stays level, all rotors equal:
time-optimal is bang-bang: full thrust (a = 4 Tmax / m - g) up to x1, then zero thrust (free fall, a = -g):
    x1 = g D / (a1 + g),  vp^2 = v0^2 + 2 a1 x1,  T* = (vp - v0) (1/a1 + 1/g)
The zero-thrust part is the c = 0 point that blew up the division-based formulas.

Same lifted equations as run.py (../2026-11-25_lifted_our_space EQ): per sample x_b, y_b, c, c', c'', w, w', 4 rotors, s.
Open clamped B-splines of degree 4 for q = u'^2 and the heading reference h (3D, a variable). Ends: q(0) = q(1) =
(v0 / D)^2. Bounds as CPC. Solved with IPOPT (this checks the formulation, not an optimiser).

    cd experiments/2026-11-25_lifted_own_optimizer
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python toy.py
"""
import importlib.util
import sys
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.interpolate import BSpline

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
argv, sys.argv = sys.argv, sys.argv[:1]
spec = importlib.util.spec_from_file_location("lifted", EXP / "2026-11-25_lifted_our_space" / "run.py")
LF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LF)
sys.argv = argv
G = 9.81
D, V0, DEG, SPS = 10.0, 1.0, 4, 4


def basis(n_int, u, d):
    t = np.r_[np.zeros(DEG), np.linspace(0, 1, n_int + 1), np.ones(DEG)]
    nb = len(t) - DEG - 1
    out = np.zeros((len(u), nb))
    for i in range(nb):
        c = np.zeros(nb)
        c[i] = 1
        b = BSpline(t, c, DEG)
        out[:, i] = (b.derivative(d) if d else b)(u)
    return out


def solve(n_int):
    M = SPS * n_int
    u = (np.arange(M) + 0.5) / M
    B = [basis(n_int, u, d) for d in range(4)]
    nb = B[0].shape[1]
    Qc = ca.MX.sym("Q", nb)            # q coefficients (q = u'^2)
    Hc = ca.MX.sym("H", nb, 3)
    Y = ca.MX.sym("Y", 20, M)
    r = [np.tile(np.array([0, 0, D])[:, None], (1, M))] + [np.zeros((3, M))] * 3      # r1..r4
    q = [ca.mtimes(ca.DM(B[d]), Qc).T for d in range(4)]
    Hm = [ca.mtimes(ca.DM(B[d]), Hc).T for d in range(3)]
    X = ca.vertcat(*[ca.DM(x) for x in r], *q)
    E = LF.EQ.map(M)(X, ca.vertcat(*Hm), Y)
    T = ca.sum2(1 / Y[19, :]) / M
    Be = basis(n_int, np.array([0.0, 1.0]), 0)
    gb = ca.mtimes(ca.DM(Be), Qc) - (V0 / D) ** 2
    x = ca.vertcat(Qc, ca.vec(Hc), ca.vec(Y))
    lbx = np.r_[np.full(nb, 1e-8), np.full(3 * nb, -np.inf), np.tile(np.r_[np.full(9, -np.inf), -LF.WXY, -LF.WXY, -LF.WZ,
                                                                          np.full(3, -np.inf), np.full(4, LF.TMIN), 1e-6], M)]
    ubx = np.r_[np.full(nb, np.inf), np.full(3 * nb, np.inf), np.tile(np.r_[np.full(9, np.inf), LF.WXY, LF.WXY, LF.WZ,
                                                                           np.full(3, np.inf), np.full(4, LF.TMAX), np.inf], M)]
    # start: constant speed v0, level, hover thrust
    q0 = np.full(nb, (V0 / D) ** 2)
    y0 = np.r_[1, 0, 0, 0, 1, 0, G, 0, 0, 0, 0, 0, 0, 0, 0, np.full(4, LF.MASS * G / 4), V0 / D]
    x0 = np.r_[q0, np.tile([1.0, 0, 0], (nb, 1)).T.ravel(), np.tile(y0, M)]
    S = ca.nlpsol("toy", "ipopt", {"x": x, "f": T, "g": ca.vertcat(ca.vec(E), gb)},
                  {"expand": True, "print_time": False, "ipopt": {"print_level": 0, "max_iter": 3000}})
    sol = S(x0=x0, lbx=lbx, ubx=ubx, lbg=0, ubg=0)
    xs = np.array(sol["x"]).ravel()
    Ys = xs[4 * nb:].reshape(M, 20).T
    Ef = np.array(LF.EQ.map(M)(np.array(ca.evalf(ca.substitute(X, x, xs))), np.array(ca.evalf(ca.substitute(ca.vertcat(*Hm), x, xs))), Ys))
    # fine check: exact time integral and thrust between the samples (vertical: c = a + g = D q'/2 + g)
    uf = np.linspace(0, 1, 200001)
    Qs = xs[:nb]
    qf = basis(n_int, uf, 0) @ Qs
    q1f = basis(n_int, uf, 1) @ Qs
    Tf = float(np.trapezoid(1 / np.sqrt(np.maximum(qf, 1e-16)), uf))
    cf = D * q1f / 2 + G
    return (float(sol["f"]), S.stats()["return_status"], S.stats()["iter_count"], Ys, u, float(np.abs(Ef).max()),
            Tf, cf.min(), cf.max())


def main():
    a1 = 4 * LF.TMAX / LF.MASS - G
    x1 = G * D / (a1 + G)
    vp = np.sqrt(V0 ** 2 + 2 * a1 * x1)
    Ts = (vp - V0) * (1 / a1 + 1 / G)
    print(f"analytic: a1 {a1:.3f} m/s^2, switch at {x1:.3f} m, peak {vp:.3f} m/s, T* = {Ts:.4f} s")
    for n_int in (4, 8, 16, 32, 64):
        T, st, it, Ys, u, ef, Tf, cmin, cmax = solve(n_int)
        c, rot, w = Ys[6], Ys[15:19], Ys[9:12]
        v = Ys[19] * D
        print(f"  {n_int:3d} intervals: {st:22s} {it:4d} it | T {T:.4f} s ({100 * (T / Ts - 1):+.2f}% vs analytic) | "
              f"|eq| {ef:.1e} | c min {c.min():.3f} max {c.max():.2f} (analytic 0 / {4 * LF.TMAX / LF.MASS:.2f}) | "
              f"rotor spread {np.ptp(rot, 0).max():.1e} N | max |w| {np.abs(w).max():.1e} | peak speed {v.max():.3f}")
        print(f"        fine check: exact time {Tf:.4f} s ({100 * (Tf / Ts - 1):+.2f}%), thrust c between samples "
              f"[{cmin:.2f}, {cmax:.2f}] (limits [0, {4 * LF.TMAX / LF.MASS:.2f}])")


if __name__ == "__main__":
    main()
