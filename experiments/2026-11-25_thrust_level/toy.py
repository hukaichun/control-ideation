"""Vertical-climb toy in the slowness formulation (bez.py): p(u) = (0, 0, D u) fixed, w(u) = dt/du a clamped B-spline
(degree 4) with n intervals, ends v = v0 (w = D / v0). Constraint: Bezier control points in the thrust cone
(collective thrust in [0, c_max]: for a vertical climb, |a + g| <= c_max with a + g >= 0 automatically... checked).
Objective T = int w du (linear). Solved with IPOPT here only to check the formulation (analytic T* = 1.5703 s).

    cd experiments/2026-11-25_thrust_level
    uv run --with casadi --with scipy --with numpy python toy.py
"""
import casadi as ca
import numpy as np
from scipy.interpolate import BSpline

import bez

G = 9.81
D, V0 = 10.0, 1.0
MASS = 0.85
CMAX = 4 * 3.3 * MASS * G / 4 / MASS
DEG = 4


def solve(n):
    t = np.r_[np.zeros(DEG), np.linspace(0, 1, n + 1), np.ones(DEG)]
    nc = len(t) - DEG - 1
    W0, _ = bez.to_bezier(t, DEG, nc, 0)
    W1, _ = bez.to_bezier(t, DEG, nc, 1)
    c = ca.SX.sym("c", nc)
    g = []
    for j in range(n):
        w = [x for x in ca.vertsplit(ca.mtimes(ca.DM(W0[j]), c))]
        w1 = [x for x in ca.vertsplit(ca.mtimes(ca.DM(W1[j]), c))]
        p1 = [ca.DM([0, 0, D])] * (DEG + 1)            # p' in Bernstein form (constant), degree of w
        p2 = [ca.DM([0, 0, 0])] * DEG                  # p''
        F, w3 = bez.cone_rows(p1, p2, w, w1, G)
        for k in range(len(F)):
            g.append((CMAX * w3[k]) ** 2 - ca.sumsqr(F[k]))   # >= 0 : in the cone
            g.append(F[k][2])                                  # collective thrust >= 0 (vertical: a + g >= 0)
        g += w                                                 # w > 0 on every Bezier control point
    # objective: int w du = sum of B-spline integrals
    ints = np.array([BSpline(t, np.eye(nc)[i], DEG).integrate(0, 1) for i in range(nc)])
    T = ca.dot(ca.DM(ints), c)
    ends = ca.vertcat(c[0] - D / V0, c[-1] - D / V0)          # clamped: end coefficients = end values
    S = ca.nlpsol("s", "ipopt", {"x": c, "f": T, "g": ca.vertcat(*g, ends)},
                  {"print_time": False, "ipopt": {"print_level": 0, "max_iter": 3000}})
    ng = len(g)
    sol = S(x0=np.full(nc, D / V0), lbg=np.r_[np.zeros(ng), 0, 0], ubg=np.r_[np.full(ng, np.inf), 0, 0])
    cs = np.array(sol["x"]).ravel()
    # dense check: thrust from the exact formula a = -D w'/w^3 (vertical), only as a check
    uu = np.linspace(0, 1, 200001)
    s = BSpline(t, cs, DEG)
    w, w1 = s(uu), s.derivative()(uu)
    cth = -D * w1 / w ** 3 + G
    return float(sol["f"]), S.stats()["return_status"], cth.min(), cth.max(), w.min()


def main():
    a1 = CMAX - G
    vp = np.sqrt(V0 ** 2 + 2 * a1 * G * D / (a1 + G))
    Ts = (vp - V0) * (1 / a1 + 1 / G)
    print(f"analytic T* = {Ts:.4f} s, c_max {CMAX:.2f}")
    for n in (4, 8, 16, 32, 64):
        T, st, cmin, cmax, wmin = solve(n)
        print(f"  {n:3d} intervals: {st:22s} T {T:.4f} s ({100 * (T / Ts - 1):+.3f}%) | thrust on a dense grid "
              f"[{cmin:.3f}, {cmax:.3f}] (limits [0, {CMAX:.2f}]) | min w {wmin:.3f}")


if __name__ == "__main__":
    main()
