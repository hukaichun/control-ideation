"""Version q (user, 2026-11-25): variable Q(u) = L^2 (du/dt)^2 (periodic spline, ~ speed^2; r = Q / L^2 = (du/dt)^2).
Our control law: speed^2 / 2 = -h, so Q is -2h up to the path-speed factor |p_u|^2 / L^2 (~1, path fitted to arc length).
    f = p_uu r + p_u r_u / 2 + g e3          (linear in r)
    j = sqrt(r) f_u,   s = r f_uu + r_u f_u / 2
    |f x j|^2 = r |f x f_u|^2,   N = (f x s)(f.f) - 2 r (f x f_u)(f.f_u)      (polynomial in r; the limits then divide by f.f, see common.py)
Objective T = int_0^1 L / sqrt(Q) du: Gauss-Legendre, NG points on each speed-knot interval (not exact; checked
against 8 NG afterwards). Bounds Q >= 0 (natural domain only). Start: 2 m/s.
    uv run --with casadi --with pyyaml --with scipy --with numpy python q_version.py [track]
"""
import sys

import casadi as ca
import numpy as np

import common as C

NG = 6


def chain(L):
    """f, f_u, f_uu (u-derivatives) from p_u..p_uuuu and Q, Q_u, Q_uu, Q_uuu."""
    X = [ca.SX.sym(f"p{k}", 3) for k in range(5)]
    Rs = [ca.SX.sym(f"q{k}") for k in range(4)]
    R = [r / L ** 2 for r in Rs]
    base, rate = ca.vertcat(*X[:4], *Rs[:3]), ca.vertcat(*X[1:], *Rs[1:])
    du = lambda E: ca.jtimes(E, base, rate)
    f = X[2] * R[0] + X[1] * R[1] / 2 + ca.DM([0, 0, C.G])
    fu = du(f)
    fuu = du(fu)
    return ca.Function("chain_q", [ca.vertcat(*X[1:], *Rs)], [f, fu, fuu, R[0], R[1]])


def parts(Pa, Pd, B, z):
    Wq = [ca.mtimes(ca.DM(b), z).T for b in B]
    return chain(Pa.L).map(Pd.shape[1])(ca.vertcat(ca.DM(Pd), *Wq))


def limits(f, fu, fuu, r, ru):
    A = ca.cross(f, fu)
    s = r * fuu + ru * fu / 2
    N = ca.cross(f, s) * ca.dot(f, f) - 2 * r * A * ca.dot(f, fu)
    return C.rows(f, ca.sqrt(r) * ca.norm_2(A), N)


_a = [ca.SX.sym(nm, 3) for nm in ("f", "fu", "fuu")] + [ca.SX.sym("r"), ca.SX.sym("ru")]
LIM = ca.Function("lim_q", _a, [limits(*_a)])


def quad_rows(Pa, ng):
    x, wt = np.polynomial.legendre.leggauss(ng)
    u = ((np.arange(Pa.n)[:, None] + (x[None, :] + 1) / 2) / Pa.n).ravel()
    return C.prows(u, Pa.n, C.DEG, 0), np.tile(wt / 2, Pa.n) / Pa.n


def T_expr(Pa, z, ng=NG):
    B0, wq = quad_rows(Pa, ng)
    return ca.dot(ca.DM(wq), Pa.L / ca.sqrt(ca.mtimes(ca.DM(B0), z)))


def lap(Pa, z, ng=NG):
    B0, wq = quad_rows(Pa, ng)
    return float(wq @ (Pa.L / np.sqrt(B0 @ z)))


def start(Pa, v=2.0):
    return np.full(Pa.n, v ** 2)


def solve(Pa, z0):
    z = ca.SX.sym("z", Pa.n)
    Pd, B = Pa.at(Pa.us)
    g = LIM.map(Pa.M)(*parts(Pa, Pd, B, z))
    return C.ipopt(z, T_expr(Pa, z), ca.vec(g), z0, np.zeros(Pa.n))


def phys(Pa, z, fac):
    Pd, B = Pa.at(Pa.grid(fac))
    f, fu, fuu, r, ru = (np.array(v) for v in parts(Pa, Pd, B, z))
    return C.physical(f, np.sqrt(r) * fu, r * fuu + ru * fu / 2)


if __name__ == "__main__":
    import run
    run.main(sys.modules[__name__], "q", int(sys.argv[1]) if len(sys.argv) > 1 else 0)
